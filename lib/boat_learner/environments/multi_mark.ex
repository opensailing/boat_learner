defmodule BoatLearner.Environments.MultiMark do
  @moduledoc """
  Environment that simulates wind coming from 0 degrees (vertical).

  Accepts a list of marks to select from at each episode.
  """
  import Nx.Defn
  import Nx.Constants

  @behaviour ReinforcementLearning.Environment

  @derive {Inspect, except: [:polar_chart]}
  @derive {Nx.Container,
           keep: [],
           containers: [
             :x,
             :y,
             :target_x,
             :target_y,
             :reward,
             :is_terminal,
             :polar_chart,
             :speed,
             :angle_to_target,
             :heading,
             :remaining_seconds,
             :max_remaining_seconds,
             :vmg,
             :tack_count,
             :has_tacked,
             :coords,
             :coord_probabilities,
             :initial_x,
             :initial_y,
             :initial_distance,
             :max_tacks,
             :delta_t,
             :has_reached_target,
             :has_collided,
             :distance,
             :prev_distance
           ]}
  defstruct [
    :x,
    :y,
    :speed,
    :heading,
    :angle_to_target,
    :target_x,
    :target_y,
    :reward,
    :is_terminal,
    :polar_chart,
    :remaining_seconds,
    :max_remaining_seconds,
    :vmg,
    :tack_count,
    :has_tacked,
    :coords,
    :coord_probabilities,
    :initial_x,
    :initial_y,
    :initial_distance,
    :max_tacks,
    :delta_t,
    :has_reached_target,
    :has_collided,
    :distance,
    :prev_distance
  ]

  @min_x -400
  @max_x 400
  @min_y -400
  @max_y 400

  @one_deg_in_rad 1 * :math.pi() / 180

  @kts_to_meters_per_sec 0.514444
  @speed_kts [
    4.4,
    5.1,
    5.59,
    5.99,
    6.2,
    6.37,
    6.374,
    6.25,
    6.02,
    5.59,
    4.82,
    4.11,
    3.57,
    3.22,
    3.08
  ]

  @dead_zone_angle @one_deg_in_rad * 30

  @speed_kts [4.62, 5.02 | @speed_kts]
  @theta_deg [42.7, 137.6 | Enum.to_list(40..180//10)]
  @theta Enum.map(@theta_deg, &(&1 * :math.pi() / 180))
  @speed Enum.map(@speed_kts, &(&1 * @kts_to_meters_per_sec))
  @max_speed @speed |> Enum.max() |> ceil()

  # 68 degrees / 9 seconds = 7.5 deg/sec
  # 75 degrees / 6 seconds = 12.5 deg/sec
  # 77 degrees / 5 seconds = 15.4 deg/sec
  # 65 degrees / 5 seconds = 13 deg/sec
  # 82 degrees / 6 = 13.6 deg/sec

  # rad/second
  # turning_rates = [68 / 9, 75 / 6, 77 / 5, 65 / 5, 82 / 6]
  # @turning_rate Enum.sum(turning_rates) / length(turning_rates) * @one_deg_in_rad -> 12.5 * @one_deg_in_rad

  @turning_rate 12.5 * @one_deg_in_rad
  @iters_per_action 5
  @speed_penalty 0.4
  @speed_recovery_in_seconds 4

  def bounding_box, do: {@min_x, @max_x, @min_y, @max_y}

  # We have a single action in the interval [-1, 1]
  # that maps linearly to angles [-pi, pi]
  @impl true
  def num_actions, do: 1

  @impl true
  def init(random_key, opts) do
    opts =
      Keyword.validate!(opts, [:coords, :coord_probabilities, :max_remaining_seconds, :max_tacks])

    max_tacks = opts[:max_tacks] || raise ArgumentError, "missing option :max_tacks"
    coords = opts[:coords] || raise ArgumentError, "missing option :coords"

    coord_probabilities =
      case opts[:coord_probabilities] do
        nil ->
          raise ArgumentError, "missing option :coord_probabilities"

        :uniform ->
          Nx.broadcast(1 / Nx.size(coords), {Nx.axis_size(coords, 0)})

        %Nx.Tensor{} = tensor ->
          tensor
      end

    max_remaining_seconds =
      opts[:max_remaining_seconds] ||
        raise ArgumentError, "missing option :max_remaining_seconds"

    [coords, coord_probabilities, _] =
      Nx.broadcast_vectors([coords, coord_probabilities, random_key])

    reset(random_key, %__MODULE__{
      max_tacks: Nx.as_type(max_tacks, :s64),
      coords: Nx.as_type(coords, :f32),
      coord_probabilities: Nx.as_type(coord_probabilities, :f32),
      polar_chart: init_polar_chart(),
      max_remaining_seconds: Nx.tensor(max_remaining_seconds, type: :f32)
    })
  end

  def init_polar_chart do
    # data for the boat at TWS=6
    theta = Nx.tensor(@theta)
    speed = Nx.tensor(@speed)

    spline_model = Scholar.Interpolation.BezierSpline.fit(theta, speed)
    # use the tail-end of the spline prediction as part of our linear model
    dtheta = 0.1
    min_theta = theta |> Nx.reduce_min() |> Nx.new_axis(0)

    dspeed =
      Scholar.Interpolation.BezierSpline.predict(spline_model, Nx.subtract(min_theta, dtheta))

    # Fit {0, 0}, {15deg, 0} and {min_theta, dspeed} as points for the linear "extrapolation"
    # dead_zone_thetas = Nx.tensor([0, @dead_zone_angle])
    # dead_zone_speeds = Nx.tensor([0, 0])

    linear_model =
      Scholar.Interpolation.Linear.fit(
        Nx.concatenate([Nx.tensor([0]), min_theta, theta]),
        # Nx.concatenate([dead_zone_thetas, min_theta, theta]),
        Nx.concatenate([Nx.tensor([0]), dspeed, speed])
        # Nx.concatenate([dead_zone_speeds, dspeed, speed])
      )

    cutoff_angle = Nx.reduce_min(spline_model.k, axes: [0])[0]

    {linear_model, spline_model, cutoff_angle}
  end

  @impl true
  def reset(random_key, state) do
    zero = Nx.tensor(0, type: :f32)
    vmg = speed = reward = delta_t = zero

    # {heading, random_key} = Nx.Random.uniform(random_key, -:math.pi(), :math.pi())

    {heading, random_key} =
      Nx.Random.choice(random_key, Nx.tensor([:math.pi() / 4]),
        samples: 1,
        axis: 0
      )

    heading = Nx.squeeze(heading)

    {coords, random_key} =
      Nx.Random.choice(random_key, state.coords, state.coord_probabilities, samples: 1, axis: 0)

    heading = wrap_phase(heading)

    initial_x = x = coords[[0, 0]]
    initial_y = y = coords[[0, 1]]
    target_x = coords[[0, 2]]
    target_y = coords[[0, 3]]

    initial_distance =
      x
      |> Nx.subtract(target_x)
      |> Nx.pow(2)
      |> Nx.add(Nx.pow(Nx.subtract(y, target_y), 2))
      |> Nx.sqrt()

    state = %__MODULE__{
      state
      | x: x,
        y: y,
        target_x: target_x,
        target_y: target_y,
        initial_x: initial_x,
        initial_y: initial_y,
        initial_distance: initial_distance,
        heading: heading,
        angle_to_target: heading,
        speed: speed,
        reward: reward,
        is_terminal: Nx.u8(0),
        has_reached_target: Nx.u8(0),
        has_tacked: Nx.u8(0),
        has_collided: Nx.u8(0),
        remaining_seconds: state.max_remaining_seconds,
        vmg: vmg,
        tack_count: Nx.s64(0),
        delta_t: delta_t,
        distance: initial_distance,
        prev_distance: initial_distance
    }

    state = %__MODULE__{state | distance: distance(state), prev_distance: state.distance}

    case random_key.vectorized_axes do
      [] ->
        {state, random_key}

      _ ->
        state =
          state
          |> Map.from_struct()
          |> Map.keys()
          |> Kernel.--([:polar_chart, :action_lower_limit, :action_upper_limit])
          |> Enum.reduce(state, fn field, state ->
            Map.update(state, field, Map.fetch!(state, field), fn value ->
              [value, _] = Nx.broadcast_vectors([value, random_key], align_ranks: false)
              value
            end)
          end)

        {state, random_key}
    end
  end

  @impl true
  defn apply_action(rl_state, action) do
    %__MODULE__{} = env = rl_state.environment_state

    action = Nx.reshape(action, {})

    next_env =
      env
      |> turn_and_move(action * pi() / 2)
      |> is_terminal_state()
      |> calculate_reward()

    %ReinforcementLearning{rl_state | environment_state: next_env}
  end

  defn turn_and_move(env, dtheta) do
    prev_heading = env.heading

    turning_time = Nx.abs(dtheta) / @turning_rate
    dt = turning_time / @iters_per_action

    dtheta_steps = dt * Nx.sign(dtheta) * Nx.broadcast(@turning_rate, {@iters_per_action})
    heading_steps = Nx.cumulative_sum(dtheta_steps) + prev_heading

    heading_steps = wrap_phase(heading_steps)

    # Calculate the speed and apply the speed penalty
    speed_steps = speed_from_heading(env.polar_chart, heading_steps)

    # because we are normalizing angles to be between 0 and 2pi,
    # this check is equivalent to checking for 0 crossings if
    # the angles were normalized between -pi and pi

    # 1 if that position has tacked since the beginning
    tacking_mask = heading_steps < pi() != prev_heading < pi()
    tacking_mask = Nx.cumulative_max(tacking_mask)

    speed_penalty_multiplier = 1 - @speed_penalty

    penalized_speed_steps =
      Nx.select(tacking_mask, speed_penalty_multiplier * speed_steps, speed_steps)

    has_tacked = Nx.any(tacking_mask)
    tack_count = env.tack_count + has_tacked

    # Calculate the position changes in x and y directions for each interval
    dy = dt * Nx.cos(heading_steps) * penalized_speed_steps
    dx = dt * Nx.sin(heading_steps) * penalized_speed_steps

    x = env.x + Nx.sum(dx)
    y = env.y + Nx.sum(dy)

    heading = Nx.take(heading_steps, Nx.axis_size(heading_steps, 0) - 1)
    heading = wrap_phase(heading)
    speed = Nx.take(speed_steps, Nx.axis_size(speed_steps, 0) - 1)

    # Recover the speed over `@speed_recovery_in_seconds` seconds
    speed_steps =
      Nx.linspace(speed, speed / speed_penalty_multiplier, n: @speed_recovery_in_seconds)

    # dx = Nx.sin(heading_steps[0]) * speed_steps[0] + Nx.sin(heading_steps[1]) * speed_steps[1] + ... + Nx.sin(heading_steps[n-1]) * speed_steps[n-1]
    # because the heading won't change, the calculation can be simplified:
    # dx = Nx.sin(heading) * speed_steps[0] + Nx.sin(heading) * speed_steps[1] + ... + Nx.sin(heading) * speed_steps[n-1]
    # dx = Nx.sin(heading) * (speed_steps[0] + speed_steps[1] + ... + speed_steps[-1])
    # dx = Nx.sin(heading) * Nx.sum(speed_steps), and likewise for dy, changing sin for cos

    x = x + Nx.sin(heading) * Nx.sum(speed_steps)
    y = y + Nx.cos(heading) * Nx.sum(speed_steps)

    speed = Nx.take(speed_steps, Nx.axis_size(speed_steps, 0) - 1)

    dx = env.target_x - x
    dy = env.target_y - y

    angle_to_target = wrap_phase(Nx.atan2(dx, dy))

    target_unit =
      Nx.stack([
        Nx.cos(angle_to_target),
        Nx.sin(angle_to_target)
      ])

    heading_unit =
      Nx.stack([
        Nx.cos(heading),
        Nx.sin(heading)
      ])

    vmg = Nx.dot(target_unit, heading_unit) * speed

    delta_t = turning_time + @speed_recovery_in_seconds

    env = %__MODULE__{
      env
      | remaining_seconds:
          Nx.select(
            env.is_terminal,
            env.remaining_seconds,
            Nx.max(env.remaining_seconds - delta_t, 0)
          ),
        delta_t: delta_t,
        heading: Nx.select(env.is_terminal, env.heading, heading),
        speed: Nx.select(env.is_terminal, env.speed, speed),
        tack_count: Nx.select(env.is_terminal, env.tack_count, tack_count),
        has_tacked: Nx.select(env.is_terminal, env.has_tacked, has_tacked),
        vmg: Nx.select(env.is_terminal, env.vmg, vmg),
        x: Nx.select(env.is_terminal, env.x, x),
        y: Nx.select(env.is_terminal, env.y, y),
        angle_to_target: Nx.select(env.is_terminal, env.angle_to_target, angle_to_target)
    }

    %{env | distance: distance(env), prev_distance: env.distance}
  end

  defn speed_from_heading({linear_model, spline_model, cutoff_angle}, angle) do
    # angle is already maintained between 0 and 2pi
    # so we only need to calculate the "absolute value"
    # (as if the angle was between -pi and pi)
    angle = Nx.select(angle > pi(), 2 * pi() - angle, angle)

    linear_pred = Scholar.Interpolation.Linear.predict(linear_model, angle)

    spline_pred =
      Scholar.Interpolation.BezierSpline.predict(
        spline_model,
        angle |> Nx.devectorize() |> Nx.flatten()
      )
      |> Nx.reshape(Nx.devectorize(angle) |> Nx.shape())
      |> Nx.vectorize(angle.vectorized_axes)

    (angle <= cutoff_angle)
    |> Nx.select(linear_pred, spline_pred)
    |> Nx.clip(0, @max_speed)
  end

  defnp is_terminal_state(env) do
    %__MODULE__{
      x: x,
      y: y,
      remaining_seconds: remaining_seconds,
      tack_count: tack_count,
      max_tacks: max_tacks
    } = env

    has_reached_target = has_reached_target(env)
    has_collided = x < @min_x or x > @max_x or y < @min_y or y > @max_y

    # is_terminal = env.is_terminal or has_reached_target or has_collided or remaining_seconds < 1

    %__MODULE__{
      env
      | has_reached_target: has_reached_target,
        has_collided: has_collided,
        is_terminal: has_reached_target or remaining_seconds <= 1
    }
  end

  defnp has_reached_target(env) do
    env.distance < 10
  end

  defnp calculate_reward(env) do
    %__MODULE__{
      has_reached_target: has_reached_target,
      has_tacked: has_tacked,
      is_terminal: is_terminal,
      distance: distance,
      initial_distance: initial_distance
    } = env

    [one, _] = Nx.broadcast_vectors([1, is_terminal])

    reward =
      cond do
        has_reached_target ->
          one

        is_terminal ->
          -distance / initial_distanc e

        has_tacked ->
          -0.1 * one

        true ->
          -0.01 * one
      end

    %__MODULE__{env | reward: reward}
  end

  defnp distance(env) do
    Nx.sqrt((env.target_y - env.y) ** 2 + (env.target_x - env.x) ** 2)
  end

  defnp wrap_phase(angle) do
    angle
    |> Nx.remainder(2 * pi())
    |> Nx.add(2 * pi())
    |> Nx.remainder(2 * pi())
  end
end
