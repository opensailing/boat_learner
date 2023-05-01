defmodule BoatLearner.Environments.DoubleTack do
  @moduledoc """
  Simple environment that provides an upwind mark
  and simulates the physics for wind at 0 degrees.
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
             :has_tacked
           ]}
  defstruct [
    :x,
    :y,
    :speed,
    :heading,
    :angle_to_target,
    :target_y,
    :reward,
    :is_terminal,
    :polar_chart,
    :remaining_seconds,
    :max_remaining_seconds,
    :vmg,
    :tack_count,
    :has_tacked
  ]

  @min_x -125
  @max_x 125
  @min_y -25
  @max_y 125

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
  @iters_per_action 10
  @speed_penalty 0.4
  @speed_recovery_in_seconds 4

  def bounding_box, do: {@min_x, @max_x, @min_y, @max_y}

  # We have a single action in the interval [-1, 1]
  # that maps linearly to angles [-pi, pi]
  @impl true
  def num_actions, do: 1

  @impl true
  def init(random_key, opts) do
    opts = Keyword.validate!(opts, [:target_y, :max_remaining_seconds])

    target_y = opts[:target_y] || raise ArgumentError, "missing option :target_y"

    max_remaining_seconds =
      opts[:max_remaining_seconds] ||
        raise ArgumentError, "missing option :max_remaining_seconds"

    reset(random_key, %__MODULE__{
      target_y: target_y,
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

    dspeed = Scholar.Interpolation.BezierSpline.predict(spline_model, Nx.add(min_theta, dtheta))

    # Fit {0, 0}, {35deg, 0} and {min_theta, dspeed} as points for the linear "extrapolation"
    zero = Nx.new_axis(0, 0)

    linear_model =
      Scholar.Interpolation.Linear.fit(
        Nx.concatenate([zero, Nx.tensor([35 * :math.pi() / 180]), min_theta, theta]),
        Nx.concatenate([zero, zero, dspeed, speed])
      )

    cutoff_angle = Nx.reduce_min(spline_model.k, axes: [0])[0]

    {linear_model, spline_model, cutoff_angle}
  end

  @impl true
  def reset(random_key, state) do
    zero = Nx.tensor(0, type: :f32)
    vmg = speed = reward = zero

    x = zero
    y = zero

    {heading, random_key} =
      Nx.Random.uniform(
        random_key,
        -:math.pi() / 2 + 15 * @one_deg_in_rad,
        :math.pi() / 2 - 15 * @one_deg_in_rad
      )

    heading = wrap_phase(heading)

    state = %__MODULE__{
      state
      | x: x,
        y: y,
        heading: heading,
        angle_to_target: heading,
        speed: speed,
        reward: reward,
        is_terminal: Nx.tensor(0, type: :u8),
        remaining_seconds: state.max_remaining_seconds,
        vmg: vmg,
        tack_count: Nx.tensor(0, type: :s64),
        has_tacked: Nx.tensor(0, type: :u8)
    }

    {state, random_key}
  end

  # defnp calculate_elapsed_seconds_from_origin(state) do
  #   # calculate the linear distance from the origin so we can estimate
  #   # a penalty in elapsed time so that the initial reward is estimated more properly.
  #   linear_distance = Nx.sqrt(state.x ** 2 + state.y ** 2)
  #   # if speed is 0, we will instead use a default of @max_speed * 0.5
  #   avg_time = linear_distance / Nx.select(state.speed, state.speed, 0.5 * @max_speed)

  #   %{state | remaining_seconds: state.remaining_seconds - avg_time}
  # end

  @impl true
  defn apply_action(rl_state, action) do
    %__MODULE__{} = env = rl_state.environment_state

    action = Nx.reshape(action, {})

    new_env =
      env
      |> turn_and_move(action * 0.6 * pi())
      |> is_terminal_state()
      |> calculate_reward()

    %ReinforcementLearning{rl_state | environment_state: new_env}
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

    target_x = 0
    dx = target_x - x
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

    %__MODULE__{
      env
      | remaining_seconds:
          Nx.max(env.remaining_seconds - (turning_time + @speed_recovery_in_seconds), 0),
        heading: heading,
        speed: speed,
        tack_count: tack_count,
        has_tacked: has_tacked,
        vmg: vmg,
        x: x,
        y: y,
        angle_to_target: angle_to_target
    }
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
      target_y: target_y
    } = env

    is_terminal =
      has_reached_target(env) or x < @min_x or x > @max_x or y < @min_y or y > target_y or
        remaining_seconds < 2

    %__MODULE__{env | is_terminal: is_terminal}
  end

  defnp has_reached_target(env) do
    # has reached if distance < 10
    distance_sq = (env.target_y - env.y) ** 2 + env.x ** 2

    # distance_sq < 15 ** 2
    distance_sq < 225
  end

  defnp calculate_reward(env) do
    %__MODULE__{
      is_terminal: is_terminal,
      vmg: vmg,
      remaining_seconds: remaining_seconds,
      max_remaining_seconds: max_remaining_seconds,
      target_y: target_y,
      y: y,
      x: x,
      has_tacked: has_tacked
    } = env

    has_reached_target = has_reached_target(env)

    x = Nx.devectorize(x)
    y = Nx.devectorize(y)
    vmg = Nx.devectorize(vmg)
    is_terminal = Nx.devectorize(is_terminal)
    remaining_seconds = Nx.devectorize(remaining_seconds)
    has_tacked = Nx.devectorize(has_tacked)
    vec = has_reached_target.vectorized_axes
    has_reached_target = Nx.devectorize(has_reached_target)

    time_decay = remaining_seconds / max_remaining_seconds

    reward =
      case Nx.shape(has_reached_target) do
        {} ->
          cond do
            has_reached_target ->
              time_decay

            is_terminal ->
              distance = Nx.sqrt(x ** 2 + (y - target_y) ** 2)
              m = -1 / target_y
              b = 1

              # Normalize the distance to the range [-1, 1],
              # such that initial_distance maps to 0 and 0 maps to 1,
              # and then clip-off negative rewards
              distance_reward =
                if vmg < 0 do
                  Nx.clip(m * distance + b, -1, 1)
                else
                  # we want to not penalize too much if
                  # we were at least heading in the right direction
                  Nx.clip(m * distance + b, -0.1, 1)
                end

              Nx.select(distance_reward > 0, distance_reward * time_decay, distance_reward)

            true ->
              # penalize tacks in the iteration where they happened only
              # this should also help with avoiding loops since they include 2 tacks
              0.01 * (vmg / @max_speed - 2 * has_tacked) * time_decay
          end

        _ ->
          # this needs vectorized cond so that we can skip duplicating and iterating
          {out, _, _, _, _, _, _, _, _} =
            while {out = Nx.broadcast(0.0, has_reached_target), is_terminal, has_reached_target,
                   x, y, target_y, vmg, time_decay, has_tacked},
                  i <- 1..Nx.axis_size(has_reached_target, 0),
                  unroll: 2 do
              rew =
                cond do
                  has_reached_target[i] ->
                    100 * time_decay[i]

                  is_terminal[i] ->
                    distance = Nx.sqrt(x[i] ** 2 + (y[i] - target_y) ** 2)
                    m = -1 / target_y
                    b = 1

                    # Normalize the distance to the range [-1, 1],
                    # such that initial_distance maps to 0 and 0 maps to 1,
                    # and then clip-off negative rewards
                    distance_reward =
                      if vmg[i] < 0 do
                        Nx.clip(m * distance + b, -1, 1)
                      else
                        # we want to not penalize too much if
                        # we were at least heading in the right direction
                        Nx.clip(m * distance + b, -0.1, 1)
                      end

                    Nx.select(
                      distance_reward > 0,
                      distance_reward * time_decay[i],
                      distance_reward
                    )

                  true ->
                    # penalize tacks in the iteration where they happened only
                    # this should also help with avoiding loops since they include 2 tacks
                    0.01 * (vmg[i] / @max_speed - 2 * has_tacked[i]) * time_decay[i]
                end

              {Nx.indexed_put(out, Nx.reshape(i, {1, 1}), Nx.reshape(rew, {1})), is_terminal,
               has_reached_target, x, y, target_y, vmg, time_decay, has_tacked}
            end

          Nx.vectorize(out, vec)
      end

    %__MODULE__{env | reward: reward}
  end

  defnp wrap_phase(angle) do
    angle
    |> Nx.remainder(2 * pi())
    |> Nx.add(2 * pi())
    |> Nx.remainder(2 * pi())
  end
end
