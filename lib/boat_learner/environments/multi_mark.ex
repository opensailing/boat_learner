defmodule BoatLearner.Environments.MultiMark do
  @moduledoc """
  Environment that simulates wind coming from 0 degrees (vertical).

  Accepts a list of marks to select from at each episode.
  """
  import Nx.Defn
  import Nx.Constants

  alias Rein.Utils.CircularBuffer

  @behaviour Rein.Environment

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
             :angle_to_mark,
             :heading,
             :remaining_seconds,
             :max_remaining_seconds,
             :vmg,
             :tack_count,
             :has_tacked,
             :coords,
             :coord_probabilities,
             :has_reached_target,
             :has_collided,
             :distance,
             :delta_t,
             :penalty_queue
           ]}
  defstruct [
    :x,
    :y,
    :speed,
    :heading,
    :angle_to_mark,
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
    :has_reached_target,
    :has_collided,
    :distance,
    :delta_t,
    :penalty_queue
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
  # dt in seconds
  @dt 5

  # 68 degrees / 9 seconds = 7.5 deg/sec
  # 75 degrees / 6 seconds = 12.5 deg/sec
  # 77 degrees / 5 seconds = 15.4 deg/sec
  # 65 degrees / 5 seconds = 13 deg/sec
  # 82 degrees / 6 = 13.6 deg/sec

  # rad/second
  # turning_rates = [68 / 9, 75 / 6, 77 / 5, 65 / 5, 82 / 6]
  # @turning_rate Enum.sum(turning_rates) / length(turning_rates) * @one_deg_in_rad -> 12.5 * @one_deg_in_rad
  @turning_rate 12.5 * @one_deg_in_rad

  def bounding_box, do: {@min_x, @max_x, @min_y, @max_y}

  # We have a single action in the interval [-1, 1]
  # that maps linearly to angles [-pi, pi]
  def num_actions, do: 1

  @impl true
  def init(random_key, opts) do
    opts =
      Keyword.validate!(opts, [:coords, :coord_probabilities, :max_remaining_seconds])

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

    # Fit {0, 0}, {@dead_zone_angle, 0} and {min_theta, dspeed} as points for the linear "extrapolation"
    dead_zone_thetas = Nx.tensor([0, @dead_zone_angle])
    dead_zone_speeds = Nx.tensor([0, 0])

    linear_model =
      Scholar.Interpolation.Linear.fit(
        Nx.concatenate([dead_zone_thetas, min_theta, theta]),
        Nx.concatenate([dead_zone_speeds, dspeed, speed])
      )

    cutoff_angle = Nx.reduce_min(spline_model.k, axes: [0])[0]

    {linear_model, spline_model, cutoff_angle}
  end

  @impl true
  def reset(random_key, state) do
    zero = Nx.tensor(0, type: :f32)
    distance = vmg = speed = reward = delta_t = zero

    {coords, random_key} =
      Nx.Random.choice(random_key, state.coords, state.coord_probabilities, samples: 1, axis: 0)

    x = coords[[0, 0]]
    y = coords[[0, 1]]
    target_x = coords[[0, 2]]
    target_y = coords[[0, 3]]

    heading = Nx.subtract(target_x, x) |> Nx.atan2(Nx.subtract(target_y, y)) |> wrap_phase()

    state = %__MODULE__{
      state
      | x: x,
        y: y,
        target_x: target_x,
        target_y: target_y,
        heading: heading,
        angle_to_mark: heading,
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
        distance: distance,
        penalty_queue:
          CircularBuffer.new({3},
            init_value:
              Nx.iota({1}, vectorized_axes: random_key.vectorized_axes, type: Nx.type(heading))
          )
    }

    case random_key.vectorized_axes do
      [] ->
        {state, random_key}

      _ ->
        state =
          state
          |> Map.from_struct()
          |> Map.keys()
          |> Kernel.--([:polar_chart, :penalty_queue, :action_lower_limit, :action_upper_limit])
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
      |> turn_and_move(action)
      |> calculate_reward()
      |> is_terminal_state()

    %Rein{rl_state | environment_state: next_env}
  end

  defn turn_and_move(env, action) do
    desired_heading_change = action * Nx.Constants.pi()
    prev_heading = env.heading

    max_heading_change = @turning_rate * @dt
    actual_heading_change = Nx.min(Nx.abs(desired_heading_change), max_heading_change)
    actual_heading_change = Nx.select(action >= 0, actual_heading_change, -actual_heading_change)
    heading = wrap_phase(env.heading + actual_heading_change)

    penalty = calculate_momentum_penalty(actual_heading_change)
    penalty_queue = CircularBuffer.append(env.penalty_queue, penalty)

    cumulative_penalty = env.penalty_queue.data |> Nx.sum(axes: [0]) |> Nx.min(1)

    speed = speed_from_heading(env.polar_chart, heading) * (1 - cumulative_penalty)

    x = env.x + Nx.sin(heading) * speed * @dt
    y = env.y + Nx.cos(heading) * speed * @dt

    has_tacked = prev_heading < Nx.Constants.pi() != heading < Nx.Constants.pi()

    dx = env.target_x - x
    dy = env.target_y - y

    angle_to_mark = wrap_phase(Nx.atan2(dx, dy))
    target_unit = Nx.stack([Nx.sin(angle_to_mark), Nx.cos(angle_to_mark)], axis: 0)
    heading_unit = Nx.stack([Nx.sin(heading), Nx.cos(heading)], axis: 0)

    vmg = Nx.dot(target_unit, heading_unit) * speed

    distance = Nx.sqrt(dx ** 2 + dy ** 2)
    tack_count = Nx.as_type(has_tacked, Nx.type(env.tack_count))
    remaining_seconds = env.remaining_seconds - @dt
    delta_t = @dt

    %__MODULE__{
      env
      | penalty_queue: penalty_queue,
        speed: speed,
        x: x,
        y: y,
        heading: heading,
        has_tacked: has_tacked,
        angle_to_mark: angle_to_mark,
        vmg: vmg,
        distance: distance,
        tack_count: tack_count,
        remaining_seconds: remaining_seconds,
        delta_t: delta_t
    }
  end

  defnp calculate_momentum_penalty(actual_heading_change) do
    Nx.abs(actual_heading_change) / Nx.Constants.pi()
  end

  defnp speed_from_heading({linear_model, spline_model, cutoff_angle}, angle) do
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
      remaining_seconds: remaining_seconds
    } = env

    has_reached_target = env.distance < 5
    has_collided = x < @min_x or x > @max_x or y < @min_y or y > @max_y

    is_terminal = env.is_terminal or has_reached_target or has_collided or remaining_seconds < 1

    %__MODULE__{
      env
      | has_reached_target: has_reached_target,
        has_collided: has_collided,
        is_terminal: is_terminal
    }
  end

  defnp calculate_reward(env) do
    %__MODULE__{
      vmg: vmg
    } = env

    reward = 0.1 * vmg

    %__MODULE__{env | reward: reward}
  end

  defnp wrap_phase(angle) do
    angle
    |> Nx.remainder(2 * pi())
    |> Nx.add(2 * pi())
    |> Nx.remainder(2 * pi())
  end
end
