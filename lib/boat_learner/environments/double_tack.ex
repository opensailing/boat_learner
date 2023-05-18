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

    # Fit {0, 0}, {15deg, 0} and {min_theta, dspeed} as points for the linear "extrapolation"
    dead_zone_thetas = Nx.tensor([0, 5 * :math.pi() / 180, 15 * :math.pi() / 180])
    dead_zone_speeds = Nx.tensor([0, 0, 1])

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
      target_y: target_y,
      tack_count: tack_count
    } = env

    is_terminal =
      has_reached_target(env) or x < @min_x or x > @max_x or y < @min_y or y > target_y or
        remaining_seconds < 1 or tack_count > 2

    %__MODULE__{env | is_terminal: is_terminal}
  end

  defnp has_reached_target(env) do
    Nx.sqrt((env.target_y - env.y) ** 2 + env.x ** 2) < 10
  end

  defnp calculate_reward(env) do
    %__MODULE__{
      is_terminal: is_terminal,
      vmg: vmg,
      remaining_seconds: remaining_seconds,
      max_remaining_seconds: max_remaining_seconds,
      # target_y: target_y,
      # y: y,
      # x: x,
      has_tacked: has_tacked
    } = env

    time_decay = Nx.exp(-(max_remaining_seconds - remaining_seconds) / 50)

    vmg_dead_zone_max_angle = 15 * pi() / 180
    vmg_dead_zone_max = speed_from_heading(env.polar_chart, vmg_dead_zone_max_angle)

    reward =
      if not is_terminal and vmg >= 0 and vmg < vmg_dead_zone_max do
        # dead-zone penalty: if the boat is in the vmg dead-zone it means that we're
        # on the verge of reaching stationarity, so we want to not reward the agent in this case.

        # at the beggining of the dead-zone, we get reward := -0.1,
        # and at the end of the dead-zone (at 0deg), we get reward := -0.2
        -0.1 * (2 - vmg / vmg_dead_zone_max)
      else
        vmg_reward = time_decay * (vmg / @max_speed - (1 - is_terminal) * 2 * has_tacked)

        # reaching the target will provide the full score regardless of time decay,
        # each iteration will receive a time-decayed score instead
        distance_reward =
          Nx.select(
            has_reached_target(env),
            10 * time_decay,
            0.1 * distance_decay(env) * time_decay
          )

        vmg_reward + distance_reward
      end

    %__MODULE__{env | reward: reward}
  end

  defnp distance_decay(env) do
    r1 = env.target_y * 0.8

    r_sq = (env.target_y - env.y) ** 2 + env.x ** 2
    r_sq = r_sq / r1 ** 2

    x = r_sq / r1 ** 2
    f = Nx.cos(x * pi() / 2)

    Nx.select(x >= 1, 0, f)
  end

  defnp wrap_phase(angle) do
    angle
    |> Nx.remainder(2 * pi())
    |> Nx.add(2 * pi())
    |> Nx.remainder(2 * pi())
  end
end
