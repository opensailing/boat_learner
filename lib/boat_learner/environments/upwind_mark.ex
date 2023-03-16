defmodule BoatLearner.Environments.UpwindMark do
  @moduledoc """
  Simple environment that provides an upwind mark
  and simulates the physics for wind at 0 degrees.
  """
  import Nx.Defn
  import Nx.Constants

  @behaviour ReinforcementLearning.Environment

  @derive {Nx.Container,
           keep: [],
           containers: [
             :x,
             :y,
             :prev_x,
             :prev_y,
             :target_x,
             :target_y,
             :reward,
             :is_terminal,
             :possible_targets,
             :polar_chart,
             :angle_memory,
             :speed,
             :prev_speed,
             :angle
           ]}
  defstruct [
    :x,
    :y,
    :prev_x,
    :prev_y,
    :speed,
    :prev_speed,
    :angle,
    :target_x,
    :target_y,
    :reward,
    :is_terminal,
    :possible_targets,
    :polar_chart,
    :angle_memory
  ]

  @min_x -50
  @max_x 50
  @min_y 0
  @max_y 100

  @angle 5 * :math.pi() / 180
  @angle_memory_num_entries 5

  def bounding_box, do: {@min_x, @max_x, @min_y, @max_y}

  @impl true
  def num_actions, do: 3

  @impl true
  def init(random_key, opts) do
    opts = Keyword.validate!(opts, [:possible_targets])

    possible_targets =
      opts[:possible_targets] || raise ArgumentError, "missing option :possible_targets"

    reset(random_key, %__MODULE__{
      possible_targets: possible_targets,
      polar_chart: init_polar_chart()
    })
  end

  @kts_to_meters_per_sec 0.514444
  defp init_polar_chart do
    # data for the boat at TWS=6
    speed_kts = [
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

    speed_kts = [4.62, 5.02 | speed_kts]
    theta_deg = [42.7, 137.6 | Enum.to_list(40..180//10)]

    theta = Enum.map(theta_deg, &(&1 * :math.pi() / 180))
    speed = Enum.map(speed_kts, &(&1 * @kts_to_meters_per_sec))

    spline_model = Scholar.Interpolation.BezierSpline.fit(theta, speed)
    # use the tail-end of the spline prediction as part of our linear model
    dtheta = 0.1
    min_theta = Nx.new_axis(dtheta + Nx.reduce_min(theta), 0)

    dspeed = Scholar.Interpolation.BezierSpline.predict(spline_model, min_theta + dtheta)

    # Fit {0, 0} and {min_theta, dspeed} as points for the linear "extrapolation"
    zero = Nx.new_axis(0, 0)

    linear_model =
      Scholar.Interpolation.Linear.fit(
        Nx.concatenate([zero, min_theta, theta]),
        Nx.concatenate([zero, dspeed, speed])
      )

    cutoff_angle = Nx.reduce_min(spline_model.k, axes: [0])[0]

    {linear_model, spline_model, cutoff_angle}
  end

  @impl true
  def reset(random_key, state) do
    zero = Nx.tensor(0, type: :f32)
    angle = speed = x = y = reward = zero

    angle_memory = Nx.broadcast(zero, {@angle_memory_num_entries})

    # possible_targets is a {n, 2} tensor that contains targets that we want to sample from
    # this is so we avoid retraining every episode on the same target, which can lead to
    # overfitting
    {target, random_key} =
      Nx.Random.choice(random_key, state.possible_targets, samples: 1, axis: 0)

    target = Nx.reshape(target, {2})

    state = %__MODULE__{
      state
      | x: x,
        y: y,
        angle: angle,
        angle_memory: angle_memory,
        speed: speed,
        prev_speed: speed,
        prev_x: x,
        prev_y: y,
        target_x: target[0],
        target_y: target[1],
        reward: reward,
        is_terminal: Nx.tensor(0, type: :u8)
    }

    {state, random_key}
  end

  @impl true
  defn apply_action(rl_state, action) do
    %__MODULE__{} = env = rl_state.environment_state

    # 0: turn left, 1: keep heading, 2: turn right
    new_env =
      cond do
        action == 0 ->
          turn_and_move(env, -@angle)

        action == 1 ->
          move(env)

        true ->
          turn_and_move(env, @angle)
      end

    new_env =
      new_env
      |> update_angle_memory()
      |> is_terminal_state()
      |> calculate_reward()

    %ReinforcementLearning{rl_state | environment_state: new_env}
  end

  defnp turn_and_move(env, angle_inc) do
    angle = env.angle + angle_inc
    two_pi = 2 * pi()
    angle = Nx.select(angle >= two_pi, angle - two_pi, angle)
    angle = Nx.select(angle < 0, angle + two_pi, angle)
    move(%__MODULE__{env | angle: angle})
  end

  defnp move(env) do
    %__MODULE__{angle: angle, x: x, y: y, polar_chart: polar_chart} = env
    speed = speed_from_angle(polar_chart, angle)

    x = x + speed * Nx.sin(angle)
    y = y + speed * Nx.cos(angle)

    %__MODULE__{
      env
      | x: x,
        y: y,
        prev_x: env.x,
        prev_y: env.y,
        speed: speed,
        prev_speed: env.speed
    }
  end

  defnp speed_from_angle({linear_model, spline_model, cutoff_angle}, angle) do
    # angle is already maintained between 0 and 2pi
    # so we only need to calculate the "absolute value"
    # (as if the angle was between -pi and pi)
    angle = Nx.select(angle > pi(), 2 * pi() - angle, angle)

    linear_pred = Scholar.Interpolation.Linear.predict(linear_model, angle)
    spline_pred = Scholar.Interpolation.BezierSpline.predict(spline_model, angle)

    Nx.select(angle <= cutoff_angle, linear_pred, spline_pred)
  end

  defnp update_angle_memory(%__MODULE__{angle_memory: angle_memory, angle: angle} = env) do
    angle_memory = Nx.concatenate([angle_memory[1..-1//1], Nx.reshape(angle, {1})])
    %__MODULE__{env | angle_memory: angle_memory}
  end

  defnp is_terminal_state(env) do
    %__MODULE__{x: x, y: y} = env

    has_reached_target(env) or x < @min_x or x > @max_x or y < @min_y or y > @max_y
  end

  defnp has_reached_target(env) do
    %__MODULE__{x: x, y: y, target_x: target_x, target_y: target_y} = env

    distance(x, y, target_x, target_y) < 1.5
  end

  defnp distance(x, y, target_x, target_y) do
    Nx.sqrt((x - target_x) ** 2 + (y - target_y) ** 2)
  end

  defnp calculate_reward(env) do
    %__MODULE__{
      x: x,
      y: y,
      speed: speed,
      prev_speed: prev_speed,
      target_x: target_x,
      target_y: target_y,
      angle_memory: angle_memory
    } = env

    distance = distance(x, y, target_x, target_y)

    # reward will be less negative if the distance is smaller
    reward = -distance

    # penalize for decreasing speed
    reward = Nx.select(speed < prev_speed, reward - 0.5, reward)

    # penalize for changing angles too much
    has_changed_angles = Nx.abs(angle_memory[0..-2//1] - angle_memory[1..-1//1]) > @angle / 2
    angle_change_penalty = Nx.sum(has_changed_angles) * 0.1
    reward = reward - angle_change_penalty

    reward
  end
end
