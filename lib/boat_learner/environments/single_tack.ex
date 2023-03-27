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
             :polar_chart,
             :speed,
             :prev_speed,
             :heading,
             :remaining_iterations,
             :max_remaining_iterations,
             :vmg,
             :has_moved
           ]}
  defstruct [
    :x,
    :y,
    :prev_x,
    :prev_y,
    :speed,
    :prev_speed,
    :heading,
    :target_y,
    :reward,
    :is_terminal,
    :polar_chart,
    :remaining_iterations,
    :max_remaining_iterations,
    :vmg,
    :has_moved
  ]

  @min_x -1250
  @max_x 1250
  @min_y 0
  @max_y 2500

  @delta_angle 1 * :math.pi() / 180

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

  @dt 5

  def bounding_box, do: {@min_x, @max_x, @min_y, @max_y}

  @impl true
  def num_actions, do: 3

  @impl true
  def init(random_key, opts) do
    opts = Keyword.validate!(opts, [:target_y, :max_remaining_iterations])

    target_y = opts[:target_y] || raise ArgumentError, "missing option :target_y"

    max_remaining_iterations = opts[:max_remaining_iterations] || raise ArgumentError, "missing option :max_remaining_iterations"

    reset(random_key, %__MODULE__{
      target_y: target_y,
      polar_chart: init_polar_chart(),
      max_remaining_iterations: Nx.tensor(max_remaining_iterations, type: :f32)
    })
  end

  defp init_polar_chart do
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
    vmg = speed = x = reward = zero

    {heading, random_key} = Nx.Random.uniform(random_key, 0, :math.pi() / 2 - @delta_angle)

    y = Nx.tensor(1, type: :f32)

    state = %__MODULE__{
      state
      | x: x,
        y: y,
        heading: heading,
        speed: speed,
        prev_speed: speed,
        prev_x: x,
        prev_y: y,
        reward: reward,
        is_terminal: Nx.tensor(0, type: :u8),
        has_moved: Nx.tensor(0, type: :u8),
        remaining_iterations: state.max_remaining_iterations,
        vmg: vmg
    }

    {state, random_key}
  end

  @impl true
  defn apply_action(rl_state, action) do
    %__MODULE__{} = env = rl_state.environment_state

    # 0: turn left, 1: keep heading, 2: turn right
    new_env =
      cond do
        not has_moved and action == 0 ->
          turn(env, -@delta_angle)

        not has_moved and action == 1 ->
          turn(env, @delta_angle)

        true ->
          move(env)
      end

    new_env =
      new_env
      |> is_terminal_state()
      |> calculate_reward()

    %ReinforcementLearning{rl_state | environment_state: new_env}
  end

  defnp turn(env, angle_inc, remaining_iterations_penalty \\ 1) do
    heading = env.heading + angle_inc
    two_pi = 2 * pi()
    heading = Nx.select(heading >= two_pi, heading - two_pi, heading)
    heading = Nx.select(heading < 0, heading + two_pi, heading)
    %__MODULE__{env | remaining_iterations: env.remaining_iterations - remaining_iterations_penalty, heading: heading}
  end

  defnp move(env) do
    %__MODULE__{
      heading: heading,
      target_y: target_y,
      x: x,
      y: y,
      polar_chart: polar_chart
      has_moved: Nx.tensor(1, type: :u8)
    } = env

    speed = speed_from_heading(polar_chart, heading)

    x = x + speed * Nx.sin(heading) * @dt
    y = y + speed * Nx.cos(heading) * @dt

    vmg = speed * Nx.cos(heading)

    %__MODULE__{
      env
      | x: x,
        y: y,
        prev_x: env.x,
        prev_y: env.y,
        speed: speed,
        prev_speed: env.speed,
        remaining_iterations: env.remaining_iterations - 1,
        vmg: vmg
    }
  end

  defn speed_from_heading({linear_model, spline_model, cutoff_angle}, angle) do
    # angle is already maintained between 0 and 2pi
    # so we only need to calculate the "absolute value"
    # (as if the angle was between -pi and pi)
    angle = Nx.select(angle > pi(), 2 * pi() - angle, angle)

    linear_pred = Scholar.Interpolation.Linear.predict(linear_model, angle)
    spline_pred = Scholar.Interpolation.BezierSpline.predict(spline_model, angle)

    (angle <= cutoff_angle)
    |> Nx.select(linear_pred, spline_pred)
    |> Nx.clip(0, @max_speed)
  end

  defnp is_terminal_state(env) do
    %__MODULE__{x: x, y: y, remaining_iterations: remaining_iterations, target_y: target_y} = env

    is_terminal =
      has_reached_target(env) or x < @min_x or x > @max_x or y < @min_y or y > target_y or
        remaining_iterations < 2

    %__MODULE__{env | is_terminal: is_terminal}
  end

  defnp has_reached_target(env) do
    %__MODULE__{x: x, y: y, target_x: target_x, target_y: target_y} = env

    target_y - y < 1.5
  end

  defnp calculate_reward(env) do
    %__MODULE__{
      is_terminal: is_terminal,
      remaining_iterations: remaining_iterations,
      max_remaining_iterations: max_remaining_iterations,
      vmg: vmg
    } = env

    reward = vmg / @max_speed

    has_reached_target = has_reached_target(env)

    reward =
      cond do
        is_terminal and not has_reached_target ->
          0

        is_terminal ->
          reward + remaining_iterations / max_remaining_iterations * 100

        true ->
          reward
      end

    %__MODULE__{env | reward: reward}
  end
end
