defmodule BoatLearner.Navigation.SouthToNorth do
  @moduledoc """
  Training and execution of Q-Learning Model

  Objective: sail from the bottom of the grid to the top of the grid with north-to-south wind.
  """
  import Nx.Defn

  @behaviour BoatLearner.Navigation

  @dt 0.1

  @impl true
  def train do
    # 180 degrees with 5 degree resolution
    possible_angles = 30
    # -50 to 50 with 1 step
    possible_xs = 100
    # turn left or turn right
    num_actions = 2
    q = Nx.broadcast(Nx.tensor(0, type: :f64), {possible_angles, possible_xs, num_actions})
    # avg reward initialized to 0
    rho = Nx.tensor(0, type: :f64)
    velocity_model = BoatLearner.Simulator.init()
    random_key = Nx.Random.key(System.system_time())

    {y, result} =
      Enum.map_reduce(0..200, {rho, q, random_key}, fn i, acc ->
        IO.puts("[#{NaiveDateTime.utc_now()}] Starting episode #{i}")
        {y, _} = result = episode(velocity_model, acc)
        IO.inspect(y, label: "y[#{i}]")

        result
      end)

    {result, y}
  end

  defnp episode(velocity_model, {rho, q, random_key}) do
    {x, random_key} = Nx.Random.uniform(random_key, -50, 50, type: :f64)
    angle = Nx.tensor(0, type: :f64)
    y = Nx.broadcast(Nx.tensor(0, type: :f64), x)
    state = {velocity_model, q, rho, x, y, angle, random_key}

    {_i, _continue, {_velocity_model, q, rho, _x, y, _angle, random_key}} =
      while {i = 0, continue = Nx.tensor(1, type: :u8), state},
            i < 200 and continue do
        {_velocity_model, _q, _rho, x, _y, _angle, _random_key} = next_state = iteration(state)
        continue = x > -50 and x < 50

        {i + 1, continue, next_state}
      end

    {y, {rho, q, random_key}}
  end

  defn iteration({velocity_model, q, rho, x, y, angle, random_key}) do
    state = angle_to_state(angle)
    x_state = x_to_state(x)

    # select action from the softmax distribution
    # q has shape {*, *, 2} so this will return {2}
    action_probabilities = Axon.Activations.softmax(q[[state, x_state]])

    # random choice: will be contributed to nx afterwards
    {action, random_key} = choice(random_key, Nx.iota({2}, type: :s64), action_probabilities)
    d_angle = Nx.select(action, -5, 5)

    next_angle = Nx.as_type(state + d_angle, :f64)
    next_state = angle_to_state(next_angle)

    v = velocity(velocity_model, next_angle) |> Nx.new_axis(0)

    next_xy = BoatLearner.Simulator.update_position(Nx.stack([x, y]), v, @dt)
    next_x = Nx.slice_along_axis(next_xy, 0, 1, axis: 1) |> Nx.reshape({})
    next_y = Nx.slice_along_axis(next_xy, 1, 1, axis: 1) |> Nx.reshape({})

    next_x_state = x_to_state(next_x)

    # Grid has a lateral bounding on [-50, 50]

    reward = reward(velocity_model, next_angle)

    delta =
      if next_x < -50 or next_x > 50 do
        # out of bounds, terminal case
        -10
      else
        reward - rho + Nx.reduce_max(q[[next_state, next_x_state]]) -
          q[[state, x_state, action]]
      end

    next_rho = rho + 0.1 * (reward - rho)

    next_q = Nx.indexed_add(q, Nx.stack([state, x_state, action]) |> Nx.new_axis(0), 0.1 * delta)

    {velocity_model, next_q, Nx.reshape(next_rho, {}), next_x, next_y, Nx.reshape(next_angle, {}),
     random_key}
  end

  defn choice(key, target, probabilities) do
    p_cumulative = Nx.cumulative_sum(probabilities)
    {value, key} = Nx.Random.uniform(key, shape: Nx.shape(target))
    r = p_cumulative[-1] * (1 - value)
    index = Nx.argmin(Nx.sort(p_cumulative) <= r, tie_break: :low)
    {target[index], key}
  end

  # We'll treat angles with 30 degree resolution
  defnp angle_to_state(angle) do
    out = 30 * Nx.remainder((angle + 180) / 360, 1)
    Nx.as_type(out, :s64)
  end

  # The grid will have 0.1m resolution
  defnp x_to_state(x) do
    Nx.as_type(50 * ((x + 50) / 100), :s64)
  end

  # velocity is {angle, speed}
  defnp velocity(model, angle) do
    speed = BoatLearner.Simulator.speed(model, angle)
    Nx.stack([angle, speed], axis: -1)
  end

  defnp reward(model, angle) do
    velocity = velocity(model, angle)

    r = Nx.slice_along_axis(velocity, 0, 1, axis: -1)
    theta = Nx.slice_along_axis(velocity, 1, 1, axis: -1)

    # This could be written as just r * Nx.cos(theta)
    # but writing it like this will enable us to change
    # targets more easily in the future

    # velocity_vector = r * Nx.stack([Nx.cos(theta), Nx.sin(theta)], axis: 1)
    # target_direction_vector = Nx.tensor([[1, 0]]) |> Nx.broadcast(velocity_vector)

    # # batched projection of velocity_vector onto target_direction_vector
    # # We want to maximize the velocity in the direction of the target,
    # # so taking our velocity's projection onto the target direction does
    # # the trick
    # Nx.dot(
    #   target_direction_vector / Nx.LinAlg.norm(target_direction_vector, axes: [1]),
    #   [1],
    #   [0],
    #   velocity_vector,
    #   [1],
    #   [0]
    # )

    r * Nx.cos(theta)
  end
end
