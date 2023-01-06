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
    possible_angles = 180 # 180 degrees with 1 degree resolution
    possible_xs = 1000 # -50 to 50 with 0.1 step
    num_actions = 2 # turn left or turn right
    q = Nx.zeros({possible_angles, possible_xs, num_actions})
    rho = 0 # avg reward initialized to 0
    velocity_model = BoatLearner.Simulator.init()

    Enum.map_reduce(0..1000, {rho, q, random_key}, fn _, acc -> episode(velocity_model, acc) end)
  end

  defnp episode(velocity_model, {rho, q, random_key}) do
    {x, random_key} = Nx.Random.uniform(random_key, -50, 50)
    y = 0
    angle = 0
    {_velocity_model, q, rho, _x, y, _angle, random_key} = Enum.reduce_while(0..200, {velocity_model, q, rho, x, y, angle, random_key}, &iteration/2)
    {y, {rho, q, random_key}}
  end

  defn iteration(_, state) do
    {_velocity_model, _q, _rho, x, _y, _angle, _random_key} = iteration(state)

    if x > 50 or x < -50 do
      {:halt, state}
    else
      {:cont, state}
    end
  end

  defn iteration({velocity_model, q, rho, x, y, angle, random_key}) do
    state = angle_to_state(angle)
    x_state = x_to_state(x)

    # select action from the softmax distribution
    # q has shape {*, *, 2}, so this will return {2}
    action_probabilities = Axon.Activations.softmax(q[[state, x_state]])

    # random choice: will be contributed to nx afterwards
    {action, random_key} = choice(random_key, Nx.iota({2}, type: :u8), action_probabilities)
    d_angle = Nx.select(action, -0.1, 0.1)

    next_angle = state + d_angle
    next_state = angle_to_state(next_angle)

    v = velocity(velocity_model, next_angle)

    next_xy = BoatLearner.Simulator.update_position(Nx.concatenate([x, y], axis: 1), v, @dt)
    next_x = Nx.slice_along_axis(next_xy, 0, 1, axis: 1)
    next_y = Nx.slice_along_axis(next_xy, 1, 1, axis: 1)

    next_x_state = x_to_state(next_x)

    # Grid has a lateral bounding on [-50, 50]
    delta =
      if next_x < -50 or next_x > 50 do
        # out of bounds, terminal case
        -10
      else
        reward(velocity_model, next_angle) - rho + Nx.max(q[next_state, next_x_state]) - q[state, x_state, action]
      end

    next_rho = rho + 0.1 * (reward(next_angle) - rho)
    next_q = Nx.indexed_add(q, Nx.concatenate([state, x_state, action]) |> Nx.next_axis(0), 0.1 * delta)

    {velocity_model, next_q, next_rho, next_x, next_y, next_angle, random_key}
  end

  defn choice(key, target, probabilities) do
    p_cumulative = Nx.cumulative_sum(probabilities)
    {value, key} = Nx.Random.uniform(key, shape: Nx.shape(target))
    r = p_cumulative[-1] * (1 - value)
    index = Nx.argmin(Nx.sort(p_cumulative) <= r, tie_break: :low)
    {target[index], key}
  end

  # We'll treat angles with 1 degree resolution
  defnp angle_to_state(angle), do: Nx.abs(Nx.floor(angle))

  # The grid will have 0.1m resolution
  defnp x_to_state(x) do
    Nx.round(x * 10) / 10 + 500
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

    velocity_vector = r * Nx.concatenate([Nx.cos(theta), Nx.sin(theta)], axis: 1)
    target_direction_vector = Nx.tensor([[1, 0]]) |> Nx.broadcast(velocity_vector)

    # batched projection of velocity_vector onto target_direction_vector
    # We want to maximize the velocity in the direction of the target,
    # so taking our velocity's projection onto the target direction does
    # the trick
    Nx.dot(
      target_direction_vector / Nx.LinAlg.norm(target_direction_vector, axes: [1]),
      [1],
      [0],
      velocity_vector,
      [1],
      [0]
    )
  end
end
