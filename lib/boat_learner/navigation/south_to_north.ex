defmodule BoatLearner.Navigation.SouthToNorth do
  @moduledoc """
  Training and execution of Q-Learning Model

  Objective: sail from the bottom of the grid to the top of the grid with north-to-south wind.
  """
  import Nx.Defn

  @dt 1
  @pi :math.pi()

  @left_wall -10
  @right_wall 10

  @d_angle_rad 0.1
  @max_iter 250

  def plot_trajectory({episode, num_points, trajectory}) do
    episode = Nx.to_number(episode)
    num_points = Nx.to_number(num_points)

    Nx.slice(trajectory, [0, 0], [num_points, 2])
    |> Nx.reduce_max(axes: [0])
    |> IO.inspect(label: "Episode #{episode}")
  end

  def train(trajectory_callback) do
    # pi rad with pi/30 rad  resolution
    possible_angles = 30
    # @left_wall to @right_wall with 1 step
    possible_xs = @right_wall - @left_wall + 1
    # turn left or turn right
    num_actions = 2
    q = Nx.broadcast(Nx.tensor(0, type: :f64), {possible_angles, possible_xs, num_actions})
    # avg reward initialized to 0
    rho = Nx.tensor(0, type: :f64)
    velocity_model = BoatLearner.Simulator.init()
    random_key = Nx.Random.key(System.system_time())

    run_ep_fn = Nx.Defn.jit(&run_episodes/4, hooks: %{plot_trajectory: trajectory_callback})

    {_, y, _rho, q, _random_key, _velocity_model} = run_ep_fn.(velocity_model, rho, q, random_key)

    {y, q}
  end

  defnp run_episodes(velocity_model, rho, q, random_key) do
    while {i = 0, y = Nx.tensor(0.0, type: :f64), rho, q, random_key, velocity_model},
          i < 15_000 do
      i =
        hook(i, fn i ->
          IO.puts("[#{NaiveDateTime.utc_now()}] Starting episode #{Nx.to_number(i)}")
        end)

      {curr_y, trajectory, num_points, {rho, q, random_key}} =
        episode(velocity_model, {rho, q, random_key})

      token = create_token()
      {token, _} = hook_token(token, {i, num_points, trajectory}, :plot_trajectory)

      y = Nx.max(curr_y, y)
      y = attach_token(token, y)

      {i + 1, y, rho, q, random_key, velocity_model}
    end
  end

  defnp episode(velocity_model, {rho, q, random_key}) do
    {x, random_key} = Nx.Random.uniform(random_key, @left_wall, @right_wall, type: :f64)
    angle = Nx.tensor(0, type: :f64)
    y = Nx.broadcast(Nx.tensor(0, type: :f64), x)
    state = {velocity_model, q, rho, x, y, angle, random_key}

    trajectory = Nx.broadcast(Nx.tensor(:nan, type: :f64), {@max_iter, 2})

    {i, _continue, trajectory, {_velocity_model, q, rho, _x, last_y, _angle, random_key}} =
      while {i = 0, continue = Nx.tensor(1, type: :u8), trajectory, state},
            i < @max_iter and continue do
        {_velocity_model, _q, _rho, x, y, _angle, _random_key} = next_state = iteration(state, i)
        continue = x > @left_wall and x < @right_wall

        idx_template =
          Nx.tensor([
            [0, 0],
            [0, 1]
          ])

        index = idx_template + Nx.stack([i, 0])

        trajectory = Nx.indexed_put(trajectory, index, Nx.stack([x, y]))

        {i + 1, continue, trajectory, next_state}
      end

    {last_y, trajectory, i, {rho, q, random_key}}
  end

  defn iteration({velocity_model, q, rho, x, y, angle, random_key}, iter) do
    state = angle_to_state(angle)
    x_state = x_to_state(x)

    # select action from the softmax distribution
    # q has shape {*, *, 2} so this will return {2}
    action_probabilities = Axon.Activations.softmax(q[[state, x_state]])

    # random choice: will be contributed to nx afterwards
    {action, random_key} = choice(random_key, Nx.iota({2}, type: :s64), action_probabilities)
    d_angle = Nx.select(action == 0, -@d_angle_rad, @d_angle_rad)

    next_angle = Nx.as_type(angle + d_angle, :f64)
    next_state = angle_to_state(next_angle)

    v = velocity(velocity_model, next_angle) |> Nx.new_axis(0)

    next_xy = BoatLearner.Simulator.update_position(Nx.stack([x, y]), v, @dt) |> print_expr()
    next_x = Nx.slice_along_axis(next_xy, 0, 1, axis: 1) |> Nx.reshape({})
    next_y = Nx.slice_along_axis(next_xy, 1, 1, axis: 1) |> Nx.reshape({})

    next_x_state = x_to_state(next_x)

    # Grid has a lateral bounding on [@left_wall, @right_wall]

    reward = reward(velocity_model, next_angle, iter, next_y)

    delta =
      if next_x < @left_wall or next_x > @right_wall do
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
  defn angle_to_state(angle) do
    out = 30 * Nx.remainder((angle + @pi) / (2 * @pi), 1)
    Nx.as_type(out, :s64)
  end

  # The grid will have 0.1m resolution
  defn x_to_state(x) do
    Nx.as_type(
      2 * (@right_wall - @left_wall) * ((x - @left_wall) / (@right_wall - @left_wall)),
      :s64
    )
  end

  # velocity is {speed, angle}
  defn velocity(model, angle) do
    speed = BoatLearner.Simulator.speed(model, angle)
    Nx.stack([speed, angle], axis: -1)
  end

  defn reward(model, angle, iter, y) do
    velocity = velocity(model, angle)

    r = Nx.slice_along_axis(velocity, 0, 1, axis: -1)
    # theta = Nx.slice_along_axis(velocity, 1, 1, axis: -1)

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

    # r = 1 - Nx.exp(-angle ** 2 / (@pi / 12))
    # Nx.new_axis(r)

    # Nx.cos(theta) + Nx.atan(y) / (@pi / 2)

    # Axon.Activations.sigmoid(y * (1 - iter / @max_iter))
    # Axon.Activations.sigmoid(r)
    (Axon.Activations.sigmoid(r / (2 * 12)) + Axon.Activations.sigmoid(y * (1 - iter / @max_iter)))
    |> Nx.reshape({:auto})
  end
end
