defmodule BoatLearner.Navigation.WaypointWithObstacles do
  @moduledoc """
  Training and execution of Q-Learning Model

  Objective: sail from the bottom of the grid to the top of the grid with north-to-south wind.
  """
  import Nx.Defn

  @behaviour BoatLearner.Navigation

  @dt 0.5
  @pi :math.pi()

  @left_wall -20
  @right_wall 20
  @max_y 300

  @d_angle_rad 5 * @pi / 180
  @max_iter 4000

  @learning_rate 1.0e-4
  @adamw_decay 0.01

  @state_vector_size 5
  @experience_replay_buffer_num_entries 50

  def train(obstacles_tensor, target_waypoint, trajectory_callback, opts \\ []) do
    # obstacles_tensor is {n, 4} where each row is [min_x, max_x, min_y, max_y]
    # target_waypoint is {2} [target_x, target_y]

    # 2pi rad with pi/90 resolution
    possible_angles = ceil(2 * @pi / @d_angle_rad)
    # @left_wall to @right_wall with 1 step
    possible_xs = @right_wall - @left_wall + 1
    possible_ys = @max_y + 1
    # turn left or turn right
    num_actions = 2

    # This is fixed on {angle, x, y, vel_x, vel_x}
    # num_observations = 5

    policy_net = dqn(@state_vector_size, num_actions)
    target_net = dqn(@state_vector_size, num_actions)

    {policy_init_fn, policy_predict_fn} = Axon.build(policy_net, seed: 0)
    {target_init_fn, target_predict_fn} = Axon.build(target_net, seed: 1)

    random_key = Nx.Random.key(System.system_time())
    num_episodes = opts[:num_episodes] || 10000

    initial_state =
      {random_key, obstacles_tensor, target_waypoint, policy_init_fn, target_init_fn}

    loop = Axon.Loop.loop(&run_iteration/2, &init/2)

    loop
    |> Axon.Loop.handle(:iteration_completed, &check_terminate_episode/1)
    |> Axon.Loop.handle(:epoch_halted, &plot_trajectory(&1, trajectory_callback))
    |> Axon.Loop.run(Stream.cycle([Nx.tensor(1)]), initial_state,
      iterations: @max_iter,
      epochs: num_episodes
    )
  end

  defn init(_, initial_state), do: init(initial_state)

  defn init({random_key, obstacles_tensor, target_waypoint, policy_init_fn, target_init_fn}) do
    velocity_model = BoatLearner.Simulator.init()

    q_policy = policy_init_fn.(Nx.template({1, @state_vector_size}, type: :f64))
    q_target = target_init_fn.(Nx.template({1, @state_vector_size}, type: :f64))

    {optimizer_init_fn, optimizer_update_fn} =
      Axon.Optimizers.adamw(@learning_rate, decay: @adamw_decay)

    q_policy_optimizer_state = optimizer_init_fn.(q_policy)
    q_target_optimizer_state = optimizer_init_fn.(q_target)

    x = y = angle = Nx.tensor(0, type: :f64)

    experience_replay_buffer_index = Nx.tensor(0, type: :s64)

    trajectory = Nx.broadcast(Nx.tensor(:nan, type: :f64), {@max_iter, 2})

    # s_i = Nx.stack([x, y, angle, velocity_x, velocity_y])
    # action_i
    # action_i -> reward_(i+1)
    # action_i -> s_(i+1)

    # {@experience_replay_buffer_num_entries, 2 * @state_vector_size + 2}
    # buffer entry: Nx.concatenate([state_vector, Nx.new_axis(action, 0), Nx.new_axis(reward, 0), next_state_vector])
    experience_replay_buffer =
      Nx.broadcast(
        Nx.tensor(:nan, type: :f64),
        {@experience_replay_buffer_num_entries, 2 * @state_vector_size + 2}
      )

    %{
      velocity_model: velocity_model,
      obstacles_tensor: obstacles_tensor,
      target_waypoint: target_waypoint,
      q_policy: q_policy,
      q_target: q_target,
      q_policy_optimizer_state: q_policy_optimizer_state,
      q_target_optimizer_state: q_target_optimizer_state,
      x: x,
      y: y,
      angle: angle,
      random_key: random_key,
      trajectory: trajectory,
      experience_replay_buffer: experience_replay_buffer,
      experience_replay_buffer_index: experience_replay_buffer_index
    }
  end

  # {experience_replay_buffer, experience_replay_buffer_index} =
  #   update_experience_replay_buffer(
  #     experience_replay_buffer,
  #     experience_replay_buffer_index,
  #     state_vector,
  #     action,
  #     reward,
  #     next_state_vector
  #   )

  defn update_experience_replay_buffer(
         experience_replay_buffer,
         experience_replay_buffer_index,
         state_vector,
         action,
         reward,
         next_state_vector
       ) do
    idx = Nx.stack([[experience_replay_buffer_index, 0]])

    shape = {Nx.size(state_vector), 1}
    index_template = Nx.concatenate([Nx.broadcast(0, shape), Nx.iota(shape, axis: 0)], axis: 1)

    updates = Nx.concatenate([state_vector, Nx.stack([action, reward]), next_state_vector])

    updated = Nx.indexed_put(experience_replay_buffer, idx + index_template, updates)

    {updated, experience_replay_buffer_index + 1}
  end

  defnp select_action(state) do
    
  end

  defn run_iteration(_axon_inputs, state) do

    {action, state} = select_action(state)
    # select action from the softmax distribution
    # q has shape {*, *, 2} so this will return {2}
    action_probabilities = Axon.Activations.softmax(q[[state, x_state, y_state]])

    # random choice: will be contributed to nx afterwards
    {action, random_key} = choice(random_key, Nx.iota({2}, type: :s64), action_probabilities)
    d_angle = Nx.select(action == 0, -@d_angle_rad, @d_angle_rad)

    next_angle = Nx.as_type(angle + d_angle, :f64)
    next_state = angle_to_state(next_angle)

    v = velocity(velocity_model, next_angle) |> Nx.new_axis(0)

    next_xy = BoatLearner.Simulator.update_position(Nx.stack([x, y]), v, @dt)
    next_x = Nx.slice_along_axis(next_xy, 0, 1, axis: 1) |> Nx.reshape({})
    next_y = Nx.slice_along_axis(next_xy, 1, 1, axis: 1) |> Nx.reshape({})

    next_x_state = x_to_state(next_x)
    next_y_state = y_to_state(next_y)

    reward =
      reward(velocity_model, next_angle, iter, next_x, next_y, obstacles_tensor, target_waypoint)

    delta =
      cond do
        reached_target(x, y, target_waypoint) ->
          10000

        y > @max_y or next_x < @left_wall or next_x > @right_wall or
            Nx.any(all_collisions(obstacles_tensor, next_x, next_y)) ->
          # out of bounds, terminal case
          -10 * (@max_iter - iter) + reward

        # -100

        true ->
          reward - rho + Nx.reduce_max(q[[next_state, next_x_state, next_y_state]]) -
            q[[state, x_state, y_state, action]]
      end

    next_rho = rho + 0.1 * (reward - rho)

    next_q =
      Nx.indexed_add(
        q,
        Nx.stack([state, x_state, y_state, action]) |> Nx.new_axis(0),
        0.1 * delta
      )

    {velocity_model, obstacles_tensor, target_waypoint, next_q, Nx.reshape(next_rho, {}), next_x,
     next_y, Nx.reshape(next_angle, {}), random_key}
  end

  @doc """
      iex> obstacles_tensor = Nx.tensor([[0, 2, 0, 3], [0, 6, 0, 1]])
      iex> x = 1
      iex> y = 2
      iex> BoatLearner.Navigation.WaypointWithObstacles.all_collisions(obstacles_tensor, x, y, keep_axes: true)
      #Nx.Tensor<
        u8[2][1]
        [
          [1],
          [0]
        ]
      >
  """
  defn all_collisions(obstacles, x, y, opts \\ []) do
    pos = Nx.stack([x, x, y, y]) |> Nx.new_axis(0)
    min_mask = Nx.tensor([[1, 0, 1, 0]]) |> Nx.broadcast(obstacles)

    min_mask
    |> Nx.select(pos >= obstacles, pos <= obstacles)
    |> Nx.all(axes: [1], keep_axes: opts[:keep_axes])
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
    out = 2 * @pi / @d_angle_rad * Nx.remainder((angle + @pi) / (2 * @pi), 1)
    Nx.as_type(out, :s64)
  end

  # The grid will have 0.1m resolution
  defn x_to_state(x) do
    Nx.as_type(
      2 * (@right_wall - @left_wall) * ((x - @left_wall) / (@right_wall - @left_wall)),
      :s64
    )
  end

  defn y_to_state(y) do
    Nx.as_type(Nx.floor(y), :s64)
  end

  # velocity is {speed, angle}
  defn velocity(model, angle) do
    speed = BoatLearner.Simulator.speed(model, angle)
    Nx.stack([speed, angle], axis: -1)
  end

  defn reward(model, angle, iter, x, y, obstacles_tensor, target_waypoint) do
    velocity = velocity(model, angle)

    r = Nx.slice_along_axis(velocity, 0, 1, axis: -1)
    theta = Nx.slice_along_axis(velocity, 1, 1, axis: -1)

    distance_to_target = distance(x, y, target_waypoint)

    # distance_score = Axon.Activations.tanh(distance_to_target)
    distance_score = -distance_to_target

    velocity_vector = Nx.reshape(velocity, {2})
    target_direction_vector = Nx.stack([x, y]) - target_waypoint
    target_direction_vector = target_direction_vector / Nx.LinAlg.norm(target_direction_vector)

    projected_speed = Nx.dot(target_direction_vector, velocity_vector)
    # speed_score = Axon.Activations.tanh(projected_speed)
    speed_score = 0
    iter_score = 0

    obstacle_score =
      -Nx.sum(
        all_collisions(obstacles_tensor, x * 1.15, y) +
          all_collisions(obstacles_tensor, x * 0.85, y) +
          all_collisions(obstacles_tensor, x, y * 1.15) +
          all_collisions(obstacles_tensor, x, y * 0.85)
      )
      |> Axon.Activations.tanh()

    Nx.reshape(speed_score + distance_score + iter_score + obstacle_score, {:auto})
  end

  defn distance(x, y, target) do
    # normalization_factor = Nx.tensor([@right_wall - @left_wall, @max_y]) |> print_value()
    # pos = Nx.stack([x, y]) / normalization_factor
    pos = Nx.stack([x, y])
    # Scholar.Metrics.Distance.euclidean(target / normalization_factor, pos)
    max_distance = Nx.sqrt(((@right_wall - @left_wall) / 2) ** 2 + @max_y ** 2)

    Scholar.Metrics.Distance.euclidean(target, pos)
  end

  defnp reached_target(x, y, target_waypoint), do: Nx.all(distance(x, y, target_waypoint) < 2.5)

  def dqn(num_observations, num_actions) do
    # shape is currently ignored
    Axon.input("state", shape: {nil, num_observations})
    |> Axon.dense(128, activation: :relu)
    |> Axon.dense(128, activation: :relu)
    |> Axon.dense(num_actions)
  end
end
