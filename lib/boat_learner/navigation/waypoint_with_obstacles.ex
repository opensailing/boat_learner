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

  @max_iter 4000

  @state_vector_size 6
  @experience_replay_buffer_num_entries 50

  # this is equivalent as having a buffer with 10k entries and
  # uniformily random sampling 50 of those without replacement
  @record_observation_probability @experience_replay_buffer_num_entries / 10_000

  @eps_start 0.9
  @eps_end 0.05
  @eps_decay 1000
  @gamma 0.99
  @batch_size 128
  @tau 0.005
  @learning_rate 1.0e-4
  @adamw_decay 0.01

  # turn left or turn right
  @num_actions 2
  @d_angle_rad 5 * @pi / 180

  def train(obstacles_tensor, target_waypoint, trajectory_callback, reward_callback, opts \\ []) do
    # obstacles_tensor is {n, 4} where each row is [min_x, max_x, min_y, max_y]
    # target_waypoint is {2} [target_x, target_y]

    # 2pi rad with pi/90 resolution
    possible_angles = ceil(2 * @pi / @d_angle_rad)
    # @left_wall to @right_wall with 1 step
    possible_xs = @right_wall - @left_wall + 1
    possible_ys = @max_y + 1

    # This is fixed on {angle, x, y, vel_x, vel_x}
    # num_observations = 5

    policy_net = dqn(@state_vector_size, @num_actions)
    target_net = dqn(@state_vector_size, @num_actions)

    {policy_init_fn, policy_predict_fn} = Axon.build(policy_net, seed: 0)
    {target_init_fn, target_predict_fn} = Axon.build(target_net, seed: 1)

    random_key = Nx.Random.key(System.system_time())
    num_episodes = opts[:num_episodes] || 10000

    initial_state =
      {random_key, obstacles_tensor, target_waypoint, policy_init_fn, policy_predict_fn,
       target_init_fn, target_predict_fn, reward_callback}

    loop = Axon.Loop.loop(&run_iteration/2, &init/2)

    loop
    |> Axon.Loop.handle(:epoch_started, &epoch_started_handler/1)
    |> Axon.Loop.handle(:iteration_completed, &iteration_completed_handler/1)
    |> Axon.Loop.handle(:epoch_halted, &plot_trajectory(&1, trajectory_callback))
    |> Axon.Loop.run(Stream.cycle([Nx.tensor(1)]), initial_state,
      iterations: @max_iter,
      epochs: num_episodes
    )
  end

  defnp plot_trajectory(state, _trajectory_callback) do
    state
  end

  defn epoch_started_handler(state) do
    # Reset state
    reset_variable_state(state)
  end

  defn iteration_completed_handler(state) do
    terminated =
      state |> as_state_vector() |> is_terminal_state(state.obstacles) |> Nx.reshape({})

    new_state = %{state | iteration: state.iteration + 1}

    if terminated do
      {:halt_epoch, state}
    else
      {:continue, new_state}
    end
  end

  defn init(_, initial_state), do: init(initial_state)

  defn init(
         {random_key, obstacles, target_waypoint, policy_init_fn, policy_predict_fn,
          target_init_fn, target_predict_fn, reward_callback}
       ) do
    q_policy = policy_init_fn.(Nx.template({1, @state_vector_size}, type: :f64))
    q_target = target_init_fn.(Nx.template({1, @state_vector_size}, type: :f64))

    {optimizer_init_fn, optimizer_update_fn} =
      Axon.Optimizers.adamw(@learning_rate, decay: @adamw_decay)

    q_policy_optimizer_state = optimizer_init_fn.(q_policy)
    q_target_optimizer_state = optimizer_init_fn.(q_target)

    reset_variable_state(%{
      reward_callback: reward_callback,
      random_key: random_key,
      obstacles: obstacles,
      target_waypoint: target_waypoint,
      optimizer_update_fn: optimizer_update_fn,
      q_policy: q_policy,
      q_policy_optimizer_state: q_policy_optimizer_state,
      policy_predict_fn: policy_predict_fn,
      q_target: q_target,
      q_target_optimizer_state: q_target_optimizer_state,
      target_predict_fn: target_predict_fn
    })
  end

  defn run_iteration(_axon_inputs, prev_state) do
    {action, state} = select_action_and_update_state(prev_state)

    reward = reward_for_state(state)

    # termination is checked on the :iteration_completed handler

    prev_state
    |> record_observation(action, reward, state)
    |> optimize_model()
    |> soft_update_target_network()
  end

  defnp reset_variable_state(current_state) do
    velocity_model = BoatLearner.Simulator.init()

    # TO-DO: initialize to random values
    x = y = angle = Nx.tensor(0, type: :f64)

    # these should always start at 0
    iteration = Nx.tensor(0, type: :s64)

    trajectory = Nx.broadcast(Nx.tensor(:nan, type: :f64), {@max_iter, 2})

    Map.merge(current_state, %{
      iteration: iteration,
      velocity_model: velocity_model,
      x: x,
      y: y,
      angle: angle,
      trajectory: trajectory
    })
    |> reset_experience_replay_buffer()
  end

  defnp reset_experience_replay_buffer(state) do
    experience_replay_buffer_index = Nx.tensor(0, type: :s64)

    # {@experience_replay_buffer_num_entries, 2 * @state_vector_size + 2}
    # buffer entry: Nx.concatenate([state_vector, Nx.new_axis(action, 0), Nx.new_axis(reward, 0), next_state_vector])
    experience_replay_buffer =
      Nx.broadcast(
        Nx.tensor(:nan, type: :f64),
        {@experience_replay_buffer_num_entries, 2 * @state_vector_size + 2}
      )

    %{
      state
      | experience_replay_buffer: experience_replay_buffer,
        experience_replay_buffer_index: experience_replay_buffer_index
    }
  end

  defnp select_action_and_update_state(state) do
    state
    |> select_action()
    |> update_state_from_action()
  end

  defnp update_state_from_action({action, state}) do
    next_angle = next_angle_from_action(state.angle, action)
    speed = BoatLearner.Simulator.speed(state.velocity_model, next_angle)

    {next_x, next_y} =
      BoatLearner.Simulator.update_position(state.x, state.y, speed, next_angle, @dt)

    new_state = %{state | x: next_x, y: next_y, angle: next_angle}
    {action, new_state}
  end

  defnp next_angle_from_action(angle, action) do
    cond do
      action == 0 ->
        angle - @d_angle_rad

      true ->
        angle + @d_angle_rad
    end
    |> Nx.as_type(:f64)
  end

  defnp select_action(state) do
    %{
      random_key: random_key,
      iteration: iteration,
      q_policy: q_policy,
      policy_predict_fn: policy_predict_fn
    } = state

    {sample, new_key} = Nx.Random.uniform(random_key)

    eps_threshold = @eps_end + (@eps_start - @eps_end) * Nx.exp(-1 * iteration / @eps_decay)

    {action, new_key} =
      if sample > eps_threshold do
        action =
          q_policy
          |> policy_predict_fn.(%{"state" => as_state_vector(state)})
          |> Nx.argmax()

        {action, new_key}
      else
        Nx.Random.randint(new_key, 0, @num_actions, type: :s64)
      end

    new_state = %{state | random_key: new_key}
    {action, new_state}
  end

  defnp as_state_vector(%{
          x: x,
          y: y,
          angle: angle,
          velocity_model: velocity_model,
          target_waypoint: target_waypoint
        }) do
    # TO-DO: model obstacles as an unwrapped vector for the input as well

    speed = BoatLearner.Simulator.speed(velocity_model, angle)

    Nx.stack([x, y, angle, speed, target_waypoint[0], target_waypoint[1]])
  end

  defnp reward_for_state(state), do: state.reward_callback.(state)

  defnp optimize_model(state) do
    if state.iteration > @batch_size and state.experience_replay_buffer_index == 0 do
      # We collected N samples in the buffer, so we can use them to optimize
      state
      |> do_optimize_model()
      |> reset_experience_replay_buffer()
    else
      state
    end
  end

  defnp do_optimize_model(state) do
    batch = state.experience_replay_buffer

    state_batch = Nx.slice_along_axis(batch, 0, @state_vector_size, axis: 1)
    action_batch = Nx.slice_along_axis(batch, @state_vector_size, 1, axis: 1)

    reward_batch = Nx.slice_along_axis(batch, @state_vector_size + 1, 1, axis: 1)

    next_state_batch =
      Nx.slice_along_axis(batch, @state_vector_size + 2, @state_vector_size, axis: 1)

    non_final_mask = not is_terminal_state(state_batch)

    state_action_values =
      state.q_policy
      |> state.policy_predict_fn.(state_batch)
      |> Nx.take(action_batch)

    next_state_values =
      Nx.select(non_final_mask, state.target_predict_fn.(state.q_target, next_state_batch), 0)

    expected_state_action_values = next_state_values * @gamma + reward_batch

    # compute smooth l1 loss
    loss = Axon.Losses.soft_margin(expected_state_action_values, state_action_values)

    state.q_policy_optimizer_state.()
  end

  defnp soft_update_target_network(state) do
    q_target =
      Axon.Shared.deep_merge(state.q_policy, state.q_target, &(&1 * @tau + &2 * (1 - @tau)))

    %{state | q_target: q_target}
  end

  defnp record_observation(prev_state, action, reward, state) do
    {record_entry_probability, new_key} = Nx.Random.uniform(state.random_key)

    {experience_replay_buffer, experience_replay_buffer_index} =
      if record_entry_probability > @record_observation_probability do
        update_experience_replay_buffer(
          state.experience_replay_buffer,
          state.experience_replay_buffer_index,
          as_state_vector(prev_state),
          action,
          reward,
          as_state_vector(state)
        )
      else
        {state.experience_replay_buffer, state.experience_replay_buffer_index}
      end

    %{
      state
      | random_key: new_key,
        experience_replay_buffer: experience_replay_buffer,
        experience_replay_buffer_index: experience_replay_buffer_index
    }
  end

  defnp update_experience_replay_buffer(
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

    {updated,
     Nx.remainder(experience_replay_buffer_index + 1, @experience_replay_buffer_num_entries)}
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
  defnp all_collisions(obstacles, x, y, opts \\ []) do
    pos = Nx.stack([x, x, y, y]) |> Nx.new_axis(0)
    min_mask = Nx.tensor([[1, 0, 1, 0]]) |> Nx.broadcast(obstacles)

    min_mask
    |> Nx.select(pos >= obstacles, pos <= obstacles)
    |> Nx.all(axes: [1], keep_axes: opts[:keep_axes])
  end

  defnp distance(x, y, target_x, target_y) do
    Nx.sqrt((x - target_x) ** 2 + (y - target_y) ** 2)
  end

  defnp reached_target(x, y, target_x, target_y),
    do: Nx.all(distance(x, y, target_x, target_y) < 2.5)

  def dqn(num_observations, num_actions) do
    # shape is currently ignored
    Axon.input("state", shape: {nil, num_observations})
    |> Axon.dense(128, activation: :relu)
    |> Axon.dense(128, activation: :relu)
    |> Axon.dense(num_actions)
  end

  defnp is_terminal_state(state_vectors, obstacles) do
    # state_vectors: {n, state_vector_size}
    # state vector: [x, y, angle, speed, target_waypoint[0], target_waypoint[1]]

    n = Nx.axis_size(state_vectors, 0)
    is_terminal = Nx.broadcast(Nx.tensor(0, type: :u8), {n})

    {is_terminal, _i, _obstacles} =
      while {obstacles, i = Nx.tensor([[0]]), is_terminal}, Nx.reshape(i, {}) < n do
        state_vector = state_vectors[i]

        is_term =
          do_is_terminal_state(
            state_vector[0],
            state_vector[1],
            state_vector[4],
            state_vector[5],
            obstacles
          )

        is_terminal = Nx.indexed_put(is_terminal, i, Nx.reshape(is_term, {1}))
        {i + 1, obstacles, is_terminal}
      end

    is_terminal
  end

  defnp do_is_terminal_state(x, y, target_waypoint_x, target_waypoint_y, obstacles) do
    reached_target(x, y, target_waypoint_x, target_waypoint_y) or
      Nx.any(all_collisions(obstacles, x, y)) or
      x < @min_x or x > @max_x or y < 0 or y > @max_y
  end
end
