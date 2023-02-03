defmodule BoatLearner.Navigation.WaypointWithObstacles do
  @moduledoc """
  Training and execution of Q-Learning Model

  Objective: sail from the bottom of the grid to the top of the grid with north-to-south wind.
  """
  import Nx.Defn

  @state_vector_size 10
  @experience_replay_buffer_num_entries 10_000
  @max_iter 4000

  @derive {Nx.Container,
           containers: [
             :random_key,
             :obstacles_tensor,
             :target_waypoint,
             :q_policy,
             :q_target,
             :q_policy_optimizer_state,
             :q_target_optimizer_state,
             :iteration,
             :x,
             :y,
             :angle,
             :trajectory,
             :experience_replay_buffer,
             :experience_replay_buffer_index,
             :persisted_experience_replay_buffer_entries
           ],
           keep: [
             :velocity_model
           ]}
  defstruct [
    :velocity_model,
    :random_key,
    :obstacles_tensor,
    :q_policy_optimizer_state,
    :q_target_optimizer_state,
    :q_policy,
    :q_target,
    :target_waypoint,
    iteration: Nx.template({}, :s64),
    x: Nx.template({}, :f32),
    y: Nx.template({}, :f32),
    angle: Nx.template({}, :f32),
    trajectory: Nx.template({@max_iter, 2}, :f32),
    experience_replay_buffer:
      Nx.template({@experience_replay_buffer_num_entries, @state_vector_size * 2 + 2}, :f32),
    experience_replay_buffer_index: Nx.template({}, :s64),
    persisted_experience_replay_buffer_entries: Nx.template({}, :s64)
  ]

  @dt 1
  @pi :math.pi()

  @min_x -20
  @max_x 20
  @min_y 0
  @max_y 300

  @eps_start 0.9
  @eps_end 0.05
  @eps_decay 1000
  @gamma 0.99
  @batch_size 128
  @tau 0.005
  @learning_rate 1.0e-5
  @adamw_decay 0.01

  # turn left or turn right
  @num_actions 2
  @d_angle_rad 5 * @pi / 180

  def train(
        obstacles_tensor,
        target_waypoint,
        trajectory_callback \\ & &1,
        reward_fn \\ &reward_for_state/1,
        opts \\ []
      ) do
    # obstacles_tensor is {n, 4} where each row is [min_x, max_x, min_y, max_y]
    # target_waypoint is {2} [target_x, target_y]

    # This is fixed on {angle, x, y, vel_x, vel_x}
    # num_observations = 5

    policy_net = dqn(@state_vector_size, @num_actions)
    target_net = dqn(@state_vector_size, @num_actions)

    {policy_init_fn, policy_predict_fn} = Axon.build(policy_net, seed: 0)
    {target_init_fn, target_predict_fn} = Axon.build(target_net, seed: 1)

    {optimizer_init_fn, optimizer_update_fn} =
      Axon.Optimizers.adamw(@learning_rate, decay: @adamw_decay)

    q_policy = policy_init_fn.(Nx.template({1, @state_vector_size}, :f32), %{})
    q_target = target_init_fn.(Nx.template({1, @state_vector_size}, :f32), %{})

    q_policy_optimizer_state = optimizer_init_fn.(q_policy)
    q_target_optimizer_state = optimizer_init_fn.(q_target)

    random_key = Nx.Random.key(System.system_time())
    num_episodes = opts[:num_episodes] || 10000

    initial_state = %__MODULE__{
      random_key: random_key,
      obstacles_tensor: obstacles_tensor,
      target_waypoint: target_waypoint,
      q_policy_optimizer_state: q_policy_optimizer_state,
      q_target_optimizer_state: q_target_optimizer_state,
      q_policy: q_policy,
      q_target: q_target,
      experience_replay_buffer:
        Nx.broadcast(
          Nx.tensor(:nan, type: :f32),
          {@experience_replay_buffer_num_entries, 2 * @state_vector_size + 2}
        ),
      experience_replay_buffer_index: Nx.tensor(0, type: :s64),
      persisted_experience_replay_buffer_entries: Nx.tensor(0, type: :s64)
    }

    loop =
      Axon.Loop.loop(
        &batch_step(&1, &2, reward_fn, policy_predict_fn, target_predict_fn, optimizer_update_fn),
        &init/2
      )

    loop
    |> Axon.Loop.handle(:epoch_started, &epoch_started_handler/1)
    |> Axon.Loop.handle(:epoch_completed, fn s -> {:continue, trajectory_callback.(s)} end)
    |> Axon.Loop.run(Stream.cycle([Nx.tensor(1)]), initial_state,
      iterations: 1,
      epochs: num_episodes
    )
  end

  defp epoch_started_handler(loop_state) do
    # Reset state
    {:continue, %{loop_state | step_state: reset_variable_state(loop_state.step_state)}}
  end

  defn(init(_, initial_state), do: init(initial_state))

  defn init(initial_state) do
    velocity_model = BoatLearner.Simulator.init()

    reset_variable_state(%{initial_state | velocity_model: velocity_model})
  end

  defn batch_step(
         _axon_inputs,
         prev_state,
         reward_fn,
         policy_predict_fn,
         target_predict_fn,
         optimizer_update_fn
       ) do
    {state, _, _} =
      while {prev_state, i = 0, is_terminal = Nx.tensor(0, type: :u8)},
            i < @max_iter and not is_terminal do
        {action, next_state} = select_action_and_update_state(prev_state, policy_predict_fn)

        reward = reward_fn.(next_state)

        next_state =
          prev_state
          |> record_observation(action, reward, next_state)
          |> optimize_model(policy_predict_fn, target_predict_fn, optimizer_update_fn)
          |> soft_update_target_network()
          |> persist_trajectory()

        is_terminal =
          next_state
          |> as_state_vector()
          |> Nx.new_axis(0)
          |> is_terminal_state(next_state.obstacles_tensor)
          |> Nx.reshape({})

        {next_state, i + 1, is_terminal}
      end

    state
  end

  defnp persist_trajectory(state) do
    idx =
      Nx.tensor([
        [0, 0],
        [0, 1]
      ])

    idx = idx + Nx.new_axis(Nx.stack([state.iteration, 0]), 0)

    trajectory = Nx.indexed_put(state.trajectory, idx, Nx.stack([state.x, state.y]))

    %{state | trajectory: trajectory, iteration: state.iteration + 1}
  end

  defnp reset_variable_state(current_state) do
    # TO-DO: initialize to random values
    y = Nx.tensor(0, type: :f32)

    {x, random_key} = Nx.Random.uniform(current_state.random_key, -5, 5)
    {angle, random_key} = Nx.Random.uniform(random_key, -5 * @d_angle_rad, 5 * @d_angle_rad)

    # these should always start at 0
    iteration = Nx.tensor(0, type: :s64)

    trajectory = Nx.broadcast(Nx.tensor(:nan, type: :f32), {@max_iter, 2})

    %{
      current_state
      | iteration: iteration,
        x: x,
        y: y,
        angle: angle,
        trajectory: trajectory,
        random_key: random_key
    }
  end

  defnp select_action_and_update_state(state, policy_predict_fn) do
    state
    |> select_action(policy_predict_fn)
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
    Nx.select(action == 0, angle + @d_angle_rad, angle - @d_angle_rad)
  end

  defnp select_action(state, policy_predict_fn) do
    %{
      random_key: random_key,
      iteration: iteration,
      q_policy: q_policy
    } = state

    {sample, new_key} = Nx.Random.uniform(random_key)

    eps_threshold = @eps_end + (@eps_start - @eps_end) * Nx.exp(-1 * iteration / @eps_decay)

    {action, new_key} =
      if sample > eps_threshold do
        action =
          q_policy
          |> policy_predict_fn.(%{"state" => as_state_vector(state) |> Nx.new_axis(0)})
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

    Nx.stack([
      x,
      y,
      angle,
      speed,
      target_waypoint[0],
      target_waypoint[1],
      @min_x,
      @max_x,
      @min_y,
      @max_y
    ])
  end

  defn reward_for_state(state) do
    %{x: x, y: y, angle: angle} = state

    speed = BoatLearner.Simulator.speed(state.velocity_model, angle)

    pos = Nx.stack([x, y])

    diff_vector = state.target_waypoint - pos
    diff_vector_complex = Nx.complex(diff_vector[0], diff_vector[1])
    distance = Nx.abs(diff_vector_complex)
    angle_to_destination = Nx.phase(diff_vector_complex)

    projected_speed = angle_to_destination |> Nx.sin() |> Nx.multiply(speed)

    Axon.Activations.tanh(Nx.divide(projected_speed, Nx.add(distance, 1.0e-7)))
  end

  defnp optimize_model(state, policy_predict_fn, target_predict_fn, optimizer_update_fn) do
    if rem(state.iteration, @batch_size * 2) == 0 do
      do_optimize_model(state, policy_predict_fn, target_predict_fn, optimizer_update_fn)
    else
      state
    end
  end

  defnp do_optimize_model(state, policy_predict_fn, target_predict_fn, optimizer_update_fn) do
    {batch, random_key} =
      Nx.Random.choice(state.random_key, slice_experience_replay_buffer(state),
        samples: @batch_size,
        replace: false,
        axis: 0
      )

    state_batch = Nx.slice_along_axis(batch, 0, @state_vector_size, axis: 1)
    action_batch = Nx.slice_along_axis(batch, @state_vector_size, 1, axis: 1)

    reward_batch = Nx.slice_along_axis(batch, @state_vector_size + 1, 1, axis: 1)

    next_state_batch =
      Nx.slice_along_axis(batch, @state_vector_size + 2, @state_vector_size, axis: 1)

    non_final_mask = not is_terminal_state(state_batch, state.obstacles_tensor)

    {_loss, gradient} =
      value_and_grad(state.q_policy, fn q_policy ->
        action_idx = Nx.as_type(action_batch, :s64)

        state_action_values =
          q_policy
          |> policy_predict_fn.(state_batch)
          |> Nx.take_along_axis(action_idx, axis: 1)

        next_state_values =
          Nx.select(
            non_final_mask,
            target_predict_fn.(state.q_target, next_state_batch) |> Nx.argmax(axis: 1),
            0
          )

        expected_state_action_values = next_state_values * @gamma + reward_batch

        huber_loss(state_action_values, expected_state_action_values)
      end)

    {scaled_updates, optimizer_state} =
      optimizer_update_fn.(gradient, state.q_policy_optimizer_state, state.q_policy)

    q_policy = Axon.Updates.apply_updates(state.q_policy, scaled_updates)

    %{
      state
      | q_policy: q_policy,
        q_policy_optimizer_state: optimizer_state,
        random_key: random_key
    }
  end

  defnp slice_experience_replay_buffer(state) do
    if state.persisted_experience_replay_buffer_entries < @experience_replay_buffer_num_entries do
      t = Nx.iota({@experience_replay_buffer_num_entries})
      idx = Nx.select(t > state.persisted_experience_replay_buffer_entries, 0, t)
      Nx.take(state.experience_replay_buffer, idx)
    else
      state.experience_replay_buffer
    end
  end

  defnp huber_loss(y_pred, y_true, opts \\ [beta: 1.0, reduction_fn: &Nx.mean/1]) do
    abs_diff = Nx.abs(y_pred - y_true)

    abs_diff
    |> Nx.select(0.5 * abs_diff ** 2 / opts[:beta], abs_diff - 0.5 * opts[:beta])
    |> then(opts[:reduction_fn])
  end

  defnp soft_update_target_network(state) do
    q_target =
      Axon.Shared.deep_merge(state.q_policy, state.q_target, &(&1 * @tau + &2 * (1 - @tau)))

    %{state | q_target: q_target}
  end

  defnp record_observation(prev_state, action, reward, state) do
    state_vector = as_state_vector(prev_state)
    next_state_vector = as_state_vector(state)

    idx = Nx.stack([state.experience_replay_buffer_index, 0]) |> Nx.new_axis(0)

    shape = {Nx.size(state_vector) + 2 + Nx.size(state_vector), 1}
    index_template = Nx.concatenate([Nx.broadcast(0, shape), Nx.iota(shape, axis: 0)], axis: 1)

    updates = Nx.concatenate([state_vector, Nx.stack([action, reward]), next_state_vector])

    experience_replay_buffer =
      Nx.indexed_put(state.experience_replay_buffer, idx + index_template, updates)

    experience_replay_buffer_index =
      Nx.remainder(
        state.experience_replay_buffer_index + 1,
        @experience_replay_buffer_num_entries
      )

    %{
      state
      | experience_replay_buffer: experience_replay_buffer,
        experience_replay_buffer_index: experience_replay_buffer_index
    }
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
    pos = Nx.concatenate([x, x, y, y]) |> Nx.new_axis(0)
    min_mask = Nx.tensor([[1, 0, 1, 0]]) |> Nx.broadcast(obstacles)

    min_mask
    |> Nx.select(pos >= obstacles, pos <= obstacles)
    |> Nx.all(axes: [1], keep_axes: opts[:keep_axes])
  end

  defnp distance(x, y, target_x, target_y) do
    Nx.sqrt((x - target_x) ** 2 + (y - target_y) ** 2)
  end

  defnp(reached_target(x, y, target_x, target_y),
    do: Nx.all(distance(x, y, target_x, target_y) < 2.5)
  )

  def dqn(num_observations, num_actions) do
    # shape is currently ignored
    Axon.input("state", shape: {nil, num_observations})
    |> Axon.dense(128, activation: :relu)
    |> Axon.dense(64, activation: :relu)
    |> Axon.dense(32, activation: :relu)
    |> Axon.dense(num_actions)
  end

  defnp is_terminal_state(state_vectors, obstacles) do
    # state_vectors: {n, state_vector_size}
    # state vector: [x, y, angle, speed, target_waypoint[0], target_waypoint[1]]

    n = Nx.axis_size(state_vectors, 0)
    is_terminal = Nx.broadcast(Nx.tensor(0, type: :u8), {n})

    {is_terminal, _i, _n, _obstacles, _state_vectors} =
      while {is_terminal, i = Nx.tensor(0), n, obstacles, state_vectors}, i < n do
        state_vector = state_vectors[i]

        is_term =
          do_is_terminal_state(
            Nx.take(state_vector, Nx.tensor([0])),
            Nx.take(state_vector, Nx.tensor([1])),
            Nx.take(state_vector, Nx.tensor([4])),
            Nx.take(state_vector, Nx.tensor([5])),
            obstacles
          )

        is_terminal = Nx.indexed_put(is_terminal, Nx.reshape(i, {1, 1}), Nx.reshape(is_term, {1}))
        {is_terminal, i + 1, n, obstacles, state_vectors}
      end

    is_terminal
  end

  defnp do_is_terminal_state(x, y, target_waypoint_x, target_waypoint_y, obstacles) do
    reached_target(x, y, target_waypoint_x, target_waypoint_y) or
      Nx.any(all_collisions(obstacles, x, y)) or
      x < @min_x or x > @max_x or y < 0 or y > @max_y
  end
end
