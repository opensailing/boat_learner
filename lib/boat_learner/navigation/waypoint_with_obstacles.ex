defmodule BoatLearner.Navigation.WaypointWithObstacles do
  @moduledoc """
  Training and execution of Q-Learning Model

  Objective: sail from the bottom of the grid to the top of the grid with north-to-south wind.
  """
  import Nx.Defn

  @state_vector_size 15
  # the boat can see a square of side length 7 centered around it
  @visibility_grid_w 31
  @visibility_grid_h 31
  @experience_replay_buffer_num_entries 10_000
  @max_iter 4000
  @headings_history_len 10

  @derive {Nx.Container,
           containers: [
             :random_key,
             :grid,
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
             :persisted_experience_replay_buffer_entries,
             :reward_stage,
             :headings_history,
             :speed,
             :previous_speed,
             :previous_x,
             :previous_y,
             :previous_angle,
             :loss,
             :total_reward
           ],
           keep: [
             :velocity_model
           ]}
  defstruct [
    :velocity_model,
    :random_key,
    :grid,
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
      Nx.template(
        {@experience_replay_buffer_num_entries,
         @state_vector_size * 2 + @visibility_grid_h * @visibility_grid_h * 2 + 2},
        :f32
      ),
    experience_replay_buffer_index: Nx.template({}, :s64),
    persisted_experience_replay_buffer_entries: Nx.template({}, :s64),
    reward_stage: Nx.template({}, :s64),
    headings_history: Nx.template({@headings_history_len}, :f32),
    speed: Nx.template({}, :f32),
    previous_speed: Nx.template({}, :f32),
    previous_x: Nx.template({}, :f32),
    previous_y: Nx.template({}, :f32),
    previous_angle: Nx.template({}, :f32),
    loss: Nx.template({@batch_size, 1}, :f32),
    total_reward: Nx.template({}, :f32)
  ]

  @dt 2.5
  @momentum 0
  @pi :math.pi()

  @min_x -250
  @max_x 250
  @min_y 0
  @max_y 500

  @eps_start 0.9
  @eps_end 0.05
  @eps_decay 1000
  @gamma 0.99
  @batch_size 128
  @train_every_steps 16
  @update_target_every_steps 128
  @tau 0.01
  @learning_rate 1.0e-4
  @adamw_decay 0.01
  @eps 1.0e-7

  # turn left, turn right, keep_straight
  @num_actions 3
  @d_angle_rad 15 * @pi / 180

  @doc """
  Trains new network from scratch
  """
  def train(
        obstacles,
        target_waypoint,
        trajectory_callback \\ & &1,
        plot_reward_callback \\ & &1,
        reward_fn \\ &reward_for_state/1,
        opts \\ []
      ) do
    # obstacles is a list of [min_x, max_x, min_y, max_y] lists
    # target_waypoint is {2} [target_x, target_y]

    # This is fixed on {angle, x, y, vel_x, vel_x}
    # num_observations = 5

    policy_net = dqn(@state_vector_size, {@visibility_grid_w, @visibility_grid_h}, @num_actions)
    target_net = dqn(@state_vector_size, {@visibility_grid_w, @visibility_grid_h}, @num_actions)

    {policy_init_fn, policy_predict_fn} = Axon.build(policy_net, seed: 0)
    {target_init_fn, target_predict_fn} = Axon.build(target_net, seed: 1)

    {optimizer_init_fn, optimizer_update_fn} =
      Axon.Updates.clip_by_global_norm()
      |> Axon.Updates.compose(
        Axon.Optimizers.adamw(@learning_rate, eps: @eps, decay: @adamw_decay)
      )

    input = %{
      "state" => Nx.template({1, @state_vector_size}, :f32),
      "obstacles" => Nx.template({1, @visibility_grid_w, @visibility_grid_h, 1}, :u8)
    }

    initial_q_policy_state = opts[:q_policy] || raise "missing initial q_policy"
    initial_q_target_state = opts[:q_target] || raise "missing initial q_target"

    q_policy = policy_init_fn.(input, initial_q_policy_state)
    q_target = target_init_fn.(input, initial_q_target_state)

    q_policy_optimizer_state = optimizer_init_fn.(q_policy)
    q_target_optimizer_state = optimizer_init_fn.(q_target)

    random_key = Nx.Random.key(System.system_time())
    num_episodes = opts[:num_episodes] || 10000

    grid =
      obstacles
      |> to_obstacles_indices()
      |> build_grid()

    initial_state = %__MODULE__{
      random_key: random_key,
      grid: grid,
      target_waypoint: target_waypoint,
      q_policy_optimizer_state: q_policy_optimizer_state,
      q_target_optimizer_state: q_target_optimizer_state,
      q_policy: q_policy,
      q_target: q_target,
      loss: Nx.broadcast(Nx.tensor(0, type: :f32), {@batch_size, 1}),
      experience_replay_buffer:
        opts[:experience_replay_buffer] ||
          Nx.broadcast(
            Nx.tensor(:nan, type: :f32),
            {@experience_replay_buffer_num_entries,
             2 * @state_vector_size + 2 * (@visibility_grid_h * @visibility_grid_w) + 2}
          ),
      experience_replay_buffer_index:
        opts[:experience_replay_buffer_index] || Nx.tensor(0, type: :s64),
      persisted_experience_replay_buffer_entries:
        opts[:persisted_experience_replay_buffer_entries] || Nx.tensor(0, type: :s64)
    }

    loop =
      Axon.Loop.loop(
        &batch_step(&1, &2, reward_fn, policy_predict_fn, target_predict_fn, optimizer_update_fn),
        &init/2
      )

    loop
    |> Axon.Loop.handle(:epoch_started, &epoch_started_handler/1)
    |> Axon.Loop.handle(:epoch_completed, fn s ->
      s =
        %{s | epoch: s.epoch + 1}
        |> tap(trajectory_callback)
        |> tap(plot_reward_callback)

      {:continue, s}
    end)
    |> Axon.Loop.run(Stream.cycle([Nx.tensor(1)]), initial_state,
      iterations: 1,
      epochs: num_episodes
    )
  end

  def to_obstacles_indices(obstacles) do
    obstacles_idx =
      for [min_x, max_x, min_y, max_y] <- obstacles do
        Nx.tensor(
          for i <- min_x..max_x, j <- min_y..max_y do
            [i - @min_x, j - @min_y]
          end
        )
      end

    Nx.concatenate(obstacles_idx, axis: 0)
  end

  defp epoch_started_handler(loop_state) do
    # Reset state
    {:continue, %{loop_state | step_state: reset_variable_state(loop_state.step_state)}}
  end

  defn build_grid(obstacles_idx) do
    %{shape: {m, n}} =
      grid = Nx.broadcast(Nx.tensor(0, type: :u8), {@max_x - @min_x, @max_y - @min_y})

    idx_top =
      Nx.concatenate(
        [
          Nx.broadcast(0, {n, 1}),
          Nx.iota({n, 1})
        ],
        axis: 1
      )

    idx_bottom =
      Nx.concatenate(
        [
          Nx.broadcast(m - 1, {n, 1}),
          Nx.iota({n, 1})
        ],
        axis: 1
      )

    idx_left =
      Nx.concatenate(
        [
          Nx.iota({m, 1}),
          Nx.broadcast(0, {m, 1})
        ],
        axis: 1
      )

    idx_right =
      Nx.concatenate(
        [
          Nx.iota({m, 1}),
          Nx.broadcast(n - 1, {m, 1})
        ],
        axis: 1
      )

    idx = Nx.concatenate([idx_top, idx_bottom, idx_left, idx_right], axis: 0)
    idx = Nx.concatenate([idx, obstacles_idx])

    updates = Nx.broadcast(Nx.tensor(1, type: :u8), {Nx.axis_size(idx, 0)})

    Nx.indexed_put(grid, idx, updates)
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

        {reward, reward_stage} = reward_fn.(next_state)

        next_state =
          prev_state
          |> record_observation(action, reward, next_state)
          |> optimize_model(policy_predict_fn, target_predict_fn, optimizer_update_fn)
          |> soft_update_target_network()
          |> persist_trajectory()

        next_state = %{
          next_state
          | reward_stage: reward_stage,
            total_reward: next_state.total_reward + reward
        }

        is_terminal =
          next_state
          |> as_state_vector()
          |> Nx.new_axis(0)
          |> is_terminal_state(next_state.grid)
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
    total_reward = speed = y = Nx.tensor(0, type: :f32)

    {x, random_key} = Nx.Random.uniform(current_state.random_key, -50, 50)
    {angle, random_key} = Nx.Random.uniform(random_key, -5 * @d_angle_rad, 5 * @d_angle_rad)

    # these should always start at 0
    iteration = Nx.tensor(0, type: :s64)
    reward_stage = Nx.tensor(0, type: :s64)

    trajectory = Nx.broadcast(Nx.tensor(:nan, type: :f32), {@max_iter, 2})
    headings_history = Nx.broadcast(angle, {@headings_history_len})

    %{
      current_state
      | iteration: iteration,
        total_reward: total_reward,
        x: x,
        y: y,
        previous_x: x,
        previous_y: y,
        angle: angle,
        previous_angle: angle,
        trajectory: trajectory,
        random_key: random_key,
        reward_stage: reward_stage,
        headings_history: headings_history,
        speed: speed,
        previous_speed: speed
    }
  end

  defnp select_action_and_update_state(state, policy_predict_fn) do
    state
    |> select_action(policy_predict_fn)
    |> update_state_from_action()
  end

  defnp update_state_from_action({action, state}) do
    next_angle = next_angle_from_action(state.angle, action)

    {next_x, next_y, speed, next_angle} =
      BoatLearner.Simulator.update_state(
        state.velocity_model,
        state.x,
        state.y,
        state.angle,
        next_angle,
        @dt,
        @momentum
      )

    headings_history =
      Nx.concatenate([
        Nx.slice(state.headings_history, [0], [@headings_history_len - 1]),
        Nx.reshape(next_angle, {1})
      ])

    new_state = %{
      state
      | x: next_x,
        y: next_y,
        previous_x: state.x,
        previous_y: state.y,
        speed: speed,
        previous_speed: state.speed,
        angle: next_angle,
        previous_angle: state.angle,
        headings_history: headings_history
    }

    {action, new_state}
  end

  defnp next_angle_from_action(angle, action) do
    cond do
      action == 0 ->
        angle

      action == 1 ->
        angle + @d_angle_rad

      true ->
        angle - @d_angle_rad
    end
  end

  defnp select_action(state, policy_predict_fn) do
    %{
      random_key: random_key,
      iteration: iteration,
      q_policy: q_policy
    } = state

    {sample, random_key} = Nx.Random.uniform(random_key)

    eps_threshold = @eps_end + (@eps_start - @eps_end) * Nx.exp(-1 * iteration / @eps_decay)

    {action, random_key} =
      if sample > eps_threshold do
        {visibility_grid, random_key} =
          as_visibility_grid(%{state | random_key: random_key}, standardize: true)

        action =
          q_policy
          |> policy_predict_fn.(%{
            "state" => as_state_vector(state) |> Nx.new_axis(0),
            "obstacles" => visibility_grid
          })
          |> Nx.argmax()

        {action, random_key}
      else
        Nx.Random.randint(random_key, 0, @num_actions, type: :s64)
      end

    new_state = %{state | random_key: random_key}
    {action, new_state}
  end

  defnp as_state_vector(%{
          x: x,
          y: y,
          angle: angle,
          speed: speed,
          target_waypoint: target_waypoint,
          previous_x: previous_x,
          previous_y: previous_y,
          previous_speed: previous_speed,
          previous_angle: previous_angle,
          reward_stage: reward_stage
        }) do
    x = (x - @min_x) / (@max_x - @min_x)
    y = (y - @min_y) / (@max_y - @min_y)

    speed = speed / 1000

    target_x = (target_waypoint[0] - @min_x) / (@max_x - @min_x)
    target_y = (target_waypoint[1] - @min_y) / (@max_y - @min_y)
    previous_x = (previous_x - @min_x) / (@max_x - @min_x)
    previous_y = (previous_y - @min_y) / (@max_y - @min_y)

    previous_speed = previous_speed / 1000

    normalized_speed_vector_diff =
      Nx.sqrt(
        (speed * Nx.cos(angle) - previous_speed * Nx.cos(previous_angle)) ** 2 +
          (speed * Nx.sin(angle) - previous_speed * Nx.sin(previous_angle)) ** 2
      )

    previous_angle = (previous_angle - @pi) / (2 * @pi)
    angle = (angle - @pi) / (2 * @pi)

    Nx.stack([
      x,
      y,
      angle,
      speed,
      target_x,
      target_y,
      previous_x,
      previous_y,
      previous_speed,
      previous_angle,
      # current distance
      Nx.sqrt((x - target_x) ** 2 + (y - target_y) ** 2) / Nx.sqrt(2),
      # previous distance
      Nx.sqrt((previous_x - target_x) ** 2 + (previous_y - target_y) ** 2) / Nx.sqrt(2),
      # absolute difference from the previous angle (for tacking)
      Nx.abs(angle - previous_angle),
      # vector difference for the speed
      normalized_speed_vector_diff,
      reward_stage
    ])
  end

  @grid_offset_x div(@visibility_grid_w, 2)
  @grid_offset_y div(@visibility_grid_h, 2)
  defn as_visibility_grid(state, opts \\ []) do
    opts = keyword!(opts, standardize: true)

    xy_offset =
      Nx.stack([state.x - @min_x, state.y - @min_y]) |> Nx.ceil() |> Nx.as_type({:s, 64})

    idx =
      [
        Nx.iota({@visibility_grid_w, @visibility_grid_h}, axis: 0),
        Nx.iota({@visibility_grid_w, @visibility_grid_h}, axis: 1)
      ]
      |> Nx.stack(axis: -1)
      |> Nx.reshape({:auto, 2})
      # shift so that we have the grid centered instead of having the origin at top left
      |> Nx.subtract(Nx.tensor([[@grid_offset_x, @grid_offset_y]]))
      # now re-center at current x and y
      |> Nx.add(xy_offset)
      # lower bound the [x, y] indices to [min_x, min_y]
      |> Nx.max(Nx.tensor([[@min_x, @min_y]]))
      # upper bound the [x, y] indices to [min_x, min_y]
      |> Nx.min(Nx.tensor([[@max_x, @max_y]]))

    # Note: we are selecting repeat indices, but this is ok because this will indicate that
    # the grid has ended. It's as if the grid border is composed of infinite obstacles that
    # extend outward.

    visibility_grid =
      state.grid
      |> Nx.gather(idx)
      |> Nx.reshape({1, @visibility_grid_w, @visibility_grid_h, 1})

    zero_mean = visibility_grid - Nx.mean(visibility_grid)

    # return a standardized grid
    stddev = Nx.standard_deviation(zero_mean)

    # {grid, random_key} =
    if opts[:standardize] do
      if stddev < @eps do
        {normal, key} = Nx.Random.normal(state.random_key, shape: Nx.shape(visibility_grid))
        {normal * 0.1, key}
      else
        {zero_mean / stddev, state.random_key}
      end
    else
      {visibility_grid, state.random_key}
    end
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
    if rem(state.persisted_experience_replay_buffer_entries, @train_every_steps) == 0 do
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

    # TO-DO: extract and reshape visibility grid in sequence here
    # It goes: state, visibility grid, action, reward, next_state, next_visibility_grid
    state_batch = Nx.slice_along_axis(batch, 0, @state_vector_size, axis: 1)

    grid_batch =
      Nx.slice_along_axis(batch, @state_vector_size, @visibility_grid_h * @visibility_grid_w,
        axis: 1
      )
      |> Nx.reshape({:auto, @visibility_grid_w, @visibility_grid_h, 1})

    action_batch =
      Nx.slice_along_axis(batch, @state_vector_size + @visibility_grid_h * @visibility_grid_w, 1,
        axis: 1
      )

    reward_batch =
      Nx.slice_along_axis(
        batch,
        @state_vector_size + @visibility_grid_h * @visibility_grid_w + 1,
        1,
        axis: 1
      )

    next_state_batch =
      Nx.slice_along_axis(
        batch,
        @state_vector_size + @visibility_grid_h * @visibility_grid_w + 2,
        @state_vector_size,
        axis: 1
      )

    next_grid_batch =
      Nx.slice_along_axis(
        batch,
        @state_vector_size + @visibility_grid_h * @visibility_grid_w + 2 + @state_vector_size,
        @visibility_grid_h * @visibility_grid_w,
        axis: 1
      )
      |> Nx.reshape({:auto, @visibility_grid_w, @visibility_grid_h, 1})

    non_final_mask = not is_terminal_state(state_batch, state.grid)

    {loss, gradient} =
      value_and_grad(state.q_policy, fn q_policy ->
        action_idx = Nx.as_type(action_batch, :s64)

        state_action_values =
          q_policy
          |> policy_predict_fn.(%{"state" => state_batch, "obstacles" => grid_batch})
          |> Nx.take_along_axis(action_idx, axis: 1)

        next_state_values =
          state.q_target
          |> target_predict_fn.(%{
            "state" => next_state_batch,
            "obstacles" => next_grid_batch
          })
          |> Nx.argmax(axis: 1, keep_axis: true)

        expected_state_action_values =
          Nx.select(
            Nx.reshape(non_final_mask, {:auto, 1}),
            next_state_values * @gamma + reward_batch,
            reward_batch
          )

        # Axon.Losses.mean_squared_error(expected_state_action_values, state_action_values)
        huber_loss(expected_state_action_values, state_action_values)
      end)

    {scaled_updates, optimizer_state} =
      optimizer_update_fn.(gradient, state.q_policy_optimizer_state, state.q_policy)

    q_policy = Axon.Updates.apply_updates(state.q_policy, scaled_updates)

    %{
      state
      | q_policy: q_policy,
        q_policy_optimizer_state: optimizer_state,
        random_key: random_key,
        loss: loss
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

  defnp huber_loss(y_true, y_pred, opts \\ [delta: 1.0]) do
    delta = opts[:delta]
    abs_diff = Nx.abs(y_pred - y_true)

    (abs_diff <= delta)
    |> Nx.select(0.5 * abs_diff ** 2, delta * abs_diff - 0.5 * delta ** 2)
    |> Nx.mean(axes: [-1], keep_axes: true)
  end

  defnp soft_update_target_network(state) do
    if rem(state.persisted_experience_replay_buffer_entries, @update_target_every_steps) == 0 do
      q_target =
        Axon.Shared.deep_merge(state.q_policy, state.q_target, &(&1 * @tau + &2 * (1 - @tau)))

      %{state | q_target: q_target}
    else
      state
    end
  end

  defnp record_observation(prev_state, action, reward, state) do
    state_vector = as_state_vector(prev_state)
    {visibility_grid, _random_key} = as_visibility_grid(prev_state, standardize: true)
    next_state_vector = as_state_vector(state)
    {next_visibility_grid, _random_key} = as_visibility_grid(state, standardize: true)

    idx = Nx.stack([state.experience_replay_buffer_index, 0]) |> Nx.new_axis(0)

    shape =
      {Nx.size(state_vector) + Nx.size(visibility_grid) + 2 + Nx.size(next_state_vector) +
         Nx.size(next_visibility_grid), 1}

    index_template = Nx.concatenate([Nx.broadcast(0, shape), Nx.iota(shape, axis: 0)], axis: 1)

    updates =
      Nx.concatenate([
        state_vector,
        Nx.flatten(visibility_grid),
        Nx.stack([action, reward]),
        next_state_vector,
        Nx.flatten(next_visibility_grid)
      ])

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
        experience_replay_buffer_index: experience_replay_buffer_index,
        persisted_experience_replay_buffer_entries:
          state.persisted_experience_replay_buffer_entries + 1
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

  defn(reached_target(x, y, target_x, target_y),
    do: Nx.all(distance(x, y, target_x, target_y) < 2.5)
  )

  defnp is_terminal_state(state_vectors, grid) do
    # state_vectors: {n, state_vector_size}
    # state vector: [x, y, angle, speed, target_waypoint[0], target_waypoint[1]]

    n = Nx.axis_size(state_vectors, 0)
    is_terminal = Nx.broadcast(Nx.tensor(0, type: :u8), {n})

    {is_terminal, _i, _n, _state_vectors, _grid} =
      while {is_terminal, i = Nx.tensor(0), n, state_vectors, grid}, i < n do
        state_vector = state_vectors[i]

        is_term =
          do_is_terminal_state(
            Nx.take(state_vector, Nx.tensor([0])) * (@max_x - @min_x) + @min_x,
            Nx.take(state_vector, Nx.tensor([1])) * (@max_y - @min_y) + @min_y,
            Nx.take(state_vector, Nx.tensor([4])) * (@max_x - @min_x) + @min_x,
            Nx.take(state_vector, Nx.tensor([5])) * (@max_y - @min_y) + @min_y,
            grid
          )

        is_terminal = Nx.indexed_put(is_terminal, Nx.reshape(i, {1, 1}), Nx.reshape(is_term, {1}))
        {is_terminal, i + 1, n, state_vectors, grid}
      end

    is_terminal
  end

  defnp do_is_terminal_state(x, y, target_waypoint_x, target_waypoint_y, grid) do
    x_clipped =
      x
      |> Nx.subtract(@min_x)
      |> Nx.ceil()
      |> Nx.as_type(:s64)
      |> Nx.clip(0, @max_x - @min_x)
      |> Nx.reshape({})

    y_clipped =
      y
      |> Nx.subtract(@min_y)
      |> Nx.ceil()
      |> Nx.as_type(:s64)
      |> Nx.clip(0, @max_y - @min_y)
      |> Nx.reshape({})

    distance = Nx.sqrt((target_waypoint_x - x) ** 2 + (target_waypoint_y - y) ** 2)

    # terminates on collision or if target is close enough
    grid[x_clipped][y_clipped] != 0 or distance < 20
  end

  def dqn(num_observations, {w, h}, num_actions) when w >= 9 and h >= 9 do
    # This NN is designed so that the state also includes
    # an observation matrix around the boat that encodes obstacles

    # obstacles_detection =
    #   Axon.input("obstacles", shape: {nil, w, h, 1})
    #   |> Axon.conv(8,
    #     activation: :leaky_relu,
    #     kernel_size: {3, 3},
    #     padding: :same
    #   )
    #   |> Axon.batch_norm(epsilon: @eps)
    #   |> Axon.max_pool(kernel_size: {2, 2})
    #   # |> Axon.dropout(rate: 0.1)
    #   |> Axon.conv(8,
    #     activation: :leaky_relu,
    #     kernel_size: {3, 3}
    #   )
    #   |> Axon.batch_norm(epsilon: @eps)
    #   |> Axon.max_pool(kernel_size: {2, 2})
    #   # |> Axon.dropout(rate: 0.1)
    #   |> Axon.flatten()
    #   |> Axon.dense(32, activation: :relu)
    #   |> Axon.dense(16, activation: :relu)
    #   |> Axon.dense(8, activation: :relu)

    # shape is currently ignored
    state_detection =
      Axon.input("state", shape: {nil, num_observations})
      |> Axon.dense(128, activation: :relu)
      |> Axon.dropout(rate: 0.5)
      |> Axon.dense(64, activation: :relu)
      |> Axon.dropout(rate: 0.5)

    # Axon.concatenate([state_detection, obstacles_detection])
    state_detection
    |> Axon.dense(64, activation: :relu)
    |> Axon.dropout(rate: 0.5)
    |> Axon.dense(32, activation: :relu)
    |> Axon.dense(num_actions)
  end
end
