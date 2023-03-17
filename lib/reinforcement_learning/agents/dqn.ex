defmodule ReinforcementLearning.Agents.DQN do
  import Nx.Defn

  @behaviour ReinforcementLearning.Agent

  @learning_rate 1.0e-5
  @adamw_decay 1.0e-2
  @eps 1.0e-7
  @experience_replay_buffer_num_entries 10_000
  @experience_replay_buffer_num_entries_on_4 div(@experience_replay_buffer_num_entries, 4)

  @eps_start 0.996
  @eps_end 0.01

  @train_every_steps 64
  @adamw_decay 0.01

  @batch_size 256
  @batch_size_on_4 div(@batch_size, 4)

  @gamma 0.99

  @derive {Nx.Container,
           containers: [
             :q_policy,
             :q_policy_optimizer_state,
             :loss,
             :experience_replay_buffer,
             :experience_replay_buffer_index,
             :persisted_experience_replay_buffer_entries,
             :total_reward,
             :eps_max_iter
           ],
           keep: [
             :optimizer_update_fn,
             :policy_predict_fn,
             :input_template,
             :state_vector_size,
             :num_actions,
             :environment_to_input_fn,
             :environment_to_state_vector_fn,
             :state_vector_to_input_fn
           ]}
  defstruct [
    :state_vector_size,
    :num_actions,
    :q_policy,
    :q_policy_optimizer_state,
    :policy_predict_fn,
    :optimizer_update_fn,
    :experience_replay_buffer,
    :experience_replay_buffer_index,
    :persisted_experience_replay_buffer_entries,
    :loss,
    :total_reward,
    :eps_max_iter,
    :environment_to_input_fn,
    :environment_to_state_vector_fn,
    :state_vector_to_input_fn,
    :input_template
  ]

  @impl true
  def init(random_key, opts \\ []) do
    opts =
      Keyword.validate!(opts, [
        :q_policy,
        :policy_net,
        :experience_replay_buffer,
        :experience_replay_buffer_index,
        :persisted_experience_replay_buffer_entries,
        :eps_max_iter,
        :environment_to_input_fn,
        :environment_to_state_vector_fn,
        :state_vector_to_input_fn
      ])

    policy_net = opts[:policy_net] || raise ArgumentError, "missing :policy_net option"

    environment_to_input_fn =
      opts[:environment_to_input_fn] ||
        raise ArgumentError, "missing :environment_to_input_fn option"

    environment_to_state_vector_fn =
      opts[:environment_to_state_vector_fn] ||
        raise ArgumentError, "missing :environment_to_state_vector_fn option"

    state_vector_to_input_fn =
      opts[:state_vector_to_input_fn] ||
        raise ArgumentError, "missing :state_vector_to_input_fn option"

    {policy_init_fn, policy_predict_fn} = Axon.build(policy_net, seed: 0)

    {optimizer_init_fn, optimizer_update_fn} =
      Axon.Updates.clip()
      |> Axon.Updates.compose(
        Axon.Optimizers.adamw(@learning_rate, eps: @eps, decay: @adamw_decay)
      )

    initial_q_policy_state = opts[:q_policy] || raise "missing initial q_policy"
    eps_max_iter = opts[:eps_max_iter] || raise "missing :eps_max_iter"

    input_template = input_template(policy_net)

    q_policy = policy_init_fn.(input_template, initial_q_policy_state)

    q_policy_optimizer_state = optimizer_init_fn.(q_policy)

    {1, num_actions} = Axon.get_output_shape(policy_net, input_template)

    state_vector_size = state_vector_size(input_template)

    loss = Nx.tensor(0, type: :f32)

    reset(random_key, %__MODULE__{
      eps_max_iter: eps_max_iter,
      loss: loss,
      state_vector_size: state_vector_size,
      num_actions: num_actions,
      input_template: input_template,
      environment_to_input_fn: environment_to_input_fn,
      environment_to_state_vector_fn: environment_to_state_vector_fn,
      state_vector_to_input_fn: state_vector_to_input_fn,
      q_policy: q_policy,
      q_policy_optimizer_state: q_policy_optimizer_state,
      policy_predict_fn: policy_predict_fn,
      optimizer_update_fn: optimizer_update_fn,
      # prev_state_vector, target_x, target_y, action, reward, is_terminal, next_state_vector
      experience_replay_buffer:
        opts[:experience_replay_buffer] ||
          Nx.broadcast(
            Nx.tensor(:nan, type: :f32),
            {@experience_replay_buffer_num_entries, 2 * state_vector_size + 4}
          ),
      experience_replay_buffer_index:
        opts[:experience_replay_buffer_index] || Nx.tensor(0, type: :s64),
      persisted_experience_replay_buffer_entries:
        opts[:persisted_experience_replay_buffer_entries] || Nx.tensor(0, type: :s64)
    })
  end

  defp input_template(model) do
    model
    |> Axon.get_inputs()
    |> Map.new(fn {name, shape} ->
      [nil | shape] = Tuple.to_list(shape)
      shape = List.to_tuple([1 | shape])
      {name, Nx.template(shape, :f32)}
    end)
  end

  defp state_vector_size(input_template) do
    Enum.reduce(input_template, 0, fn {_field, tensor}, acc ->
      div(Nx.size(tensor), Nx.axis_size(tensor, 0)) + acc
    end)
  end

  @impl true
  def reset(random_key, state) do
    total_reward = Nx.tensor(0, type: :f32)
    {%{state | total_reward: total_reward}, random_key}
  end

  @impl true
  defn select_action(
         %ReinforcementLearning{random_key: random_key, agent_state: agent_state} = state,
         iteration
       ) do
    %{
      q_policy: q_policy,
      policy_predict_fn: policy_predict_fn,
      environment_to_input_fn: environment_to_input_fn,
      num_actions: num_actions,
      eps_max_iter: eps_max_iter
    } = agent_state

    {sample, random_key} = Nx.Random.uniform(random_key)

    eps_threshold =
      Nx.select(
        eps_max_iter > 0,
        @eps_end + (@eps_start - @eps_end) * Nx.exp(-5 * iteration / eps_max_iter),
        -1
      )

    {action, random_key} =
      if sample > eps_threshold do
        action =
          q_policy
          |> policy_predict_fn.(environment_to_input_fn.(state.environment_state))
          |> Nx.argmax()

        {action, random_key}
      else
        Nx.Random.randint(random_key, 0, num_actions, type: :s64)
      end

    {action, %{state | random_key: random_key}}
  end

  @impl true
  defn record_observation(
         %{
           environment_state: env_state,
           agent_state: %{
             q_policy: q_policy,
             policy_predict_fn: policy_predict_fn,
             state_vector_to_input_fn: state_vector_to_input_fn,
             environment_to_state_vector_fn: as_state_vector_fn
           }
         },
         action,
         reward,
         is_terminal,
         %{environment_state: next_env_state} = state
       ) do
    state_vector = as_state_vector_fn.(env_state)
    next_state_vector = as_state_vector_fn.(next_env_state)

    idx = Nx.stack([state.agent_state.experience_replay_buffer_index, 0]) |> Nx.new_axis(0)

    shape = {Nx.size(state_vector) + 4 + Nx.size(next_state_vector), 1}

    index_template = Nx.concatenate([Nx.broadcast(0, shape), Nx.iota(shape, axis: 0)], axis: 1)

    predicted_reward =
      reward +
        policy_predict_fn.(q_policy, state_vector_to_input_fn.(next_state_vector)) * @gamma *
          (1 - is_terminal)

    %{shape: {1}} = predicted_reward = Nx.reduce_max(predicted_reward, axes: [-1])

    temporal_difference = Nx.reshape(Nx.abs(reward - predicted_reward), {1})

    updates =
      Nx.concatenate([
        Nx.flatten(state_vector),
        Nx.stack([action, reward, is_terminal]),
        Nx.flatten(next_state_vector),
        temporal_difference
      ])

    experience_replay_buffer =
      Nx.indexed_put(state.agent_state.experience_replay_buffer, idx + index_template, updates)

    experience_replay_buffer_index =
      Nx.remainder(
        state.agent_state.experience_replay_buffer_index + 1,
        @experience_replay_buffer_num_entries
      )

    entries = state.agent_state.persisted_experience_replay_buffer_entries

    persisted_experience_replay_buffer_entries =
      Nx.select(
        entries < @experience_replay_buffer_num_entries,
        entries + 1,
        entries
      )

    %{
      state
      | agent_state: %{
          state.agent_state
          | experience_replay_buffer: experience_replay_buffer,
            experience_replay_buffer_index: experience_replay_buffer_index,
            persisted_experience_replay_buffer_entries:
              persisted_experience_replay_buffer_entries,
            total_reward: state.agent_state.total_reward + reward
        }
    }
  end

  @impl true
  defn optimize_model(state) do
    %{
      persisted_experience_replay_buffer_entries: persisted_experience_replay_buffer_entries,
      experience_replay_buffer_index: experience_replay_buffer_index
    } = state.agent_state

    if persisted_experience_replay_buffer_entries > @batch_size and
         rem(experience_replay_buffer_index, @train_every_steps) == 0 do
      do_optimize_model(state)
    else
      state
    end
  end

  defnp do_optimize_model(state) do
    %{
      agent_state: %{
        q_policy: q_policy,
        q_policy_optimizer_state: q_policy_optimizer_state,
        policy_predict_fn: policy_predict_fn,
        optimizer_update_fn: optimizer_update_fn,
        state_vector_to_input_fn: state_vector_to_input_fn,
        state_vector_size: state_vector_size
      },
      random_key: random_key
    } = state

    {batch, random_key} = sample_experience_replay_buffer(random_key, state.agent_state)

    state_batch =
      batch
      |> Nx.slice_along_axis(0, state_vector_size, axis: 1)
      |> then(state_vector_to_input_fn)

    action_batch = Nx.slice_along_axis(batch, state_vector_size, 1, axis: 1)
    reward_batch = Nx.slice_along_axis(batch, state_vector_size + 1, 1, axis: 1)
    is_terminal_batch = Nx.slice_along_axis(batch, state_vector_size + 2, 1, axis: 1)

    next_state_batch =
      batch
      |> Nx.slice_along_axis(state_vector_size + 3, state_vector_size, axis: 1)
      |> then(state_vector_to_input_fn)

    non_final_mask = not is_terminal_batch

    {loss, gradient} =
      value_and_grad(q_policy, fn q_policy ->
        action_idx = Nx.as_type(action_batch, :s64)

        %{shape: {m, 1}} =
          state_action_values =
          q_policy
          |> policy_predict_fn.(state_batch)
          |> Nx.take_along_axis(action_idx, axis: 1)

        expected_state_action_values =
          reward_batch + policy_predict_fn.(q_policy, next_state_batch) * @gamma * non_final_mask

        %{shape: {n, 1}} =
          expected_state_action_values =
          Nx.reduce_max(expected_state_action_values, axes: [-1], keep_axes: true)

        case {m, n} do
          {m, n} when m != n ->
            raise "shape mismatch for batch values"

          _ ->
            1
        end

        huber_loss(expected_state_action_values, state_action_values)
      end)

    {scaled_updates, optimizer_state} =
      optimizer_update_fn.(gradient, q_policy_optimizer_state, q_policy)

    q_policy = Axon.Updates.apply_updates(q_policy, scaled_updates)

    %{
      state
      | agent_state: %{
          state.agent_state
          | q_policy: q_policy,
            q_policy_optimizer_state: optimizer_state,
            loss: (state.agent_state.loss + loss) / 2
        },
        random_key: random_key
    }
  end

  defnp sample_experience_replay_buffer(
          random_key,
          %{state_vector_size: state_vector_size} = agent_state
        ) do
    %{shape: {@experience_replay_buffer_num_entries, _}} =
      exp_replay_buffer = slice_experience_replay_buffer(agent_state)

    # Temporal Difference prioritizing:
    # We are going to sort experiences by temporal difference
    # and divide our buffer into 4 slices, from which we will
    # then uniformily sample.
    # The temporal difference is already stored in the end of our buffer.

    temporal_difference =
      exp_replay_buffer
      |> Nx.slice_along_axis(state_vector_size * 2 + 3, 1, axis: 1)
      |> Nx.flatten()

    idx = Nx.argsort(temporal_difference)

    top1 = Nx.slice_along_axis(idx, 0, @experience_replay_buffer_num_entries_on_4, axis: 0)

    top2 =
      Nx.slice_along_axis(
        idx,
        @experience_replay_buffer_num_entries_on_4,
        @experience_replay_buffer_num_entries_on_4,
        axis: 0
      )

    top3 =
      Nx.slice_along_axis(
        idx,
        2 * @experience_replay_buffer_num_entries_on_4,
        @experience_replay_buffer_num_entries_on_4,
        axis: 0
      )

    top4 =
      Nx.slice_along_axis(
        idx,
        3 * @experience_replay_buffer_num_entries_on_4,
        @experience_replay_buffer_num_entries_on_4,
        axis: 0
      )

    {batch1, random_key} =
      Nx.Random.choice(random_key, top1, samples: @batch_size_on_4, replace: false, axis: 0)

    {batch2, random_key} =
      Nx.Random.choice(random_key, top2, samples: @batch_size_on_4, replace: false, axis: 0)

    {batch3, random_key} =
      Nx.Random.choice(random_key, top3, samples: @batch_size_on_4, replace: false, axis: 0)

    {batch4, random_key} =
      Nx.Random.choice(random_key, top4, samples: @batch_size_on_4, replace: false, axis: 0)

    batch_idx = Nx.concatenate([batch1, batch2, batch3, batch4])
    batch = Nx.take(exp_replay_buffer, batch_idx)
    {batch, random_key}
  end

  defnp slice_experience_replay_buffer(state) do
    %{
      experience_replay_buffer: experience_replay_buffer,
      persisted_experience_replay_buffer_entries: entries
    } = state

    if entries < @experience_replay_buffer_num_entries do
      t = Nx.iota({@experience_replay_buffer_num_entries})
      idx = Nx.select(t < entries, t, 0)
      Nx.take(experience_replay_buffer, idx)
    else
      experience_replay_buffer
    end
  end

  defnp huber_loss(y_true, y_pred, opts \\ [delta: 1.0]) do
    delta = opts[:delta]

    abs_diff = Nx.abs(y_pred - y_true)

    (abs_diff <= delta)
    |> Nx.select(0.5 * abs_diff ** 2, delta * abs_diff - 0.5 * delta ** 2)
    |> Nx.mean()
  end
end
