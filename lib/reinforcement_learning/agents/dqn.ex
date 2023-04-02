defmodule ReinforcementLearning.Agents.DQN do
  import Nx.Defn

  @behaviour ReinforcementLearning.Agent

  @learning_rate 1.0e-3
  @adamw_decay 1.0e-2
  @eps 1.0e-7
  @experience_replay_buffer_num_entries 10_000

  @eps_start 1
  @eps_decay_rate 0.995
  @eps_end 0.01

  @train_every_steps 32
  @adamw_decay 0.01

  @batch_size 128

  @gamma 0.99
  @tau 0.001

  @derive {Nx.Container,
           containers: [
             :q_policy,
             :q_target,
             :q_policy_optimizer_state,
             :loss,
             :loss_denominator,
             :experience_replay_buffer,
             :experience_replay_buffer_index,
             :persisted_experience_replay_buffer_entries,
             :total_reward,
             :epsilon_greedy_eps
           ],
           keep: [
             :optimizer_update_fn,
             :policy_predict_fn,
             :input_template,
             :state_vector_size,
             :num_actions,
             :environment_to_input_fn,
             :environment_to_state_vector_fn,
             :state_vector_to_input_fn,
             :learning_rate,
             :batch_size,
             :training_frequency,
             :target_training_frequency,
             :gamma,
             :eps_start,
             :eps_end,
             :decay_rate
           ]}
  defstruct [
    :state_vector_size,
    :num_actions,
    :q_policy,
    :q_target,
    :q_policy_optimizer_state,
    :policy_predict_fn,
    :optimizer_update_fn,
    :experience_replay_buffer,
    :experience_replay_buffer_index,
    :persisted_experience_replay_buffer_entries,
    :loss,
    :loss_denominator,
    :total_reward,
    :environment_to_input_fn,
    :environment_to_state_vector_fn,
    :state_vector_to_input_fn,
    :input_template,
    :learning_rate,
    :batch_size,
    :training_frequency,
    :target_training_frequency,
    :gamma,
    :epsilon_greedy_eps,
    :eps_start,
    :eps_end,
    :decay_rate
  ]

  @impl true
  def init(random_key, opts \\ []) do
    opts =
      Keyword.validate!(opts, [
        :q_policy,
        :q_target,
        :policy_net,
        :experience_replay_buffer,
        :experience_replay_buffer_index,
        :persisted_experience_replay_buffer_entries,
        :environment_to_input_fn,
        :environment_to_state_vector_fn,
        :state_vector_to_input_fn,
        target_training_frequency: @train_every_steps * 4,
        learning_rate: @learning_rate,
        batch_size: @batch_size,
        training_frequency: @train_every_steps,
        gamma: @gamma,
        eps_start: @eps_start,
        eps_end: @eps_end,
        eps_decay_rate: @eps_decay_rate
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
      Axon.Updates.clip_by_global_norm()
      |> Axon.Updates.compose(
        Axon.Optimizers.adamw(@learning_rate, eps: @eps, decay: @adamw_decay)
      )

    initial_q_policy_state = opts[:q_policy] || raise "missing initial q_policy"
    initial_q_target_state = opts[:q_target] || initial_q_policy_state

    input_template = input_template(policy_net)

    q_policy = policy_init_fn.(input_template, initial_q_policy_state)
    q_target = policy_init_fn.(input_template, initial_q_target_state)

    q_policy_optimizer_state = optimizer_init_fn.(q_policy)

    {1, num_actions} = Axon.get_output_shape(policy_net, input_template)

    state_vector_size = state_vector_size(input_template)

    loss = loss_denominator = Nx.tensor(0, type: :f32)

    eps_start = opts[:eps_start]
    eps_end = opts[:eps_end]
    decay_rate = opts[:eps_decay_rate]

    state = %__MODULE__{
      epsilon_greedy_eps: eps_start,
      eps_start: eps_start,
      eps_end: eps_end,
      decay_rate: decay_rate,
      learning_rate: opts[:learning_rate],
      batch_size: opts[:batch_size],
      training_frequency: opts[:training_frequency],
      target_training_frequency: opts[:target_training_frequency],
      gamma: opts[:gamma],
      loss: loss,
      loss_denominator: loss_denominator,
      state_vector_size: state_vector_size,
      num_actions: num_actions,
      input_template: input_template,
      environment_to_input_fn: environment_to_input_fn,
      environment_to_state_vector_fn: environment_to_state_vector_fn,
      state_vector_to_input_fn: state_vector_to_input_fn,
      q_policy: q_policy,
      q_policy_optimizer_state: q_policy_optimizer_state,
      q_target: q_target,
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
    }

    {state, random_key}
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
  def reset(random_key, %ReinforcementLearning{agent_state: state, episode: episode}) do
    total_reward = loss = loss_denominator = Nx.tensor(0, type: :f32)

    eps = Nx.max(Nx.multiply(state.eps_start, Nx.pow(state.decay_rate, episode)), state.eps_end)

    {%{
       state
       | epsilon_greedy_eps: eps,
         total_reward: total_reward,
         loss: loss,
         loss_denominator: loss_denominator
     }, random_key}
  end

  @impl true
  defn select_action(
         %ReinforcementLearning{random_key: random_key, agent_state: agent_state} = state,
         _iteration
       ) do
    %{
      q_policy: q_policy,
      policy_predict_fn: policy_predict_fn,
      environment_to_input_fn: environment_to_input_fn,
      num_actions: num_actions,
      epsilon_greedy_eps: eps_threshold
    } = agent_state

    {sample, random_key} = Nx.Random.uniform(random_key)

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
             environment_to_state_vector_fn: as_state_vector_fn,
             gamma: gamma
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
        policy_predict_fn.(q_policy, state_vector_to_input_fn.(next_state_vector)) * gamma *
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
      experience_replay_buffer_index: experience_replay_buffer_index,
      batch_size: batch_size,
      training_frequency: training_frequency,
      target_training_frequency: target_training_frequency
    } = state.agent_state

    has_at_least_one_batch = persisted_experience_replay_buffer_entries > batch_size
    should_update_policy_net = rem(experience_replay_buffer_index, training_frequency) == 0
    should_update_target_net = rem(experience_replay_buffer_index, target_training_frequency) == 0

    cond do
      not has_at_least_one_batch ->
        state

      should_update_policy_net and not should_update_target_net ->
        update_policy_network(state)

      should_update_policy_net and should_update_target_net ->
        state
        |> update_policy_network()
        |> soft_update_target_network()

      true ->
        state
    end
  end

  defnp update_policy_network(state) do
    %{
      agent_state: %{
        q_policy: q_policy,
        q_target: q_target,
        q_policy_optimizer_state: q_policy_optimizer_state,
        policy_predict_fn: policy_predict_fn,
        optimizer_update_fn: optimizer_update_fn,
        state_vector_to_input_fn: state_vector_to_input_fn,
        state_vector_size: state_vector_size,
        experience_replay_buffer: experience_replay_buffer,
        gamma: gamma
      },
      random_key: random_key
    } = state

    {batch, batch_idx, random_key} =
      sample_experience_replay_buffer(random_key, state.agent_state)

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

    {{experience_replay_buffer, loss}, gradient} =
      value_and_grad(
        q_policy,
        fn q_policy ->
          action_idx = Nx.as_type(action_batch, :s64)

          %{shape: {m, 1}} =
            state_action_values =
            q_policy
            |> policy_predict_fn.(state_batch)
            |> Nx.take_along_axis(action_idx, axis: 1)

          expected_state_action_values =
            reward_batch +
              policy_predict_fn.(q_target, next_state_batch) * gamma * non_final_mask

          %{shape: {n, 1}} =
            expected_state_action_values =
            Nx.reduce_max(expected_state_action_values, axes: [-1], keep_axes: true)

          case {m, n} do
            {m, n} when m != n ->
              raise "shape mismatch for batch values"

            _ ->
              1
          end

          td_errors = Nx.abs(expected_state_action_values - state_action_values)

          {
            update_priorities(
              experience_replay_buffer,
              batch_idx,
              state_vector_size * 2 + 3,
              td_errors
            ),
            huber_loss(expected_state_action_values, state_action_values)
            # Axon.Losses.mean_squared_error(expected_state_action_values, state_action_values,
            #   reduction: :mean
            # )
          }
        end,
        &elem(&1, 1)
      )

    {scaled_updates, optimizer_state} =
      optimizer_update_fn.(gradient, q_policy_optimizer_state, q_policy)

    q_policy = Axon.Updates.apply_updates(q_policy, scaled_updates)

    %{
      state
      | agent_state: %{
          state.agent_state
          | q_policy: q_policy,
            q_policy_optimizer_state: optimizer_state,
            loss: state.agent_state.loss + loss,
            loss_denominator: state.agent_state.loss_denominator + 1,
            experience_replay_buffer: experience_replay_buffer
        },
        random_key: random_key
    }
  end

  defnp soft_update_target_network(state) do
    %{agent_state: %{q_target: q_target, q_policy: q_policy} = agent_state} = state

    q_target = Axon.Shared.deep_merge(q_policy, q_target, &(&1 * @tau + &2 * (1 - @tau)))

    %{state | agent_state: %{agent_state | q_target: q_target}}
  end

  @alpha 0.6
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

    priorities = temporal_difference ** @alpha
    probs = priorities / Nx.sum(priorities)

    {batch_idx, random_key} =
      Nx.Random.choice(random_key, Nx.iota(temporal_difference.shape), probs,
        samples: @batch_size,
        replace: false,
        axis: 0
      )

    batch = Nx.take(exp_replay_buffer, batch_idx)
    {batch, batch_idx, random_key}
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

  defn update_priorities(
         buffer,
         %{shape: {n}} = row_idx,
         target_column,
         td_errors
       ) do
    case td_errors.shape do
      {^n, 1} -> :ok
      shape -> raise "invalid shape for td_errors, got: #{inspect(shape)}"
    end

    indices = Nx.stack([row_idx, Nx.broadcast(target_column, {n})], axis: -1)

    Nx.indexed_put(buffer, indices, Nx.reshape(td_errors, {n}))
  end

  defnp huber_loss(y_true, y_pred, opts \\ [delta: 1.0]) do
    delta = opts[:delta]

    abs_diff = Nx.abs(y_pred - y_true)

    (abs_diff <= delta)
    |> Nx.select(0.5 * abs_diff ** 2, delta * abs_diff - 0.5 * delta ** 2)
    |> Nx.mean()
  end
end
