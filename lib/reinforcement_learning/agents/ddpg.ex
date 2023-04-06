defmodule ReinforcementLearning.Agents.DDPG do
  @moduledoc """
  Deep Deterministic Policy Graditent agent
  """
  import Nx.Defn

  @behaviour ReinforcementLearning.Agent

  @derive {Nx.Container,
           containers: [
             :actor_params,
             :target_actor_params,
             :critic_params,
             :target_critic_params,
             :experience_replay_buffer,
             :experience_replay_buffer_index,
             :persisted_experience_replay_buffer_entries,
             :environment_to_state_vector_fn,
             :state_vector_to_input_fn,
             :gamma,
             :tau,
             :batch_size,
             :noise_stddev,
             :training_frequency,
             :target_update_frequency,
             :loss,
             :loss_denominator,
             :total_reward,
             :actor_optimizer_state,
             :critic_optimizer_state
           ],
           keep: [
             :experience_replay_buffer_max_size,
             :actor_predict_fn,
             :critic_predict_fn,
             :actor_update_fn,
             :critic_update_fn,
             :state_vector_size,
             :num_actions,
             :environment_to_input_fn,
             :actor_optimizer_update_fn,
             :critic_optimizer_update_fn
           ]}
  defstruct [
    :state_vector_size,
    :num_actions,
    :actor_params,
    :target_actor_params,
    :actor_net,
    :critic_params,
    :target_critic_params,
    :critic_net,
    :actor_predict_fn,
    :critic_predict_fn,
    :actor_update_fn,
    :critic_update_fn,
    :experience_replay_buffer,
    :experience_replay_buffer_index,
    :persisted_experience_replay_buffer_entries,
    :environment_to_state_vector_fn,
    :state_vector_to_input_fn,
    :gamma,
    :experience_replay_buffer_max_size,
    :tau,
    :batch_size,
    :noise_stddev,
    :training_frequency,
    :target_update_frequency,
    :actor_optimizer_state,
    :critic_optimizer_state,
    :environment_to_input_fn
  ]

  @impl true
  def init(random_key, opts \\ []) do
    opts =
      Keyword.validate!(opts, [
        :actor_params,
        :target_actor_params,
        :actor_net,
        :critic_params,
        :target_critic_params,
        :critic_net,
        :experience_replay_buffer,
        :experience_replay_buffer_index,
        :persisted_experience_replay_buffer_entries,
        :environment_to_state_vector_fn,
        :state_vector_to_input_fn,
        gamma: 0.99,
        experience_replay_buffer_max_size: 1_000_000,
        tau: 0.005,
        batch_size: 64,
        noise_stddev: 0.1,
        training_frequency: 32,
        target_update_frequency: 128,
        actor_optimizer_params: [learning_rate: 1.0e-3, eps: 1.0e-7, adamw_decay: 1.0e-2],
        critic_optimizer_params: [learning_rate: 2.0e-3, eps: 1.0e-7, adamw_decay: 1.0e-2]
      ])

    actor_net = opts[:actor_net] || raise ArgumentError, "missing :actor_net option"
    critic_net = opts[:critic_net] || raise ArgumentError, "missing :critic_net option"

    environment_to_state_vector_fn =
      opts[:environment_to_state_vector_fn] ||
        raise ArgumentError, "missing :environment_to_state_vector_fn option"

    state_vector_to_input_fn =
      opts[:state_vector_to_input_fn] ||
        raise ArgumentError, "missing :state_vector_to_input_fn option"

    {actor_init_fn, actor_predict_fn} = Axon.build(actor_net, seed: 0)
    {critic_init_fn, critic_predict_fn} = Axon.build(critic_net, seed: 0)

    critic_predict_fn = fn state_vector, action_vector ->
      state_vector
      |> state_vector_to_input_fn.()
      |> Map.put("action", action_vector)
      |> critic_predict_fn.()
    end

    {actor_optimizer_init_fn, actor_optimizer_update_fn} =
      Axon.Updates.clip_by_global_norm()
      |> Axon.Updates.compose(
        Axon.Optimizers.adamw(
          opts[:actor_optimizer_params][:learning_rate],
          eps: opts[:actor_optimizer_params][:eps],
          decay: opts[:actor_optimizer_params][:adamw_decay]
        )
      )

    {critic_optimizer_init_fn, critic_optimizer_update_fn} =
      Axon.Updates.clip_by_global_norm()
      |> Axon.Updates.compose(
        Axon.Optimizers.adamw(
          opts[:critic_optimizer_params][:learning_rate],
          eps: opts[:critic_optimizer_params][:eps],
          decay: opts[:critic_optimizer_params][:adamw_decay]
        )
      )

    initial_actor_params_state = opts[:actor_params] || raise "missing initial actor_params"
    initial_actor_target_params_state = opts[:actor_target_params] || initial_actor_params_state
    initial_critic_params_state = opts[:critic_params] || raise "missing initial critic_params"

    initial_critic_target_params_state =
      opts[:critic_target_params] || initial_critic_params_state

    input_template = input_template(actor_net)

    case input_template do
      %{"action " => _} ->
        raise ArgumentError,
              "the input template for the actor_network must not contain the reserved key \"action\""
    end

    {1, num_actions} = Axon.get_output_shape(policy_net, input_template)

    critic_template = input_template(critic_net)

    case critic_template do
      %{"action " => action_input} ->
        unless action_input != Nx.template({1, num_actions}, :f32) do
          raise ArgumentError,
                "the critic network must accept the \"action\" input with shape {nil, #{num_actions}} and type :f32"
        end

        if Map.delete(critic_template, "action") != input_template do
          raise ArgumentError,
                "the critic network must have the same input template as the actor network + the \"action\" input"
        end
    end

    actor_params = actor_init_fn.(input_template, initial_actor_params_state)
    actor_target_params = actor_init_fn.(input_template, initial_actor_target_params_state)

    actor_optimizer_state = optimizer_init_fn.(actor_params)

    critic_params = critic_init_fn.(input_template, initial_critic_params_state)
    critic_target_params = critic_init_fn.(input_template, initial_critic_target_params_state)

    critic_optimizer_state = optimizer_init_fn.(critic_params)

    state_vector_size = state_vector_size(input_template)

    loss = loss_denominator = Nx.tensor(0, type: :f32)

    state = %__MODULE__{
      state_vector_size: state_vector_size,
      num_actions: num_actions,
      actor_params: actor_params,
      target_actor_params: target_actor_params,
      actor_net: actor_net,
      critic_params: critic_params,
      target_critic_params: target_critic_params,
      critic_net: critic_net,
      actor_predict_fn: actor_predict_fn,
      critic_predict_fn: critic_predict_fn,
      actor_update_fn: actor_update_fn,
      critic_update_fn: critic_update_fn,
      experience_replay_buffer:
        opts[:experience_replay_buffer] ||
          Nx.broadcast(
            Nx.tensor(:nan, type: :f32),
            {experience_replay_buffer_max_size, 2 * state_vector_size + num_actions + 4}
          ),
      experience_replay_buffer_index:
        opts[:experience_replay_buffer_index] || Nx.tensor(0, type: :s64),
      persisted_experience_replay_buffer_entries:
        opts[:persisted_experience_replay_buffer_entries] || Nx.tensor(0, type: :s64),
      environment_to_state_vector_fn: environment_to_state_vector_fn,
      state_vector_to_input_fn: state_vector_to_input_fn,
      input_template: input_template,
      gamma: gamma,
      tau: tau,
      batch_size: batch_size,
      noise: noise,
      training_frequency: training_frequency,
      target_update_frequency: target_update_frequency,
      environment_to_input_fn: environment_to_input_fn,
      loss: loss,
      loss_denominator: loss_denominator,
      actor_optimizer_update_fn: actor_optimizer_update_fn,
      critic_optimizer_update_fn: critic_optimizer_update_fn,
      actor_optimizer_state: actor_optimizer_state,
      critic_optimizer_state: critic_optimizer_state
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

    {%{
       state
       | total_reward: total_reward,
         loss: loss,
         loss_denominator: loss_denominator
     }, random_key}
  end

  @impl true
  defn select_action(
         %ReinforcementLearning{random_key: random_key, agent_state: agent_state} = state,
         _iteration
       ) do
    %__MODULE__{
      actor_params: actor_params,
      actor_predict_fn: actor_predict_fn,
      environment_to_state_vector_fn: environment_to_state_vector_fn,
      state_vector_to_input_fn: state_vector_to_input_fn,
      num_actions: num_actions,
      noise_stddev: noise_stddev
    } = agent_state

    state_vector = environment_to_state_vector_fn.(state.environment_state)

    input = state_vector_to_input_fn.(state_vector)

    action_vector = actor_predict_fn.(actor_params, input)

    {additive_noise, random_key} =
      if noise_stddev <= 0 do
        # pass 0 or negative to turn off randomness
        {0, random_key}
      else
        Nx.Random.normal(random_key, 0, noise_stddev, shape: {num_actions}, type: :f32)
      end

    action_vector = action_vector + additive_noise

    {action_vector, %{state | state_vector: state_vector, random_key: random_key}}
  end

  @impl true
  defn record_observation(
         %{
           environment_state: env_state,
           agent_state: %__MODULE__{
             actor_params: actor_params,
             critic_params: critic_params,
             critic_target_params: critic_target_params,
             critic_predict_fn: critic_predict_fn,
             state_vector: state_vector,
             state_vector_to_input_fn: state_vector_to_input_fn,
             environment_to_state_vector_fn: as_state_vector_fn,
             gamma: gamma
           }
         },
         action_vector,
         reward,
         is_terminal,
         %{environment_state: next_env_state} = state
       ) do
    next_state_vector = as_state_vector_fn.(next_env_state)

    idx = Nx.stack([state.agent_state.experience_replay_buffer_index, 0]) |> Nx.new_axis(0)

    shape = {2 * Nx.size(state_vector) + Nx.size(action_vector) + 4, 1}

    index_template = Nx.concatenate([Nx.broadcast(0, shape), Nx.iota(shape, axis: 0)], axis: 1)

    target_action_vector = actor_predict_fn.(actor_target_params, next_state_vector)

    target_prediction =
      critic_predict_fn.(critic_target_params, next_state_vector, target_action_vector)

    temporal_difference =
      reward + gamma * target_prediction * (1 - is_terminal) -
        critic_predict_fn.(critic_params, state_vector, action_vector)

    temporal_difference = temporal_difference |> Nx.abs() |> Nx.reshape({1})

    updates =
      Nx.concatenate([
        Nx.flatten(state_vector),
        action_vector,
        Nx.new_axis(reward, 0),
        Nx.new_axis(is_terminal, 0),
        Nx.flatten(next_state_vector),
        temporal_difference
      ])

    experience_replay_buffer =
      Nx.indexed_put(state.agent_state.experience_replay_buffer, idx + index_template, updates)

    experience_replay_buffer_index =
      Nx.remainder(
        state.agent_state.experience_replay_buffer_index + 1,
        state.agent_state.experience_replay_buffer_max_size
      )

    entries = state.agent_state.persisted_experience_replay_buffer_entries

    persisted_experience_replay_buffer_entries =
      Nx.select(
        entries < state.agent_state.experience_replay_buffer_max_size,
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
        train(state)

      should_update_policy_net and should_update_target_net ->
        state
        |> train()
        |> soft_update_targets()

      true ->
        state
    end
  end

  defnp train(state) do
    %{
      agent_state: %{
        actor_params: actor_params,
        actor_predict_fn: actor_predict_fn,
        critic_params: critic_params,
        critic_target_params: critic_target_params,
        critic_predict_fn: critic_predict_fn,
        actor_optimizer_state: actor_optimizer_state,
        critic_optimizer_state: critic_optimizer_state,
        actor_optimizer_update_fn: actor_optimizer_update_fn,
        critic_optimizer_update_fn: critic_optimizer_update_fn,
        state_vector_to_input_fn: state_vector_to_input_fn,
        state_vector_size: state_vector_size,
        experience_replay_buffer: experience_replay_buffer,
        num_actions: num_actions,
        gamma: gamma
      },
      random_key: random_key
    } = state

    {batch, batch_idx, random_key} =
      sample_experience_replay_buffer(random_key, state.agent_state)

    state_batch = Nx.slice_along_axis(batch, 0, state_vector_size, axis: 1)

    action_batch = Nx.slice_along_axis(batch, state_vector_size, num_actions, axis: 1)
    reward_batch = Nx.slice_along_axis(batch, state_vector_size + num_actions, 1, axis: 1)

    is_terminal_batch =
      Nx.slice_along_axis(batch, state_vector_size + num_actions + 1, 1, axis: 1)

    next_state_batch =
      Nx.slice_along_axis(
        batch,
        state_vector_size + num_actions + 2,
        state_vector_size,
        state_vector_size,
        axis: 1
      )

    non_final_mask = not is_terminal_batch

    # train critic network
    {{experience_replay_buffer, critic_loss}, critic_gradient} =
      value_and_grad(
        critic_params,
        fn critic_params ->
          target_actions = actor_predict_fn.(actor_target_params, state_batch)

          target_critic_prediction =
            critic_predict_fn.(critic_target_params, state_batch, target_action)

          %{shape: {n, 1}} =
            critic_predition = critic_predict_fn.(critic_params, state_batch, action_batch)

          %{shape: {m, 1}} =
            target = reward_batch + gamma * non_final_mask * target_critic_prediction

          case {m, n} do
            {m, n} when m != n ->
              raise "shape mismatch for batch values"

            _ ->
              1
          end

          td_errors = Nx.abs(target - critic_prediction)

          {
            update_priorities(
              experience_replay_buffer,
              batch_idx,
              state_vector_size * 2 + num_actions + 2,
              td_errors
            ),
            huber_loss(target, critic_prediction)
          }
        end,
        &elem(&1, 1)
      )

    {critic_updates, optimizer_state} =
      critic_optimizer_update_fn.(critic_gradient, critic_optimizer_state, critic_params)

    critic_params = Axon.Updates.apply_updates(critic_params, critic_updates)

    # train actor network
    {actor_loss, actor_gradient} =
      value_and_grad(
        actor_params,
        fn actor_params ->
          actions = actor_predict_fn.(actor_params, state_batch)
          # the training comes from us using the new critic_params to predict new values
          critic_prediction = critic_predict_fn.(critic_params, state_batch, actions)
          # negate because we want to perform gradient ascent using a gradient descent optimizer
          Nx.mean(-critic_prediction)
        end
      )

    {actor_updates, optimizer_state} =
      actor_optimizer_update_fn.(actor_gradient, actor_optimizer_state, actor_params)

    actor_params = Axon.Updates.apply_updates(actor_params, actor_updates)

    %{
      state
      | agent_state: %{
          state.agent_state
          | actor_params: actor_params,
            actor_optimizer_state: actor_optimizer_state,
            critic_params: critic_params,
            critic_optimizer_state: critic_optimizer_state,
            loss: state.agent_state.loss + critic_loss,
            loss_denominator: state.agent_state.loss_denominator + 1,
            experience_replay_buffer: experience_replay_buffer
        },
        random_key: random_key
    }
  end

  defnp soft_update_targets(state) do
    %{
      agent_state:
        %{
          actor_target_params: actor_target_params,
          actor_params: actor_params,
          critic_target_params: critic_target_params,
          critic_params: critic_params
        } = agent_state
    } = state

    actor_target_params =
      Axon.Shared.deep_merge(actor_params, actor_target_params, &(&1 * @tau + &2 * (1 - @tau)))

    critic_target_params =
      Axon.Shared.deep_merge(critic_params, critic_target_params, &(&1 * @tau + &2 * (1 - @tau)))

    %{
      state
      | agent_state: %{
          agent_state
          | actor_target_params: actor_target_params,
            critic_target_params: critic_target_params
        }
    }
  end

  @alpha 0.6
  defnp sample_experience_replay_buffer(
          random_key,
          %{state_vector_size: state_vector_size, num_actions: num_actions} = agent_state
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
      |> Nx.slice_along_axis(state_vector_size * 2 + num_actions + 2, 1, axis: 1)
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
      persisted_experience_replay_buffer_entries: entries,
      experience_replay_buffer_max_size: experience_replay_buffer_max_size
    } = state

    if entries < experience_replay_buffer_max_size do
      t = Nx.iota({experience_replay_buffer_max_size})
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