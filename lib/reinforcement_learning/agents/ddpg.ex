defmodule ReinforcementLearning.Agents.DDPG do
  @moduledoc """
  Deep Deterministic Policy Gradient implementation.

  This assumes that the Actor network will output `{nil, num_actions}` actions,
  and that the Critic network accepts `"actions"` input with the same shape.

  Actions are deemed to be in a continuous space of type `:f32`.
  """
  import Nx.Defn

  alias ReinforcementLearning.Utils.Noise.OUProcess
  alias ReinforcementLearning.Utils.CircularBuffer

  @behaviour ReinforcementLearning.Agent

  @derive {Inspect,
           except: [
             :actor_params,
             :actor_target_params,
             :critic_params,
             :critic_target_params,
             :experience_replay_buffer,
             :actor_optimizer_state,
             :critic_optimizer_state
           ]}
  @derive {Nx.Container,
           containers: [
             :actor_params,
             :actor_target_params,
             :critic_params,
             :critic_target_params,
             :experience_replay_buffer,
             :target_update_frequency,
             :loss,
             :loss_denominator,
             :total_reward,
             :actor_optimizer_state,
             :critic_optimizer_state,
             :action_lower_limit,
             :action_upper_limit,
             :state_vector,
             :ou_process,
             :max_sigma,
             :min_sigma,
             :exploration_decay_rate,
             :exploration_increase_rate,
             :performance_memory,
             :performance_threshold,
             :gamma,
             :tau,
             :exploration_warmup_episodes
           ],
           keep: [
             :environment_to_state_vector_fn,
             :actor_predict_fn,
             :critic_predict_fn,
             :state_vector_size,
             :num_actions,
             :actor_optimizer_update_fn,
             :critic_optimizer_update_fn,
             :batch_size,
             :training_frequency
           ]}
  defstruct [
    :state_vector,
    :state_vector_size,
    :num_actions,
    :actor_params,
    :actor_target_params,
    :actor_net,
    :critic_params,
    :critic_target_params,
    :critic_net,
    :actor_predict_fn,
    :critic_predict_fn,
    :experience_replay_buffer,
    :environment_to_state_vector_fn,
    :gamma,
    :tau,
    :batch_size,
    :training_frequency,
    :target_update_frequency,
    :actor_optimizer_state,
    :critic_optimizer_state,
    :action_lower_limit,
    :action_upper_limit,
    :loss,
    :loss_denominator,
    :total_reward,
    :actor_optimizer_update_fn,
    :critic_optimizer_update_fn,
    :ou_process,
    :max_sigma,
    :min_sigma,
    :exploration_decay_rate,
    :exploration_increase_rate,
    :performance_memory,
    :performance_threshold,
    :exploration_warmup_episodes
  ]

  @impl true
  def init(random_key, opts \\ []) do
    expected_opts = [
      :actor_params,
      :actor_target_params,
      :actor_net,
      :critic_params,
      :critic_target_params,
      :critic_net,
      :experience_replay_buffer,
      :environment_to_state_vector_fn,
      :performance_memory,
      :state_vector_to_input_fn,
      ou_process_opts: [],
      performance_memory_length: 500,
      exploration_decay_rate: 0.9995,
      exploration_increase_rate: 1.1,
      performance_threshold: 0.01,
      exploration_warmup_episodes: 750,
      gamma: 0.99,
      experience_replay_buffer_max_size: 100_000,
      tau: 0.005,
      batch_size: 64,
      training_frequency: 32,
      target_update_frequency: 100,
      actor_optimizer_params: [learning_rate: 1.0e-3, eps: 1.0e-7],
      critic_optimizer_params: [learning_rate: 2.0e-3, eps: 1.0e-7, adamw_decay: 1.0e-2],
      action_lower_limit: -1.0,
      action_upper_limit: 1.0
    ]

    opts = Keyword.validate!(opts, expected_opts)

    # TO-DO: use NimbleOptions
    expected_opts
    |> Enum.filter(fn x -> is_atom(x) or (is_tuple(x) and is_nil(elem(x, 1))) end)
    |> Enum.reduce(opts, fn
      k, opts ->
        case List.keytake(opts, k, 0) do
          {{^k, _}, opts} -> opts
          nil -> raise ArgumentError, "missing option #{k}"
        end
    end)
    |> Enum.each(fn {k, v} ->
      if is_nil(v) do
        raise ArgumentError, "option #{k} cannot be nil"
      end
    end)

    actor_optimizer_params = opts[:actor_optimizer_params]
    critic_optimizer_params = opts[:critic_optimizer_params]

    optimizer_keys = [:learning_rate, :eps]

    for {name, opts} <- [actor: actor_optimizer_params, critic: critic_optimizer_params],
        k <- optimizer_keys do
      v = opts[k]

      unless is_number(v) or is_function(v) do
        raise ArgumentError,
              "expected [:#{name}_optimizer_params][#{k}] option to be a number, got: #{inspect(v)}"
      end
    end

    actor_net = opts[:actor_net]
    critic_net = opts[:critic_net]

    environment_to_state_vector_fn = opts[:environment_to_state_vector_fn]
    state_vector_to_input_fn = opts[:state_vector_to_input_fn]

    {actor_init_fn, actor_predict_fn} = Axon.build(actor_net, seed: 0)
    {critic_init_fn, critic_predict_fn} = Axon.build(critic_net, seed: 1)

    actor_predict_fn = fn params, state_vector ->
      actor_predict_fn.(params, state_vector_to_input_fn.(state_vector))
    end

    critic_predict_fn = fn params, state_vector, action_vector ->
      input =
        state_vector
        |> state_vector_to_input_fn.()
        |> Map.put("actions", action_vector)

      critic_predict_fn.(params, input)
    end

    {actor_optimizer_init_fn, actor_optimizer_update_fn} =
      Axon.Updates.clip(delta: 10)
      |> Axon.Updates.compose(
        Axon.Optimizers.adam(
          actor_optimizer_params[:learning_rate],
          eps: actor_optimizer_params[:eps]
        )
      )
      |> Axon.Updates.clip(delta: 1)

    {critic_optimizer_init_fn, critic_optimizer_update_fn} =
      Axon.Updates.compose(
        Axon.Optimizers.adam(
          critic_optimizer_params[:learning_rate],
          eps: critic_optimizer_params[:eps]
          # decay: critic_optimizer_params[:adamw_decay]
        ),
        Axon.Updates.clip()
      )

    initial_actor_params_state = opts[:actor_params]
    initial_actor_target_params_state = opts[:actor_target_params] || initial_actor_params_state
    initial_critic_params_state = opts[:critic_params]

    initial_critic_target_params_state =
      opts[:critic_target_params] || initial_critic_params_state

    input_template = input_template(actor_net)

    case input_template do
      %{"actions" => _} ->
        raise ArgumentError,
              "the input template for the actor_network must not contain the reserved key \"actions\""

      _ ->
        :ok
    end

    {1, num_actions} = Axon.get_output_shape(actor_net, input_template)

    {max_sigma, ou_process_opts} = Keyword.pop!(opts[:ou_process_opts], :max_sigma)
    {min_sigma, ou_process_opts} = Keyword.pop!(ou_process_opts, :min_sigma)

    unless max_sigma do
      raise ArgumentError, "option [:ou_process_opts][:max_sigma] cannot be nil"
    end

    unless min_sigma do
      raise ArgumentError, "option [:ou_process_opts][:min_sigma] cannot be nil"
    end

    ou_process = OUProcess.init({1, num_actions}, ou_process_opts)

    critic_template = input_template(critic_net)

    case critic_template do
      %{"actions" => action_input} ->
        unless action_input != Nx.template({nil, num_actions}, :f32) do
          raise ArgumentError,
                "the critic network must accept the \"actions\" input with shape {nil, #{num_actions}} and type :f32, got input template: #{critic_template}"
        end

        if Map.delete(critic_template, "actions") != input_template do
          raise ArgumentError,
                "the critic network must have the same input template as the actor network + the \"action\" input"
        end

      _ ->
        :ok
    end

    actor_params = actor_init_fn.(input_template, initial_actor_params_state)
    actor_target_params = actor_init_fn.(input_template, initial_actor_target_params_state)

    actor_optimizer_state = actor_optimizer_init_fn.(actor_params)

    critic_params = critic_init_fn.(critic_template, initial_critic_params_state)
    critic_target_params = critic_init_fn.(critic_template, initial_critic_target_params_state)

    critic_optimizer_state = critic_optimizer_init_fn.(critic_params)

    state_vector_size = state_vector_size(input_template)

    total_reward = loss = loss_denominator = Nx.tensor(0, type: :f32)
    experience_replay_buffer_max_size = opts[:experience_replay_buffer_max_size]

    state = %__MODULE__{
      max_sigma: max_sigma,
      min_sigma: min_sigma,
      exploration_warmup_episodes: opts[:exploration_warmup_episodes],
      exploration_decay_rate: opts[:exploration_decay_rate],
      exploration_increase_rate: opts[:exploration_increase_rate],
      state_vector: Nx.broadcast(0.0, {1, state_vector_size}),
      state_vector_size: state_vector_size,
      num_actions: num_actions,
      actor_params: actor_params,
      actor_target_params: actor_target_params,
      actor_net: actor_net,
      critic_params: critic_params,
      critic_target_params: critic_target_params,
      critic_net: critic_net,
      actor_predict_fn: actor_predict_fn,
      critic_predict_fn: critic_predict_fn,
      performance_threshold: opts[:performance_threshold],
      performance_memory:
        opts[:performance_memory] || CircularBuffer.new({opts[:performance_memory_length]}),
      experience_replay_buffer:
        opts[:experience_replay_buffer] ||
          CircularBuffer.new(
            {experience_replay_buffer_max_size, 2 * state_vector_size + num_actions + 3}
          ),
      environment_to_state_vector_fn: environment_to_state_vector_fn,
      gamma: opts[:gamma],
      tau: opts[:tau],
      batch_size: opts[:batch_size],
      ou_process: ou_process,
      training_frequency: opts[:training_frequency],
      target_update_frequency: opts[:target_update_frequency],
      total_reward: total_reward,
      loss: loss,
      loss_denominator: loss_denominator,
      actor_optimizer_update_fn: actor_optimizer_update_fn,
      critic_optimizer_update_fn: critic_optimizer_update_fn,
      actor_optimizer_state: actor_optimizer_state,
      critic_optimizer_state: critic_optimizer_state,
      action_lower_limit: opts[:action_lower_limit],
      action_upper_limit: opts[:action_upper_limit]
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
  def reset(random_key, %ReinforcementLearning{episode: episode, agent_state: state}) do
    total_reward = loss = loss_denominator = Nx.tensor(0, type: :f32)

    state = adapt_exploration(episode, state)

    {%{
       state
       | total_reward: total_reward,
         loss: loss,
         loss_denominator: loss_denominator
     }, random_key}
  end

  defnp adapt_exploration(
          episode,
          %__MODULE__{
            exploration_warmup_episodes: exploration_warmup_episodes,
            experience_replay_buffer: experience_replay_buffer,
            ou_process: ou_process,
            exploration_decay_rate: exploration_decay_rate,
            exploration_increase_rate: exploration_increase_rate,
            min_sigma: min_sigma,
            max_sigma: max_sigma,
            total_reward: reward,
            performance_memory: performance_memory,
            performance_threshold: performance_threshold
          } = state
        ) do
    n = Nx.axis_size(performance_memory.data, 0)

    {ou_process, performance_memory} =
      cond do
        episode == 0 ->
          {ou_process, performance_memory}

        episode < n or experience_replay_buffer.size < n or
            episode < exploration_warmup_episodes ->
          {ou_process, CircularBuffer.append(performance_memory, reward)}

        true ->
          performance_memory = CircularBuffer.append(performance_memory, reward)

          # After we take and reshape, the first row contains the oldest `n//2` samples
          # and the second row, the remaining newest samples.
          windows =
            performance_memory
            |> CircularBuffer.ordered_data()
            |> Nx.reshape({2, :auto})

          # avg[0]: avg of the previous performance window
          # avg[1]: avg of the current performance window
          avg = Nx.mean(windows, axes: [1])

          abs_diff = Nx.abs(avg[0] - avg[1])

          sigma =
            if abs_diff < performance_threshold do
              # If decayed to less than an "eps" value,
              # we force it to increase from that "eps" instead.
              Nx.min(ou_process.sigma * exploration_increase_rate, max_sigma)
            else
              # can decay to 0
              Nx.max(ou_process.sigma * exploration_decay_rate, min_sigma)
            end

          {%OUProcess{ou_process | sigma: sigma}, performance_memory}
      end

    %__MODULE__{
      state
      | ou_process: OUProcess.reset(ou_process),
        performance_memory: performance_memory
    }
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
      action_lower_limit: action_lower_limit,
      action_upper_limit: action_upper_limit,
      ou_process: ou_process
    } = agent_state

    state_vector = environment_to_state_vector_fn.(state.environment_state)

    action_vector = actor_predict_fn.(actor_params, state_vector)

    {%OUProcess{x: additive_noise} = ou_process, random_key} =
      OUProcess.sample(random_key, ou_process)

    action_vector = action_vector + additive_noise

    clipped_action_vector =
      action_vector
      |> Nx.max(action_lower_limit)
      |> Nx.min(action_upper_limit)

    {clipped_action_vector,
     %{
       state
       | agent_state: %{agent_state | ou_process: ou_process, state_vector: state_vector},
         random_key: random_key
     }}
  end

  @impl true
  defn record_observation(
         %{
           agent_state: %__MODULE__{
             actor_target_params: actor_target_params,
             actor_predict_fn: actor_predict_fn,
             critic_params: critic_params,
             critic_target_params: critic_target_params,
             critic_predict_fn: critic_predict_fn,
             state_vector: state_vector,
             environment_to_state_vector_fn: as_state_vector_fn,
             experience_replay_buffer: experience_replay_buffer,
             gamma: gamma
           }
         },
         action_vector,
         reward,
         is_terminal,
         %{environment_state: next_env_state} = state
       ) do
    next_state_vector = as_state_vector_fn.(next_env_state)

    target_action_vector = actor_predict_fn.(actor_target_params, next_state_vector)

    target_prediction =
      critic_predict_fn.(critic_target_params, next_state_vector, target_action_vector)

    temporal_difference =
      reward + gamma * target_prediction * (1 - is_terminal) -
        critic_predict_fn.(critic_params, state_vector, action_vector)

    temporal_difference = Nx.abs(temporal_difference)

    updates =
      Nx.concatenate([
        Nx.flatten(state_vector),
        Nx.flatten(action_vector),
        Nx.new_axis(reward, 0),
        Nx.new_axis(is_terminal, 0),
        Nx.flatten(next_state_vector),
        Nx.reshape(temporal_difference, {1})
      ])

    experience_replay_buffer =
      CircularBuffer.append(experience_replay_buffer, updates)

    %{
      state
      | agent_state: %{
          state.agent_state
          | experience_replay_buffer: experience_replay_buffer,
            total_reward: state.agent_state.total_reward + reward
        }
    }
  end

  @impl true
  defn optimize_model(state) do
    %{
      experience_replay_buffer: experience_replay_buffer,
      batch_size: batch_size,
      training_frequency: training_frequency,
      exploration_warmup_episodes: exploration_warmup_episodes
    } = state.agent_state

    warming_up = state.episode < exploration_warmup_episodes
    has_at_least_one_batch = experience_replay_buffer.size > batch_size

    should_train =
      not warming_up and has_at_least_one_batch and
        rem(experience_replay_buffer.index, training_frequency) == 0

    if should_train do
      train_loop(state, training_frequency)
    else
      state
    end
  end

  defnp train_loop(state, training_frequency) do
    while state, _ <- 0..(training_frequency - 1)//1, unroll: true do
      {batch, batch_indices, random_key} =
        sample_experience_replay_buffer(state.random_key, state.agent_state)

      %{state | random_key: random_key}
      |> train(batch, batch_indices)
      |> soft_update_targets()
    end
  end

  defnp train(state, batch, batch_idx) do
    %{
      agent_state: %{
        actor_params: actor_params,
        actor_target_params: actor_target_params,
        actor_predict_fn: actor_predict_fn,
        critic_params: critic_params,
        critic_target_params: critic_target_params,
        critic_predict_fn: critic_predict_fn,
        actor_optimizer_state: actor_optimizer_state,
        critic_optimizer_state: critic_optimizer_state,
        actor_optimizer_update_fn: actor_optimizer_update_fn,
        critic_optimizer_update_fn: critic_optimizer_update_fn,
        state_vector_size: state_vector_size,
        experience_replay_buffer: experience_replay_buffer,
        num_actions: num_actions,
        gamma: gamma
      }
    } = state

    state_batch = Nx.slice_along_axis(batch, 0, state_vector_size, axis: 1)

    action_batch = Nx.slice_along_axis(batch, state_vector_size, num_actions, axis: 1)
    reward_batch = Nx.slice_along_axis(batch, state_vector_size + num_actions, 1, axis: 1)

    is_terminal_batch =
      Nx.slice_along_axis(batch, state_vector_size + num_actions + 1, 1, axis: 1)

    next_state_batch =
      Nx.slice_along_axis(batch, state_vector_size + num_actions + 2, state_vector_size, axis: 1)

    non_final_mask = not is_terminal_batch

    ### Train Critic

    {{experience_replay_buffer, critic_loss}, critic_gradient} =
      value_and_grad(
        critic_params,
        fn critic_params ->
          target_actions = actor_predict_fn.(actor_target_params, next_state_batch)

          q_target = critic_predict_fn.(critic_target_params, next_state_batch, target_actions)

          %{shape: {n, 1}} = q = critic_predict_fn.(critic_params, state_batch, action_batch)

          %{shape: {m, 1}} = backup = reward_batch + gamma * non_final_mask * q_target

          case {m, n} do
            {m, n} when m != n ->
              raise "shape mismatch for batch values"

            _ ->
              1
          end

          td_errors = Nx.abs(backup - q)

          {
            update_priorities(
              experience_replay_buffer,
              batch_idx,
              td_errors
            ),
            Nx.mean(td_errors ** 2)
          }
        end,
        &elem(&1, 1)
      )

    {critic_updates, critic_optimizer_state} =
      critic_optimizer_update_fn.(critic_gradient, critic_optimizer_state, critic_params)

    critic_params = Axon.Updates.apply_updates(critic_params, critic_updates)

    ### Train Actor

    actor_gradient =
      grad(actor_params, fn actor_params ->
        actions = actor_predict_fn.(actor_params, state_batch)
        q = critic_predict_fn.(critic_params, state_batch, actions)
        -Nx.mean(q)
      end)

    {actor_updates, actor_optimizer_state} =
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
        }
    }
  end

  defnp soft_update_targets(state) do
    %{
      agent_state:
        %{
          actor_target_params: actor_target_params,
          actor_params: actor_params,
          critic_target_params: critic_target_params,
          critic_params: critic_params,
          tau: tau
        } = agent_state
    } = state

    actor_target_params =
      Axon.Shared.deep_merge(
        actor_params,
        actor_target_params,
        &Nx.as_type(&1 * tau + &2 * (1 - tau), Nx.type(&1))
      )

    critic_target_params =
      Axon.Shared.deep_merge(
        critic_params,
        critic_target_params,
        &Nx.as_type(&1 * tau + &2 * (1 - tau), Nx.type(&1))
      )

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
          %{
            batch_size: batch_size,
            state_vector_size: state_vector_size,
            num_actions: num_actions
          } = agent_state
        ) do
    data = agent_state.experience_replay_buffer.data

    temporal_difference =
      data
      |> Nx.slice_along_axis(state_vector_size * 2 + num_actions + 2, 1, axis: 1)
      |> Nx.flatten()

    priorities = temporal_difference ** @alpha
    probs = priorities / Nx.sum(priorities)

    {batch_idx, random_key} =
      Nx.Random.choice(random_key, Nx.iota(temporal_difference.shape), probs,
        samples: batch_size,
        replace: false,
        axis: 0
      )

    batch = Nx.take(data, batch_idx)

    {batch, batch_idx, random_key}
  end

  defn update_priorities(%{data: %{shape: {_, item_size}}} = buffer, %{shape: {n}} = entry_indices, td_errors) do
    case td_errors.shape do
      {^n, 1} -> :ok
      shape -> raise "invalid shape for td_errors, got: #{inspect(shape)}"
    end

    indices =
      Nx.stack([entry_indices, Nx.broadcast(item_size - 1, {n})], axis: -1)

    %{buffer | data: Nx.indexed_put(buffer.data, indices, Nx.reshape(td_errors, {n}))}
  end
end
