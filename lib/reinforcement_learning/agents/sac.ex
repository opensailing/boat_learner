defmodule ReinforcementLearning.Agents.SAC do
  @moduledoc """
  Soft Actor-Critic implementation.

  This assumes that the Actor network will output `{nil, num_actions, 2}`,
  where for each action they output the $\\mu$ and $\\sigma$ values of a random
  normal distribution, and that the Critic network accepts `"actions"` input with
  shape `{nil, num_actions}`, where the action is calculated by sampling from
  said random distribution.

  Actions are deemed to be in a continuous space of type `:f32`.

  For simplicity in the implementation of the Dual Q implementation,
  `:critic_params` is vectorized with the axis `:critics` with default
  size 2. Likewise, `:critic_target_params` and `:critic_optimizer_state`
  are also vectorized in the same way.

  Vectorized axes from `:random_key` are still propagated normally throughout
  the agent state for parallel training.
  """
  import Nx.Defn

  import Nx.Constants, only: [pi: 1]

  alias ReinforcementLearning.Utils.Noise.OUProcess
  alias ReinforcementLearning.Utils.CircularBuffer

  @behaviour ReinforcementLearning.Agent

  @derive {Nx.Container,
           containers: [
             :actor_params,
             :critic_params,
             :critic_target_params,
             :experience_replay_buffer,
             :loss,
             :loss_denominator,
             :total_reward,
             :actor_optimizer_state,
             :critic_optimizer_state,
             :action_lower_limit,
             :action_upper_limit,
             :ou_process,
             :max_sigma,
             :min_sigma,
             :exploration_decay_rate,
             :exploration_increase_rate,
             :performance_memory,
             :performance_threshold,
             :gamma,
             :tau,
             :state_features_memory,
             :entropy_coefficient
           ],
           keep: [
             :exploration_fn,
             :environment_to_state_features_fn,
             :actor_predict_fn,
             :critic_predict_fn,
             :num_actions,
             :actor_optimizer_update_fn,
             :critic_optimizer_update_fn,
             :batch_size,
             :training_frequency,
             :input_entry_size
           ]}

  defstruct [
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
    :environment_to_state_features_fn,
    :gamma,
    :tau,
    :batch_size,
    :training_frequency,
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
    :exploration_fn,
    :state_features_memory,
    :input_entry_size,
    :entropy_coefficient
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
      :environment_to_state_features_fn,
      :performance_memory,
      :state_features_memory_to_input_fn,
      :state_features_memory,
      :state_features_size,
      :actor_optimizer,
      :critic_optimizer,
      ou_process_opts: [max_sigma: 0.2, min_sigma: 0.001, sigma: 0.01],
      performance_memory_length: 500,
      state_features_memory_length: 1,
      exploration_decay_rate: 0.9995,
      exploration_increase_rate: 1.1,
      performance_threshold: 0.01,
      exploration_fn: &Nx.less(&1, 500),
      gamma: 0.99,
      experience_replay_buffer_max_size: 100_000,
      tau: 0.005,
      batch_size: 64,
      training_frequency: 32,
      action_lower_limit: -1.0,
      action_upper_limit: 1.0,
      entropy_coefficient: 0.2
    ]

    opts = Keyword.validate!(opts, expected_opts)

    # TO-DO: use NimbleOptions
    expected_opts
    |> Enum.filter(fn x -> is_atom(x) or (is_tuple(x) and is_nil(elem(x, 1))) end)
    |> Enum.reject(fn k ->
      k in [:state_features_memory, :performance_memory, :experience_replay_buffer]
    end)
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

    {actor_optimizer_init_fn, actor_optimizer_update_fn} = opts[:actor_optimizer]
    {critic_optimizer_init_fn, critic_optimizer_update_fn} = opts[:critic_optimizer]

    actor_net = opts[:actor_net]
    critic_net = opts[:critic_net]

    environment_to_state_features_fn = opts[:environment_to_state_features_fn]
    state_features_memory_to_input_fn = opts[:state_features_memory_to_input_fn]

    {actor_init_fn, actor_predict_fn} = Axon.build(actor_net, seed: 0)
    {critic_init_fn, critic_predict_fn} = Axon.build(critic_net, seed: 1)

    actor_predict_fn = fn random_key, params, state_features_memory ->
      action_distribution_vector =
        actor_predict_fn.(params, state_features_memory_to_input_fn.(state_features_memory))

      mu = action_distribution_vector[[.., .., 0]]
      logstddev = action_distribution_vector[[.., .., 1]]

      log_stddev = Nx.clip(logstddev, -20, 2)

      stddev = Nx.exp(log_stddev)

      eps_shape = Nx.shape(stddev)

      # Nx.Random.normal is treated as a constant, so we obtain `eps` from a mean-0 stddev-1
      # normal distribution and scale it by our stddev below to obtain our sample in a way that
      # the grads a propagated through properly.
      {eps, random_key} = Nx.Random.normal(random_key, shape: eps_shape)

      {mu + stddev * eps, mu, stddev, random_key}
    end

    critic_predict_fn = fn params, state_features_memory, action_vector ->
      input =
        state_features_memory
        |> state_features_memory_to_input_fn.()
        |> Map.put("actions", action_vector)

      critic_predict_fn.(params, input)
    end

    initial_actor_params_state = opts[:actor_params]
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

    critic_template = input_template(critic_net, critics: 2)

    case critic_template do
      %{"actions" => action_input} ->
        unless Nx.devectorize(action_input)[0] != Nx.template({nil, num_actions}, :f32) do
          raise ArgumentError,
                "the critic network must accept the \"actions\" input with shape {nil, #{num_actions}} and type :f32, got input template: #{critic_template}"
        end

        critic_template_devec =
          critic_template
          |> Map.delete("actions")
          |> Map.new(fn {k, v} -> {k, Nx.devectorize(v)[0]} end)

        if critic_template_devec != input_template do
          raise ArgumentError,
                "the critic network must have the same input template as the actor network + the \"action\" input"
        end

      _ ->
        :ok
    end

    actor_params = actor_init_fn.(input_template, initial_actor_params_state)
    actor_optimizer_state = actor_optimizer_init_fn.(actor_params)

    critic_params = critic_init_fn.(critic_template, initial_critic_params_state)
    critic_target_params = critic_init_fn.(critic_template, initial_critic_target_params_state)

    critic_optimizer_state = critic_optimizer_init_fn.(critic_params)

    state_features_size = opts[:state_features_size]

    total_reward = loss = loss_denominator = Nx.tensor(0, type: :f32)
    experience_replay_buffer_max_size = opts[:experience_replay_buffer_max_size]
    state_features_memory_length = opts[:state_features_memory_length]
    input_entry_size = state_features_size * state_features_memory_length

    {exp_replay_buffer, random_key} =
      if buffer = opts[:experience_replay_buffer] do
        {buffer, random_key}
      else
        {random_data_1, random_key} =
          Nx.Random.normal(random_key, 0, 10,
            shape: {experience_replay_buffer_max_size, input_entry_size + num_actions}
          )

        init_reward = Nx.broadcast(-1.0e-8, {experience_replay_buffer_max_size, 1})

        {random_data_2, random_key} =
          Nx.Random.normal(random_key, 0, 10,
            shape: {experience_replay_buffer_max_size, state_features_size + 1}
          )

        data =
          [random_data_1, init_reward, random_data_2]
          |> Nx.concatenate(axis: 1)
          |> then(&Nx.revectorize(&1, [], target_shape: Tuple.insert_at(&1.shape, 0, :auto)))

        buffer = %CircularBuffer{
          data: data[[0, .., ..]],
          index: 0,
          size: 0
        }

        {buffer, random_key}
      end

    state = %__MODULE__{
      max_sigma: max_sigma,
      min_sigma: min_sigma,
      input_entry_size: input_entry_size,
      exploration_fn: opts[:exploration_fn],
      exploration_decay_rate: opts[:exploration_decay_rate],
      exploration_increase_rate: opts[:exploration_increase_rate],
      state_features_memory:
        opts[:state_features_memory] ||
          CircularBuffer.new({state_features_memory_length, state_features_size}),
      num_actions: num_actions,
      actor_params: actor_params,
      actor_net: actor_net,
      critic_params: critic_params,
      critic_target_params: critic_target_params,
      critic_net: critic_net,
      actor_predict_fn: actor_predict_fn,
      critic_predict_fn: critic_predict_fn,
      performance_threshold: opts[:performance_threshold],
      performance_memory:
        opts[:performance_memory] || CircularBuffer.new({opts[:performance_memory_length]}),
      experience_replay_buffer: exp_replay_buffer,
      environment_to_state_features_fn: environment_to_state_features_fn,
      gamma: opts[:gamma],
      tau: opts[:tau],
      batch_size: opts[:batch_size],
      ou_process: ou_process,
      training_frequency: opts[:training_frequency],
      total_reward: total_reward,
      loss: loss,
      loss_denominator: loss_denominator,
      actor_optimizer_update_fn: actor_optimizer_update_fn,
      critic_optimizer_update_fn: critic_optimizer_update_fn,
      actor_optimizer_state: actor_optimizer_state,
      critic_optimizer_state: critic_optimizer_state,
      action_lower_limit: opts[:action_lower_limit],
      action_upper_limit: opts[:action_upper_limit],
      entropy_coefficient: opts[:entropy_coefficient]
    }

    case random_key.vectorized_axes do
      [] ->
        {state, random_key}

      _ ->
        vectorizable_paths = [
          [Access.key(:ou_process), Access.key(:theta)],
          [Access.key(:ou_process), Access.key(:sigma)],
          [Access.key(:ou_process), Access.key(:mu)],
          [Access.key(:ou_process), Access.key(:x)],
          [Access.key(:loss)],
          [Access.key(:loss_denominator)],
          [Access.key(:total_reward)],
          [Access.key(:performance_memory), Access.key(:data)],
          [Access.key(:performance_memory), Access.key(:index)],
          [Access.key(:performance_memory), Access.key(:size)],
          [Access.key(:performance_threshold)],
          [Access.key(:state_features_memory), Access.key(:data)],
          [Access.key(:state_features_memory), Access.key(:index)],
          [Access.key(:state_features_memory), Access.key(:size)]
        ]

        vectorized_state =
          Enum.reduce(vectorizable_paths, state, fn path, state ->
            update_in(state, path, fn value ->
              [value, _] = Nx.broadcast_vectors([value, random_key], align_ranks: false)
              value
            end)
          end)

        {vectorized_state, random_key}
    end
  end

  defp input_template(model, vectorized_axes \\ []) do
    {vec_names, vec_sizes} = Enum.unzip(vectorized_axes)

    model
    |> Axon.get_inputs()
    |> Map.new(fn {name, shape} ->
      [nil | shape] = Tuple.to_list(shape)
      shape = List.to_tuple(vec_sizes ++ [1 | shape])
      {name, Nx.vectorize(Nx.template(shape, :f32), vec_names)}
    end)
  end

  @impl true
  def reset(random_key, %ReinforcementLearning{
        episode: episode,
        environment_state: env,
        agent_state: state
      }) do
    [zero, _] = Nx.broadcast_vectors([Nx.tensor(0, type: :f32), random_key], align_ranks: false)
    total_reward = loss = loss_denominator = zero

    state = adapt_exploration(episode, state)

    init_state_features = state.environment_to_state_features_fn.(env)

    {n, _} = state.state_features_memory.data.shape

    zero = Nx.as_type(zero, :s64)

    state_features_memory = %{
      state.state_features_memory
      | data: Nx.tile(init_state_features, [n, 1]),
        index: zero,
        size: Nx.add(n, zero)
    }

    {%{
       state
       | total_reward: total_reward,
         loss: loss,
         loss_denominator: loss_denominator,
         state_features_memory: state_features_memory
     }, random_key}
  end

  defnp adapt_exploration(
          episode,
          %__MODULE__{
            # exploration_fn: exploration_fn,
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

        episode < n or experience_replay_buffer.size < n ->
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

    ou_process = %{ou_process | x: Nx.squeeze(ou_process.x)}

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
      environment_to_state_features_fn: environment_to_state_features_fn,
      state_features_memory: state_features_memory,
      action_lower_limit: action_lower_limit,
      action_upper_limit: action_upper_limit,
      ou_process: ou_process
    } = agent_state

    state_features = environment_to_state_features_fn.(state.environment_state)

    state_features_memory = CircularBuffer.append(state_features_memory, state_features)

    {action_vector, _mu, _stddev, random_key} =
      actor_predict_fn.(
        random_key,
        actor_params,
        CircularBuffer.ordered_data(state_features_memory)
      )

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
       | agent_state: %{
           agent_state
           | state_features_memory: state_features_memory,
             ou_process: ou_process
         },
         random_key: random_key
     }}
  end

  @impl true
  defn record_observation(
         %{
           agent_state: %__MODULE__{
             state_features_memory: state_features_memory,
             environment_to_state_features_fn: environment_to_state_features_fn,
             experience_replay_buffer: experience_replay_buffer
           }
         },
         action_vector,
         reward,
         is_terminal,
         %{environment_state: next_env_state} = state
       ) do
    next_state_features = environment_to_state_features_fn.(next_env_state)
    state_data = CircularBuffer.ordered_data(state_features_memory)

    updates =
      Nx.concatenate([
        Nx.flatten(state_data),
        Nx.flatten(action_vector),
        Nx.new_axis(reward, 0),
        Nx.new_axis(is_terminal, 0),
        Nx.flatten(next_state_features)
      ])

    updates =
      Nx.revectorize(updates, [],
        target_shape: {:auto, Nx.axis_size(experience_replay_buffer.data, -1)}
      )

    experience_replay_buffer = CircularBuffer.append_multiple(experience_replay_buffer, updates)

    ensure_not_vectorized!(experience_replay_buffer.data)

    %{
      state
      | agent_state: %{
          state.agent_state
          | experience_replay_buffer: experience_replay_buffer,
            total_reward: state.agent_state.total_reward + reward
        }
    }
  end

  deftransformp ensure_not_vectorized!(t) do
    case t do
      %{vectorized_axes: []} ->
        :ok

      %{vectorized_axes: _vectorized_axes} ->
        raise "found unexpected vectorized axes"
    end
  end

  @impl true
  defn optimize_model(state) do
    %{
      experience_replay_buffer: experience_replay_buffer,
      batch_size: batch_size,
      training_frequency: training_frequency,
      exploration_fn: exploration_fn
    } = state.agent_state

    exploring = state.episode |> Nx.devectorize() |> Nx.take(0) |> exploration_fn.()
    has_at_least_one_batch = experience_replay_buffer.size > batch_size

    should_train =
      has_at_least_one_batch and rem(experience_replay_buffer.index, training_frequency) == 0

    should_train = should_train |> Nx.devectorize() |> Nx.any()

    if should_train do
      train_loop(state, training_frequency, exploring)
    else
      state
    end
  end

  deftransformp train_loop(state, training_frequency, exploring) do
    if training_frequency == 1 do
      train_loop_step(state, exploring)
    else
      train_loop_while(state, training_frequency, exploring)
    end
    |> elem(0)
  end

  defnp train_loop_while(state, training_frequency, exploring) do
    while {state, exploring}, _ <- 0..(training_frequency - 1)//1, unroll: false do
      train_loop_step(state, exploring)
    end
  end

  defnp train_loop_step(state, exploring) do
    {batch, random_key} = sample_experience_replay_buffer(state.random_key, state.agent_state)

    train_actor = not exploring

    updated_state =
      %{state | random_key: random_key}
      |> train(batch, train_actor)
      |> soft_update_targets(train_actor)

    {updated_state, exploring}
  end

  defnp train(state, batch, train_actor) do
    %{
      agent_state: %__MODULE__{
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
        state_features_memory: state_features_memory,
        input_entry_size: input_entry_size,
        experience_replay_buffer: experience_replay_buffer,
        num_actions: num_actions,
        gamma: gamma,
        entropy_coefficient: entropy_coefficient
      },
      random_key: random_key
    } = state

    batch_len = Nx.axis_size(batch, 0)
    {num_states, state_features_size} = state_features_memory.data.shape

    state_batch =
      batch
      |> Nx.slice_along_axis(0, input_entry_size, axis: 1)
      |> Nx.reshape({batch_len, num_states, state_features_size})

    action_batch = Nx.slice_along_axis(batch, input_entry_size, num_actions, axis: 1)
    reward_batch = Nx.slice_along_axis(batch, input_entry_size + num_actions, 1, axis: 1)

    is_terminal_batch = Nx.slice_along_axis(batch, input_entry_size + num_actions + 1, 1, axis: 1)

    # we only persisted the new state, so we need to manipulate the `state_batch` to get the actual state
    # with state memory
    next_state_batch =
      batch
      |> Nx.slice_along_axis(input_entry_size + num_actions + 2, state_features_size, axis: 1)
      |> Nx.reshape({batch_len, 1, state_features_size})

    next_state_batch =
      if num_states == 1 do
        next_state_batch
      else
        next_state_batch =
          [
            state_batch,
            next_state_batch
          ]
          |> Nx.concatenate(axis: 1)
          |> Nx.slice_along_axis(1, num_states, axis: 1)

        expected_shape = {batch_len, num_states, state_features_size}
        actual_shape = Nx.shape(next_state_batch)

        case {actual_shape, expected_shape} do
          {x, x} ->
            :ok

          {actual_shape, expected_shape} ->
            raise "incorrect size for next_state_batch, expected #{inspect(expected_shape)}, got: #{inspect(actual_shape)}"
        end

        next_state_batch
      end

    non_final_mask = not is_terminal_batch

    ### Train Critic

    {{critic_loss, random_key}, critic_gradient} =
      value_and_grad(
        critic_params,
        fn critic_params ->
          # y_i = r_i + γ * min_{j=1,2} Q'(s_{i+1}, π(s_{i+1}|θ)|φ'_j)

          {target_actions, mus, stddevs, random_key} =
            actor_predict_fn.(random_key, actor_target_params, next_state_batch)

          # get the target Q value as the minimum over all target networks
          %{shape: {k, 1}} =
            q_target =
            critic_target_params
            |> critic_predict_fn.(next_state_batch, target_actions)
            |> Nx.devectorize()
            |> Nx.reduce_min(axes: [0])
            |> stop_grad()

          next_log_prob =
            action_log_probability(target_actions, mus, stddevs)
            |> Nx.devectorize()
            |> Nx.sum(axes: [0])

          q_target = q_target - entropy_coefficient * next_log_prob

          %{shape: {m, 1}} = backup = reward_batch + gamma * non_final_mask * q_target

          # q values for each critic network. We're vectorized here, so the non-vectorized
          # backup will be propagated accordingly to each of the networks.
          %{shape: {n, 1}} = q = critic_predict_fn.(critic_params, state_batch, action_batch)

          case {k, m, n} do
            {k, m, n} when m != n or m != k or n != k ->
              raise "shape mismatch for batch values"

            _ ->
              1
          end

          critic_losses = Nx.mean((backup - q) ** 2)

          {critic_losses, random_key}
        end,
        &elem(&1, 0)
      )

    {critic_updates, critic_optimizer_state} =
      critic_optimizer_update_fn.(critic_gradient, critic_optimizer_state, critic_params)

    critic_params = Axon.Updates.apply_updates(critic_params, critic_updates)

    ### Train Actor

    {actor_params, actor_optimizer_state, random_key} =
      if train_actor do
        {{_actor_loss, random_key}, actor_gradient} =
          value_and_grad(actor_params, fn actor_params ->
            # TO-DO: check how to get u+sigma in here.
            # Maybe we just don't need Nx.Random for the actions?

            {actions, mus, stddevs, random_key} =
              actor_predict_fn.(random_key, actor_params, state_batch)

            q = critic_predict_fn.(critic_params, state_batch, actions)

            log_probability = action_log_probability(mus, stddevs, actions)

            q = q |> Nx.devectorize() |> Nx.reduce_min()

            {Nx.mean(entropy_coefficient * log_probability - q), random_key}
          end)

        {actor_updates, actor_optimizer_state} =
          actor_optimizer_update_fn.(actor_gradient, actor_optimizer_state, actor_params)

        actor_params = Axon.Updates.apply_updates(actor_params, actor_updates)
        {actor_params, actor_optimizer_state, random_key}
      else
        {actor_params, actor_optimizer_state, random_key}
      end

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

  defnp soft_update_targets(state, train_actor) do
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
      if train_actor do
        Axon.Shared.deep_merge(
          actor_params,
          actor_target_params,
          &Nx.as_type(&1 * tau + &2 * (1 - tau), Nx.type(&1))
        )
      else
        actor_target_params
      end

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

  defnp sample_experience_replay_buffer(
          random_key,
          %__MODULE__{
            batch_size: batch_size
          } = agent_state
        ) do
    data = agent_state.experience_replay_buffer.data

    # split and devectorize random_key because we want to keep the replay buffer
    # and its samples devectorized at all times
    split_key = Nx.Random.split(random_key)

    random_key = split_key[0]
    vec_k = split_key[1]

    k = Nx.devectorize(vec_k, keep_names: false)

    k =
      case Nx.shape(k) do
        {2} ->
          k

        {_, 2} ->
          Nx.take(k, 0)
      end

    {batch, _} = Nx.Random.choice(k, data, samples: batch_size, replace: false, axis: 0)

    {batch, random_key}
  end

  defnp action_log_probability(mu, stddev, action) do
    -0.5 * ((action - mu) / stddev) ** 2 - Nx.log(stddev * Nx.sqrt(2 * pi(Nx.type(stddev))))
  end
end
