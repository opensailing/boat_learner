defmodule ReinforcementLearningTest do
  use ExUnit.Case, async: true

  test "DDPG + multi-mark environment vectorizes based on the random_key" do
    state_input = Axon.input("state", shape: {nil, 3, 2})
    action_input = Axon.input("actions", shape: {nil, 1})

    actor_net =
      state_input
      |> Axon.dense(1)
      |> Axon.flatten()
      |> Axon.dense(1)

    critic_net =
      state_input
      |> Axon.dense(2)
      |> Axon.dense(3)
      |> Axon.nx(&Nx.take(&1, 0, axis: 1))
      |> Axon.flatten()
      |> Axon.dense(5)
      |> Axon.dense(7)
      |> then(&Axon.concatenate([&1, action_input]))
      |> Axon.dense(1)

    environment_to_state_features_fn = fn env_state ->
      Nx.stack([env_state.x, env_state.y])
    end

    state_features_memory_to_input_fn = fn state_features ->
      %{
        "state" => Nx.reshape(state_features, {:auto, 3, 2})
      }
    end

    ddpg = {
      ReinforcementLearning.Agents.DDPG,
      tau: 0.001,
      performance_memory_length: 6,
      actor_net: actor_net,
      critic_net: critic_net,
      actor_params: %{},
      actor_target_params: %{},
      critic_params: %{},
      critic_target_params: %{},
      state_features_memory_length: 3,
      experience_replay_buffer_max_size: 11,
      environment_to_state_features_fn: environment_to_state_features_fn,
      state_features_memory_to_input_fn: state_features_memory_to_input_fn,
      state_features_size: 2,
      training_frequency: 1,
      batch_size: 7,
      actor_optimizer: Axon.Optimizers.adam(),
      critic_optimizer: Axon.Optimizers.adam()
    }

    env = {
      BoatLearner.Environments.MultiMark,
      max_remaining_seconds: 10,
      coords: Nx.tensor([[0, 0, 10, 10], [0, 0, -10, -10]]),
      coord_probabilities: :uniform
    }

    random_key_init = Nx.Random.key(42)

    random_key_devec =
      Nx.Random.randint_split(random_key_init, 0, Nx.Constants.max_finite(:u32),
        type: :u32,
        shape: {5, 2}
      )

    vectorized_axes = [random_key: 5]

    random_key = Nx.revectorize(random_key_devec, vectorized_axes, target_shape: {2})

    assert %{vectorized_axes: [random_key: 5], shape: {2}, type: {:u, 32}} = random_key

    assert %Axon.Loop.State{
             step_state: %ReinforcementLearning{
               agent_state: agent_state,
               environment_state: environment_state
             }
           } =
             ReinforcementLearning.train(
               env,
               ddpg,
               &Function.identity/1,
               fn _ -> Nx.tensor([:nan]) end,
               num_episodes: 2,
               max_iter: 10,
               random_key: random_key
             )

    assert_vectorization = fn containers, expected ->
      containers
      |> Nx.Defn.Composite.flatten_list()
      |> Enum.each(fn
        %Nx.Tensor{} = t ->
          assert t.vectorized_axes == expected
          t

        t ->
          t
      end)
    end

    assert_vectorization.(
      [
        agent_state.actor_params,
        agent_state.actor_target_params,
        agent_state.critic_params,
        agent_state.critic_target_params,
        agent_state.actor_optimizer_state,
        agent_state.critic_optimizer_state
      ],
      []
    )

    assert_vectorization.(
      [
        agent_state.experience_replay_buffer,
        agent_state.performance_memory,
        agent_state.state_features_memory,
        agent_state.ou_process,
        agent_state.total_reward,
        agent_state.performance_threshold
      ],
      vectorized_axes
    )

    assert_vectorization.(
      [agent_state.gamma, agent_state.tau, agent_state.target_update_frequency],
      []
    )

    environment_state
    |> Map.drop([:__struct__, :polar_chart, :action_lower_limit, :action_upper_limit])
    |> Map.values()
    |> assert_vectorization.(vectorized_axes)

    environment_state
    |> Map.take([:polar_chart, :action_lower_limit, :action_upper_limit])
    |> Map.values()
    |> assert_vectorization.([])
  end

  test "DDPG + multi-mark environment stops correctly for partially finalized states" do
    state_input = Axon.input("state", shape: {nil, 3, 2})
    action_input = Axon.input("actions", shape: {nil, 1})

    actor_net =
      state_input
      |> Axon.dense(1)
      |> Axon.flatten()
      |> Axon.dense(1)

    critic_net =
      state_input
      |> Axon.dense(2)
      |> Axon.dense(3)
      |> Axon.nx(&Nx.take(&1, 0, axis: 1))
      |> Axon.flatten()
      |> Axon.dense(5)
      |> Axon.dense(7)
      |> then(&Axon.concatenate([&1, action_input]))
      |> Axon.dense(1)

    environment_to_state_features_fn = fn env_state ->
      Nx.stack([env_state.x, env_state.y])
    end

    state_features_memory_to_input_fn = fn state_features ->
      %{
        "state" => Nx.reshape(state_features, {:auto, 3, 2})
      }
    end

    ddpg = {
      ReinforcementLearning.Agents.DDPG,
      tau: 0.001,
      performance_memory_length: 6,
      actor_net: actor_net,
      critic_net: critic_net,
      actor_params: %{},
      actor_target_params: %{},
      critic_params: %{},
      critic_target_params: %{},
      state_features_memory_length: 3,
      experience_replay_buffer_max_size: 11,
      environment_to_state_features_fn: environment_to_state_features_fn,
      state_features_memory_to_input_fn: state_features_memory_to_input_fn,
      state_features_size: 2,
      training_frequency: 1,
      batch_size: 7,
      actor_optimizer: Axon.Optimizers.adam(),
      critic_optimizer: Axon.Optimizers.adam()
    }

    env = {
      BoatLearner.Environments.MultiMark,
      max_remaining_seconds: 10,
      coords: Nx.tensor([[10, 10, 10, 10], [0, 0, -10, -10]]),
      coord_probabilities: :uniform
    }

    random_key_init = Nx.Random.key(42)

    random_key_devec =
      Nx.Random.randint_split(random_key_init, 0, Nx.Constants.max_finite(:u32),
        type: :u32,
        shape: {5, 2}
      )

    vectorized_axes = [random_key: 5]

    random_key = Nx.revectorize(random_key_devec, vectorized_axes, target_shape: {2})

    assert %{vectorized_axes: [random_key: 5], shape: {2}, type: {:u, 32}} = random_key

    assert %Axon.Loop.State{
             step_state: %ReinforcementLearning{
               agent_state: agent_state,
               environment_state: environment_state
             }
           } =
             ReinforcementLearning.train(
               env,
               ddpg,
               &Function.identity/1,
               fn _ -> Nx.tensor([:nan]) end,
               num_episodes: 2,
               max_iter: 10,
               random_key: random_key
             )
  end
end
