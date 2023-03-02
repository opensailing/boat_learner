defmodule BoatLearner.Navigation do
  @moduledoc """
  Behaviour for different navigation strategies and objectives
  """

  import Nx.Defn

  @derive {Nx.Container,
           containers: [:agent_state, :environment_state], keep: [:agent, :environment]}
  defstruct [
    :agent,
    :agent_state,
    :environment,
    :environment_state,
  ]

  @doc """

  """
  @spec train(
          environment :: module,
          initial_environment_state :: map(),
          agent :: module,
          initial_agent_state :: map(),
          epoch_completed_callback :: (map() -> :ok),
          num_episodes :: pos_integer(),
          random_key :: Nx.Tensor.t()
        )
  def train(
        env,
        initial_environment_state,
        agent,
        initial_agent_state,
        epoch_completed_callback,
        num_episodes,
        random_key \\ nil
      ) do
    random_key = random_key \\ Nx.Random.key(System.system_time())
    {agent_state, random_key} = agent.init(random_key, initial_agent_state)
    {environment_state, random_key} = env.init(random_key, initial_environment_state)

    loop(random_key,
      agent: agent,
      agent_state: agent_state,
      environment: environment,
      environment_state: environment_state,
      epoch_completed_callback: epoch_completed_callback
      num_episodes: num_episodes
    )
  end

  defp loop(random_key, opts) do
    epoch_completed_callback = Keyword.fetch!(opts, :epoch_completed_callback)
    num_episodes = Keyword.fetch!(opts, :num_episodes)
    state_opts = Keyword.take(opts, [:agent, :agent_state, :environment, :environment_state])
    initial_state = reset_state(struct!(%__MODULE__{}, state_opts))


    loop =
      Axon.Loop.loop(&batch_step(&1, &2, opts), fn _input, loop_state ->
        reset_state(loop_state)
      end)

    loop
    |> Axon.Loop.handle(
      :epoch_started,
      &{:continue, %{&1 | step_state: reset_state(&1.step_state)}}
    )
    |> Axon.Loop.handle(:epoch_completed, fn loop_state ->
      loop_state = %{loop_state | loop_state.epoch + 1}

      loop_state = tap(loop_state, epoch_completed_callback)
      {:continue, loop_state}
    end)
    |> Axon.Loop.run(Stream.cycle([Nx.tensor(1)]), initial_state, iterations: 1, epochs: num_episodes)
  end

  defnp reset_state(
          %__MODULE__{
            agent: agent,
            agent_state: agent_state,
            environment: environment,
            environment_state: environment_state
          } = loop_state
        ) do
    %{
      loop_state
      | agent_state: agent.init(agent_state),
        environment_state: environment.init(environment_state)
    }
  end

  defnp batch_step(_inputs, prev_state) do
    {state, _, _} =
      while {%{agent: agent} = prev_state, iter = 0, is_terminal = Nx.tensor(0, type: :u8)}, i < @max_iter and not is_terminal do
        {action, state, random_key} = agent.select_action(prev_state, policy_predict_fn)
        {reward, reward_stage, is_terminal, state, random_key} = apply_action_to_environment(state, action)

        state =
          prev_state
          |> agent.record_observation(action, reward, state)
          |> agent.optimize_model()
          |> agent.persist_trajectory()

        {state, iter + 1, is_terminal}
      end

    state
  end
end
