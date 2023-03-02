defmodule BoatLearner.Navigation do
  @moduledoc """
  Behaviour for different navigation strategies and objectives
  """

  @derive {Nx.Container,
           containers: [:agent_state, :environment_state, :random_key], keep: [:trajectory]}
  defstruct [
    :agent,
    :agent_state,
    :environment,
    :environment_state,
    :random_key,
    :trajectory
  ]

  @doc """

  ## Examples

      iex> BoatLearner.Navigation.train(
      ...>  BoatLearner.Environments.Gridworld,
      ...>  BoatLearner.Agents.DQN,
      ...>  &IO.puts(inspect(&1)),
      ...>  2,
      ...>  Nx.tensor([[-5, 10], [5, 20], [3, 10], [-2, 15]]),
      ...>  Nx.tensor([[-25, -25, 0, 0]])
      ...>  )
      #<Axon.Loop>
  """
  @spec train(
          environment :: module,
          agent :: module,
          epoch_completed_callback :: (map() -> :ok),
          num_episodes :: pos_integer(),
          possible_targets :: Nx.Tensor.t(),
          obstacles :: Nx.Tensor.t(),
          opts :: keyword()
        ) :: Axon.Loop.t()
  def train(
        environment,
        agent,
        epoch_completed_callback,
        num_episodes,
        possible_targets,
        obstacles,
        opts \\ []
      ) do
    opts =
      Keyword.validate!(opts, [
        :random_key,
        :q_policy,
        :experience_replay_buffer,
        :experience_replay_buffer_index,
        :persisted_experience_replay_buffer_entries,
        :max_iter
      ])

    random_key = opts[:random_key] || Nx.Random.key(System.system_time())
    max_iter = opts[:max_iter]

    agent_opts =
      Keyword.take(opts, [
        :q_policy,
        :experience_replay_buffer,
        :experience_replay_buffer_index,
        :persisted_experience_replay_buffer_entries
      ])

    {agent_state, random_key} =
      agent.init(
        random_key,
        environment.state_vector_size(),
        environment.num_actions(),
        agent_opts
      )

    {environment_state, random_key} = environment.init(random_key, obstacles, possible_targets)

    initial_state = %__MODULE__{
      agent: agent,
      agent_state: agent_state,
      environment: environment,
      environment_state: environment_state,
      random_key: random_key,
      trajectory: []
    }

    loop(
      agent,
      environment,
      possible_targets,
      initial_state,
      epoch_completed_callback: epoch_completed_callback,
      num_episodes: num_episodes,
      max_iter: max_iter
    )
  end

  defp loop(agent, environment, possible_targets, initial_state, opts) do
    epoch_completed_callback = Keyword.fetch!(opts, :epoch_completed_callback)
    num_episodes = Keyword.fetch!(opts, :num_episodes)
    max_iter = Keyword.fetch!(opts, :max_iter)

    loop = Axon.Loop.loop(&batch_step(&1, &2, agent, environment, max_iter))

    loop
    |> Axon.Loop.handle(
      :epoch_started,
      &{:continue,
       %{&1 | step_state: reset_state(&1.step_state, agent, environment, possible_targets)}}
    )
    |> Axon.Loop.handle(:epoch_completed, fn loop_state ->
      loop_state = %{loop_state | epoch: loop_state.epoch + 1}

      loop_state = tap(loop_state, epoch_completed_callback)
      {:continue, loop_state}
    end)
    |> Axon.Loop.run(Stream.cycle([Nx.tensor(1)]), initial_state,
      iterations: 1,
      epochs: num_episodes
    )
  end

  defp reset_state(
         %__MODULE__{
           agent_state: agent_state,
           environment_state: environment_state,
           random_key: random_key
         } = loop_state,
         agent,
         environment,
         possible_targets
       ) do
    {agent_state, random_key} = agent.reset(random_key, agent_state)

    {environment_state, random_key} =
      environment.reset(random_key, possible_targets, environment_state)

    %{
      loop_state
      | agent_state: agent_state,
        environment_state: environment_state,
        random_key: random_key,
        trajectory: []
    }
  end

  defp batch_step(
         _inputs,
         initial_state,
         agent,
         environment,
         max_iter
       ) do
    Enum.reduce_while(1..max_iter, initial_state, fn
      iter, prev_state ->
        {action, state} = agent.select_action(prev_state, iter, &environment.as_state_vector/1)
        {reward, _reward_stage, is_terminal, state} = environment.apply_action(state, action)

        state =
          prev_state
          |> agent.record_observation(
            action,
            reward,
            is_terminal,
            state,
            &environment.as_state_vector/1
          )
          |> agent.optimize_model()

        # |> persist_trajectory(iter)

        # is_terminal = Nx.to_number(is_terminal) == 1

        # if is_terminal do
        #   {:halt, state}
        # else
        #   {:cont, state}
        # end
        {:cont, state}
    end)
  end

  defp persist_trajectory(state, iter) do
    %{x: x, y: y} = state.environment_state
    %{state | trajectory: [{Nx.to_number(x), Nx.to_number(y), iter} | state.trajectory]}
  end
end
