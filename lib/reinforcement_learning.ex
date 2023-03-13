defmodule ReinforcementLearning do
  @moduledoc """
  Reinforcement Learning training (to-do: and inference) framework
  """

  import Nx.Defn

  @derive {Nx.Container,
           containers: [:agent_state, :environment_state, :random_key, :iteration, :trajectory],
           keep: []}
  defstruct [
    :agent,
    :agent_state,
    :environment,
    :environment_state,
    :random_key,
    :iteration,
    :trajectory
  ]

  @doc """

  ## Examples

      iex> ReinforcementLearning.train(
      ...>  BoatLearner.Environments.Gridworld,
      ...>  ReinforcementLearning.Agents.DQN,
      ...>  &IO.puts(inspect(&1)),
      ...>  2,
      ...>  Nx.tensor([[-5, 10], [5, 20], [3, 10], [-2, 15]]),
      ...>  Nx.tensor([[-25, -25, 0, 0]])
      ...>  )
      #Axon.Loop<>
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
        :agent_init_opts,
        :max_iter
      ])

    random_key = opts[:random_key] || Nx.Random.key(System.system_time())
    max_iter = opts[:max_iter]

    agent_init_opts = Keyword.fetch!(opts, :agent_init_opts)

    {agent_state, random_key} =
      agent.init(
        random_key,
        environment.state_vector_size(),
        environment.num_actions(),
        agent_init_opts
      )

    {environment_state, random_key} = environment.init(random_key, obstacles, possible_targets)

    initial_state = %__MODULE__{
      agent: agent,
      agent_state: agent_state,
      environment: environment,
      environment_state: environment_state,
      random_key: random_key,
      trajectory: Nx.broadcast(Nx.tensor(:nan, type: :f32), {3, max_iter}),
      iteration: Nx.tensor(0, type: :s64)
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

    loop = Axon.Loop.loop(&batch_step(&1, &2, agent, environment))

    loop
    |> Axon.Loop.handle(
      :epoch_started,
      &{:continue,
       %{&1 | step_state: reset_state(&1.step_state, agent, environment, possible_targets)}}
    )
    |> Axon.Loop.handle(:epoch_completed, fn loop_state ->
      loop_state = tap(loop_state, epoch_completed_callback)
      {:continue, loop_state}
    end)
    |> Axon.Loop.handle(:iteration_completed, fn loop_state ->
      is_terminal = Nx.to_number(loop_state.step_state.environment_state.is_terminal)

      if is_terminal == 1 do
        {:halt_epoch, loop_state}
      else
        {:continue, loop_state}
      end
    end)
    |> Axon.Loop.handle(:epoch_halted, fn loop_state ->
      loop_state = tap(loop_state, epoch_completed_callback)
      {:halt_epoch, loop_state}
    end)
    |> Axon.Loop.run(Stream.cycle([Nx.tensor(1)]), initial_state,
      iterations: max_iter,
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
        trajectory: Nx.broadcast(Nx.tensor(:nan, type: :f32), loop_state.trajectory),
        iteration: Nx.tensor(0, type: :s64)
    }
  end

  defp batch_step(
         _inputs,
         prev_state,
         agent,
         environment
       ) do
    # Enum.reduce_while(1..max_iter, initial_state, fn
    # iter, prev_state ->
    {action, state} =
      agent.select_action(prev_state, prev_state.iteration, &environment.as_state_vector/1)

    %{environment_state: %{reward: reward, is_terminal: is_terminal}} =
      state = environment.apply_action(state, action)

    prev_state
    |> agent.record_observation(
      action,
      reward,
      is_terminal,
      state,
      &environment.as_state_vector/1
    )
    |> agent.optimize_model()
    |> persist_trajectory()
  end

  defnp persist_trajectory(%{trajectory: trajectory, iteration: iteration} = step_state) do
    %{x: x, y: y} = step_state.environment_state

    idx =
      Nx.tensor([
        [0, 0],
        [1, 0],
        [2, 0]
      ]) + Nx.new_axis(Nx.stack([0, iteration]), 0)

    trajectory = Nx.indexed_put(trajectory, idx, Nx.stack([iteration, x, y]))
    %{step_state | trajectory: trajectory, iteration: iteration + 1}
  end
end
