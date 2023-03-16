defmodule ReinforcementLearning.Environments.Gridworld do
  import Nx.Defn

  @behaviour ReinforcementLearning.Environment

  @derive {Nx.Container,
           containers: [
             :x,
             :y,
             :prev_x,
             :prev_y,
             :target_x,
             :target_y,
             :reward,
             :reward_stage,
             :is_terminal,
             :possible_targets
           ],
           keep: []}
  defstruct [
    :x,
    :y,
    :prev_x,
    :prev_y,
    :target_x,
    :target_y,
    :reward,
    :reward_stage,
    :is_terminal,
    :possible_targets
  ]

  @min_x 0
  @max_x 10
  @min_y 0
  @max_y 10

  def bounding_box, do: {@min_x, @max_x, @min_y, @max_y}

  # x, y, target_x, target_y, prev_x, prev_y, reward_stage
  def state_vector_size, do: 7

  @impl true
  # up, down, left, right
  def num_actions, do: 4

  @impl true
  def init(random_key, opts) do
    opts = Keyword.validate!(opts, [:possible_targets])

    possible_targets =
      opts[:possible_targets] || raise ArgumentError, "missing option :possible_targets"

    reset(random_key, %__MODULE__{possible_targets: possible_targets})
  end

  @impl true
  def reset(random_key, %__MODULE__{} = state) do
    reward = Nx.tensor(0, type: :f32)
    {x, random_key} = Nx.Random.randint(random_key, @min_x, @max_x)

    # possible_targets is a {n, 2} tensor that contains targets that we want to sample from
    # this is so we avoid retraining every episode on the same target, which can lead to
    # overfitting
    {target, random_key} =
      Nx.Random.choice(random_key, state.possible_targets, samples: 1, axis: 0)

    target = Nx.reshape(target, {2})

    reward_stage = y = Nx.tensor(0, type: :s64)

    state = %{
      state
      | x: x,
        y: y,
        prev_x: x,
        prev_y: y,
        target_x: target[0],
        target_y: target[1],
        reward: reward,
        reward_stage: reward_stage,
        is_terminal: Nx.tensor(0, type: :u8)
    }

    {state, random_key}
  end

  @impl true
  defn apply_action(state, action) do
    %{x: x, y: y} = env = state.environment_state

    # 0: up, 1: down, 2: right, 3: left
    {new_x, new_y} =
      cond do
        action == 0 ->
          {x, Nx.min(y + 1, @max_y + 1)}

        action == 1 ->
          {x, Nx.max(y - 1, @min_y - 1)}

        action == 2 ->
          {Nx.min(@max_x + 1, x + 1), y}

        true ->
          {Nx.max(x - 1, @min_x - 1), y}
      end

    new_env = %{env | x: new_x, y: new_y, prev_x: x, prev_y: y}

    updated_env =
      new_env
      |> is_terminal_state()
      |> calculate_reward()

    %{state | environment_state: updated_env}
  end

  defnp calculate_reward(env) do
    %{
      x: x,
      y: y,
      target_x: target_x,
      target_y: target_y,
      is_terminal: is_terminal,
      reward_stage: reward_stage
    } = env

    reward =
      cond do
        is_terminal and Nx.abs(target_x - x) <= 1.5 and Nx.abs(target_y - y) <= 1.5 ->
          1

        is_terminal ->
          -1

        true ->
          -0.01
      end

    %{env | reward_stage: reward_stage, reward: reward}
  end

  defnp is_terminal_state(%{x: x, y: y, target_x: target_x, target_y: target_y} = env) do
    is_terminal =
      (Nx.abs(target_x - x) <= 1.5 and Nx.abs(target_y - y) <= 1.5) or x < @min_x or x > @max_x or
        y < @min_y or
        y > @max_y

    %{env | is_terminal: is_terminal}
  end

  defn as_state_vector(%{
         x: x,
         y: y,
         target_x: target_x,
         target_y: target_y,
         prev_x: prev_x,
         prev_y: prev_y,
         reward_stage: reward_stage
       }) do
    x = (x - @min_x) / (@max_x - @min_x)
    y = (y - @min_y) / (@max_y - @min_y)

    target_x = (target_x - @min_x) / (@max_x - @min_x)
    target_y = (target_y - @min_y) / (@max_y - @min_y)
    prev_x = (prev_x - @min_x) / (@max_x - @min_x)
    prev_y = (prev_y - @min_y) / (@max_y - @min_y)

    Nx.stack([
      x,
      y,
      target_x,
      target_y,
      prev_x,
      prev_y,
      reward_stage
    ])
    |> Nx.new_axis(0)
  end
end
