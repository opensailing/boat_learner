defmodule BoatLearner.Environments.Gridworld do
  import Nx.Defn

  @derive {Nx.Container,
           containers: [
             :x,
             :y,
             :prev_x,
             :prev_y,
             :target_x,
             :target_y,
             :grid,
             :reward,
             :reward_stage,
             :is_terminal
           ],
           keep: [:obstacles]}
  defstruct [
    :x,
    :y,
    :prev_x,
    :prev_y,
    :target_x,
    :target_y,
    :grid,
    :obstacles,
    :reward,
    :reward_stage,
    :is_terminal
  ]

  @type t :: %__MODULE__{}

  @min_x 0
  @max_x 10
  @min_y 0
  @max_y 10

  @x_tol (@max_x - @min_x) * 0.01
  @y_tol (@max_y - @min_y) * 0.01

  @max_distance :math.sqrt((@max_x - @min_x) ** 2 + (@max_y - @min_y) ** 2)

  def bounding_box, do: {@min_x, @max_x, @min_y, @max_y}

  # x, y, target_x, target_y, prev_x, prev_y, reward_stage
  @state_vector_size 7
  def state_vector_size, do: @state_vector_size

  # up, down, left, right
  @num_actions 4
  def num_actions, do: @num_actions

  @type state :: ReinforcementLearning.t()
  @type tensor :: Nx.Tensor.t()

  @spec init(random_key :: tensor, obstacles :: tensor, possible_targets :: tensor) ::
          {t(), random_key :: tensor}
  def init(random_key, obstacles, possible_targets) do
    grid =
      if obstacles == [] do
        empty_grid()
      else
        obstacles
        |> to_obstacles_indices()
        |> build_grid()
      end

    reset(random_key, possible_targets, %__MODULE__{
      obstacles: obstacles,
      grid: grid
    })
  end

  @spec reset(random_key :: tensor, possible_targets :: tensor, t()) ::
          {t(), random_key :: tensor}
  def reset(random_key, possible_targets, %__MODULE__{} = state) do
    reward = Nx.tensor(0, type: :f32)
    {x, random_key} = Nx.Random.randint(random_key, @min_x, @max_x)

    # possible_targets is a {n, 2} tensor that contains targets that we want to sample from
    # this is so we avoid retraining every episode on the same target, which can lead to
    # overfitting
    {target, random_key} = Nx.Random.choice(random_key, possible_targets, samples: 1, axis: 0)
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

  defp to_obstacles_indices(obstacles) do
    obstacles_idx =
      for [min_x, max_x, min_y, max_y] <- obstacles do
        Nx.tensor(
          for i <- min_x..max_x, j <- min_y..max_y do
            [i - @min_x, j - @min_y]
          end
        )
      end

    Nx.concatenate(obstacles_idx, axis: 0)
  end

  defnp empty_grid, do: Nx.broadcast(Nx.tensor(0, type: :u8), {@max_x - @min_x, @max_y - @min_y})

  defnp build_grid(obstacles_idx) do
    %{shape: {m, n}} = grid = empty_grid()

    idx_top =
      Nx.concatenate(
        [
          Nx.broadcast(0, {n, 1}),
          Nx.iota({n, 1})
        ],
        axis: 1
      )

    idx_bottom =
      Nx.concatenate(
        [
          Nx.broadcast(m - 1, {n, 1}),
          Nx.iota({n, 1})
        ],
        axis: 1
      )

    idx_left =
      Nx.concatenate(
        [
          Nx.iota({m, 1}),
          Nx.broadcast(0, {m, 1})
        ],
        axis: 1
      )

    idx_right =
      Nx.concatenate(
        [
          Nx.iota({m, 1}),
          Nx.broadcast(n - 1, {m, 1})
        ],
        axis: 1
      )

    idx = Nx.concatenate([idx_top, idx_bottom, idx_left, idx_right], axis: 0)
    idx = Nx.concatenate([idx, obstacles_idx])

    updates = Nx.broadcast(Nx.tensor(1, type: :u8), {Nx.axis_size(idx, 0)})

    Nx.indexed_put(grid, idx, updates)
  end

  @neg_1 -1
  @spec apply_action(state :: state, action :: tensor) ::
          {reward :: tensor, reward_stage :: tensor, is_terminal :: tensor, state :: state}
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
      prev_x: prev_x,
      prev_y: prev_y,
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

  defnp distance(x, target_x, y, target_y), do: Nx.sqrt((x - target_x) ** 2 + (y - target_y) ** 2)

  defnp is_terminal_state(%{x: x, y: y, target_x: target_x, target_y: target_y, grid: grid} = env) do
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
