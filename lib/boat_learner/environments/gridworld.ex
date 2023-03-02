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
             :obstacles,
             :reward_stage
           ],
           keep: []}
  defstruct [
    :x,
    :y,
    :prev_x,
    :prev_y,
    :target_x,
    :target_y,
    :grid,
    :obstacles,
    :reward_stage
  ]

  @type t :: %__MODULE__{}

  @min_x -25
  @max_x 25
  @min_y 0
  @max_y 50

  @x_tol (@max_x - @min_x) * 0.01
  @y_tol (@max_y - @min_y) * 0.01

  @max_distance :math.sqrt((@max_x - @min_x) ** 2 + (@max_y - @min_y) ** 2)

  # x, y, target_x, target_y, prev_x, prev_y
  @state_vector_size 6
  def state_vector_size, do: @state_vector_size

  # up, down, left, right
  @num_actions 4
  def num_actions, do: @num_actions

  @type state :: BoatLearner.Navigation.t()
  @type tensor :: Nx.Tensor.t()

  @spec init(random_key :: tensor, obstacles :: tensor, possible_targets :: tensor) ::
          {t(), random_key :: tensor}
  def init(random_key, obstacles, possible_targets) do
    grid =
      obstacles
      |> to_obstacles_indices()
      |> build_grid()

    reset(random_key, possible_targets, %__MODULE__{
      obstacles: obstacles,
      grid: grid
    })
  end

  @spec reset(random_key :: tensor, possible_targets :: tensor, t()) ::
          {t(), random_key :: tensor}
  def reset(random_key, possible_targets, %__MODULE__{} = state) do
    y = Nx.tensor(0, type: :f32)
    {x, random_key} = Nx.Random.randint(random_key, -5, 5)

    # possible_targets is a {n, 2} tensor that contains targets that we want to sample from
    # this is so we avoid retraining every episode on the same target, which can lead to
    # overfitting
    {target, random_key} = Nx.Random.choice(random_key, possible_targets, samples: 1, axis: 0)
    target = Nx.reshape(target, {2})

    reward_stage = Nx.tensor(0, type: :s64)

    state = %{
      state
      | x: x,
        y: y,
        prev_x: x,
        prev_y: y,
        target_x: target[0],
        target_y: target[1],
        reward_stage: reward_stage
    }

    {state, random_key}
  end

  defp to_obstacles_indices(obstacles) do
    obstacles = obstacles |> Nx.to_flat_list() |> Enum.chunk_every(4)

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

  defnp build_grid(obstacles_idx) do
    %{shape: {m, n}} =
      grid = Nx.broadcast(Nx.tensor(0, type: :u8), {@max_x - @min_x, @max_y - @min_y})

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

    # 0: up, 1: down, 2: left, 3: right
    x_updates = Nx.tensor([0, 0, 1, @neg_1])
    y_updates = Nx.tensor([1, @neg_1, 0, 0])
    new_x = x + x_updates[action]
    new_y = y + y_updates[action]

    new_env = %{env | x: new_x, y: new_y, prev_x: x, prev_y: y}

    is_terminal = is_terminal_state(new_env)

    {reward, reward_stage, updated_env} = calculate_reward(new_env)
    {reward, reward_stage, is_terminal, %{state | environment_state: updated_env}}
  end

  defnp calculate_reward(env) do
    %{
      x: x,
      y: y,
      target_x: target_x,
      target_y: target_y,
      prev_x: prev_x,
      prev_y: prev_y,
      reward_stage: reward_stage
    } = env

    distance = distance(x, target_x, y, target_y)
    prev_distance = distance(prev_x, target_x, prev_y, target_y)

    distance_reward = 1 - distance / @max_distance
    progress_reward = Nx.select(distance < prev_distance, 1, 0)

    reward_stage =
      if reward_stage == 0 and distance < 50 do
        1
      else
        reward_stage
      end

    reward =
      cond do
        y < @min_y + @y_tol ->
          -1

        # avoid the bounding box
        x < @min_x + 5 * @x_tol or x > @max_x - 5 * @x_tol or y > @max_y - 5 * @y_tol ->
          -100

        reward_stage == 1 and distance < 50 ->
          Nx.select(distance < prev_distance, 10, -1)

        true ->
          distance_reward + progress_reward
      end

    {reward, reward_stage, %{env | reward_stage: reward_stage}}
  end

  defnp distance(x, target_x, y, target_y), do: Nx.sqrt((x - target_x) ** 2 + (y - target_y) ** 2)

  defnp is_terminal_state(%{x: x, y: y, target_x: target_x, target_y: target_y, grid: grid}) do
    x_clipped = clip_to_grid(x, @min_x, @max_x)
    y_clipped = clip_to_grid(y, @min_y, @max_y)

    # terminates on collision or if reached target
    (target_x == x and target_y == y) or grid[x_clipped][y_clipped] != 0
  end

  defnp clip_to_grid(coord, min, max) do
    (coord - min)
    |> Nx.ceil()
    |> Nx.as_type(:s64)
    |> Nx.clip(0, max - min)
  end

  defn as_state_vector(%{
         x: x,
         y: y,
         target_x: target_x,
         target_y: target_y,
         prev_x: prev_x,
         prev_y: prev_y
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
      prev_y
    ])
    |> Nx.new_axis(0)
  end
end
