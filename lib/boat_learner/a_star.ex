defmodule BoatLearner.AStar do
  @moduledoc """
  Uses the A* path finding algorithm for
  path planning in a sailboat course.
  """

  defmodule PriorityQueue do
    def new do
      :gb_trees.empty()
    end

    def put(queue, key, value) do
      :gb_trees.insert(key, value, queue)
    end

    def put_if_cheaper(queue, key, value) do
      case :gb_trees.lookup(queue, key) do
        :none ->
          put(queue, key, value)

        {:value, current} ->
          if value < current do
            :gb_trees.update(key, value, queue)
          else
            queue
          end
      end
    end

    def pop(queue) do
      if :gb_trees.is_empty(queue) do
        {:error, :empty}
      else
        {:ok, :gb_trees.take_smallest(queue)}
      end
    end
  end

  defmodule Node do
    defstruct [:x, :y, :parent, f: 0, g: 0, h: 0, tacking: false, heading: 0, speed: 0]

    def equal?(this, that) do
      key(this) == key(that)
    end

    def push(queue, node) do
      PriorityQueue.put(queue, key(node), {node.f, node})
    end

    def push_if_cheaper(queue, node) do
      PriorityQueue.put_if_cheaper(queue, key(node), {node.f, node})
    end

    def pop(queue) do
      PriorityQueue.pop(queue, :empty)
    end

    def close(closed_list, node) do
      MapSet.put(closed_list, key(node))
    end

    def closed?(closed_list, node) do
      MapSet.member?(closed_list, key(node))
    end

    def key(node), do: {node.x, node.y}
  end

  @speed_over_ground_max 10

  def solve({start_x, start_y}, {goal_x, goal_y}, opts \\ []) do
    opts = Keyword.validate!(opts, [:start_heading])
    open_list = :ets.new(:open_list, [:ordered_set])
    closed_list = MapSet.new()

    start_node = %Node{x: start_x, y: start_y, heading: opts[:start_heading]}
    goal_node = %Node{x: goal_x, y: goal_y}

    open_list = Node.push(open_list, start_node)

    solve_loop(open_list, closed_list, goal_node, opts)
  end

  defp solve_loop(open_list, closed_list, goal_node, opts) do
    case Node.pop(open_list) do
      {:error, :empty} ->
        {:error, :no_path}

      {:ok, {_key, {_, current_node}, open_list}} ->
        closed_list = Node.close(closed_list, current_node)

        if goal_reached?(current_node, goal_node, opts) do
          {:ok, get_path(current_node)}
        else
          neighbors =
            current_node
            |> neighbors(goal_node, opts)
            |> Enum.reject(&Node.closed?(closed_list, &1))
            |> Enum.map(&update_neighbor_cost(&1, current_node, goal_node, opts))

          Enum.reduce(neighbors, open_list, &Node.push_if_cheaper(&2, &1))
        end
    end
  end

  def neighbors(node, goal_node, opts) do
    angles = Enum.map(40..180//5, &(&1 * :math.pi() / 180))
    angles = angles ++ Enum.map(angles, &(2 * :math.pi() - &1))

    angles =
      Enum.filter(angles, fn angle ->
        abs(angle_difference(angle, node.heading)) <= :math.pi() / 2
      end)

    case angles do
      [] ->
        []

      angles ->
        angles_t = Nx.tensor(angles)
        speeds = speed_interpolation(angles_t)
        water_current = get_water_current(node.x, node.y, opts)
        vmgs = vmg(node.position, goal_node.position, angles_t, speeds, water_current)
        idx = Nx.argsort(vmgs, direction: :desc)[0..(min(Nx.size(vmgs), 2) - 1)]
        tacking_angles = Nx.take(angles_t, idx)
        angles = Nx.concatenate([Nx.reshape(node.heading, {1}), tacking_angles])
        speeds = Nx.concatenate([Nx.reshape(node.speed, {1}), tacking_angles])

        speed_x = Nx.sin(angles) |> Nx.multiply(speeds) |> Nx.add(water_current[0])
        speed_y = Nx.cos(angles) |> Nx.multiply(speeds) |> Nx.add(water_current[1])

        next_x = Nx.add(node.position[0], Nx.multiply(speed_x, dt))
        next_y = Nx.add(node.position[1], Nx.multiply(speed_y, dt))

        next_pos = Nx.stack([next_x, next_y], axis: 1)

        next_pos =
          Nx.round(Nx.divide(next_pos, opts[:grid_resolution]))
          |> Nx.multiply(opts[:grid_resolution])
          |> Nx.to_list()

        speeds = Nx.add(Nx.pow(speed_x, 2), Nx.pow(speed_y, 2)) |> Nx.sqrt()

        Enum.zip_reduce(
          [
            next_pos,
            Nx.to_list(angles),
            Nx.to_list(speeds)
          ],
          [],
          fn [
               [x, y],
               angle,
               speed
             ],
             acc ->
            if valid_position?(x, y, opts) do
              [
                %Node{
                  x: x,
                  y: y,
                  speed: speed,
                  heading: angle,
                  tacking: angle != node.heading,
                  parent: node
                }
                | acc
              ]
            else
              acc
            end
          end
        )
    end
  end

  defp valid_position?(x, y, opts) do
    x >= opts[:min_x] and x <= opts[:max_x] and y >= opts[:min_y] and y <= opts[:max_y]
  end

  defp angle_difference(a, b) do
    # signed angle difference
    a = :math.fmod(a, :math.pi() * 2)
    b = :math.fmod(b, :math.pi() * 2)
    diff = a - b

    cond do
      diff > :math.pi() ->
        diff - 2 * :math.pi()

      diff < -:math.pi() ->
        diff + 2 * :math.pi()

      true ->
        diff
    end
  end

  def heuristic(node_a, node_b) do
    distance(node_a, node_b) / @speed_over_ground_max
  end

  def goal_reached?(current_node, goal_node, opts) do
    distance(current_node, goal_node) <= 3 * opts[:boat_length]
  end

  defp distance(node_a, node_b) do
    dx = node_a.x - node_b.x
    dy = node_a.y - node_b.y
    :math.sqrt(dx ** 2 + dy ** 2)
  end

  defp update_neighbor_cost(next_node, current_node, goal_node, opts) do
    penalty = if(next_node.tacking, do: opts[:tacking_penalty], else: 0)
    g = current_node.g + opts[:dt] + penalty
    h = heuristic(next_node, goal_node)
    f = g + h
    %{next_node | f: f, g: g, h: h}
  end

  defp get_path(node, path \\ [])
  defp get_path(%Node{parent: nil} = node, path), do: Enum.reverse([node | path])
  defp get_path(node, path), do: get_path(node.parent, [node | path])
end
