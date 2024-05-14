defmodule BoatLearner.AStar do
  @moduledoc """
  Uses the A* path finding algorithm for
  path planning in a sailboat course.
  """

  import Nx.Defn
  import Nx.Constants, only: [pi: 0]

  @one_deg_in_rad 1 * :math.pi() / 180

  @kts_to_meters_per_sec 0.514444
  @speed_kts [
    4.4,
    5.1,
    5.59,
    5.99,
    6.2,
    6.37,
    6.374,
    6.25,
    6.02,
    5.59,
    4.82,
    4.11,
    3.57,
    3.22,
    3.08
  ]

  @dead_zone_angle 30 * @one_deg_in_rad

  @speed_kts [4.62, 5.02 | @speed_kts]
  @theta_deg [42.7, 137.6 | Enum.to_list(40..180//10)]

  # @theta_deg Enum.to_list(40..180//10)
  @theta Enum.map(@theta_deg, &(&1 * @one_deg_in_rad))
  @speed Enum.map(@speed_kts, &(&1 * @kts_to_meters_per_sec))
  @speed_over_ground_max Enum.max(@speed)

  defmodule PriorityQueue do
    defstruct [:f_tree, :node_tree]

    def new do
      %__MODULE__{f_tree: :gb_trees.empty(), node_tree: :gb_trees.empty()}
    end

    def put(queue, key, {f, node}) do
      %__MODULE__{
        f_tree: :gb_trees.insert(key, f, queue.f_tree),
        node_tree: :gb_trees.insert(key, node, queue.node_tree)
      }
    end

    def put_if_cheaper(queue, key, {f, node}) do
      case :gb_trees.lookup(key, queue.node_tree) do
        :none ->
          put(queue, key, {f, node})

        {:value, %{g: current_g}} ->
          if node.g <= current_g do
            %__MODULE__{
              f_tree: :gb_trees.update(key, f, queue.f_tree),
              node_tree: :gb_trees.update(key, node, queue.node_tree)
            }
          else
            queue
          end
      end
    end

    def pop(queue) do
      if :gb_trees.is_empty(queue.f_tree) do
        {:error, :empty}
      else
        {key, f, f_tree} = :gb_trees.take_smallest(queue.f_tree)
        {node, node_tree} = :gb_trees.take(key, queue.node_tree)

        queue = %__MODULE__{
          f_tree: f_tree,
          node_tree: node_tree
        }

        {:ok, {key, {f, node}, queue}}
      end
    end
  end

  defmodule Node do
    @derive {Inspect, only: [:x, :y, :f, :g, :heading, :tacking, :speed]}
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
      PriorityQueue.pop(queue)
    end

    def close(closed_list, node) do
      MapSet.put(closed_list, key(node))
    end

    def closed?(closed_list, node) do
      MapSet.member?(closed_list, key(node))
    end

    def key(node), do: {node.x, node.y, node.tacking}
  end

  def solve({start_x, start_y}, {goal_x, goal_y}, opts \\ []) do
    opts =
      Keyword.validate!(opts, [
        :start_heading,
        :boat_length,
        :max_x,
        :min_x,
        :max_y,
        :min_y,
        :dt,
        :grid_resolution,
        :tacking_penalty
      ])

    open_list = PriorityQueue.new()
    closed_list = MapSet.new()
    polar_chart = init_polar_chart()

    start_node = %Node{x: start_x, y: start_y, heading: opts[:start_heading]}
    goal_node = %Node{x: goal_x, y: goal_y}

    open_list = Node.push(open_list, start_node)

    solve_loop(open_list, closed_list, goal_node, [{:polar_chart, polar_chart} | opts])
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

          open_list = Enum.reduce(neighbors, open_list, &Node.push_if_cheaper(&2, &1))
          solve_loop(open_list, closed_list, goal_node, opts)
        end
    end
  end

  deftransformp wrap_phase(angles) when is_list(angles) do
    two_pi = :math.pi() * 2

    Enum.map(angles, fn angle ->
      :math.fmod(:math.fmod(angle, two_pi) + two_pi, two_pi)
    end)
  end

  deftransformp wrap_phase(angle) do
    two_pi = Nx.multiply(pi(), 2)

    angle
    |> Nx.remainder(two_pi)
    |> Nx.add(two_pi)
    |> Nx.remainder(two_pi)
  end

  def neighbors(node, goal_node, opts) do
    angles = Enum.map(40..180//1, &(&1 * :math.pi() / 180))

    angles =
      if is_nil(node.parent) do
        angles
      else
        Enum.filter(angles, fn angle ->
          abs(angle_difference(angle, node.heading)) <= :math.pi() / 2
        end)
      end

    angles = wrap_phase(angles ++ Enum.map(angles, &(2 * :math.pi() - &1)))

    case angles do
      [] ->
        []

      angles ->
        dt = opts[:dt]
        angles_t = Nx.tensor(angles, type: :f64)
        speeds = speed_from_heading(opts[:polar_chart], angles_t)
        water_current = get_water_current(node.x, node.y, opts)
        vmgs = vmg(node.x, node.y, goal_node.x, goal_node.y, angles_t, speeds, water_current)

        idx = Nx.argsort(vmgs, direction: :desc)[0..1]
        angles = Nx.concatenate([Nx.take(angles_t, idx), Nx.reshape(node.heading, {1})])
        speeds = Nx.concatenate([Nx.take(speeds, idx), Nx.reshape(node.speed, {1})])

        speed_x =
          Nx.sin(angles) |> Nx.multiply(speeds) |> Nx.add(water_current[0])

        speed_y =
          Nx.cos(angles) |> Nx.multiply(speeds) |> Nx.add(water_current[1])

        next_x = Nx.add(node.x, Nx.multiply(speed_x, dt))
        next_y = Nx.add(node.y, Nx.multiply(speed_y, dt))

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

  defp get_water_current(_x, _y, _opts) do
    Nx.tensor([0.0, 0.0])
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
  defp get_path(%Node{parent: nil} = node, path), do: [node | path]
  defp get_path(node, path), do: get_path(node.parent, [%{node | parent: nil} | path])

  def init_polar_chart do
    # data for the boat at TWS=6
    theta = Nx.tensor(@theta, type: :f64)
    speed = Nx.tensor(@speed, type: :f64)

    spline_model = Scholar.Interpolation.BezierSpline.fit(theta, speed)
    # use the tail-end of the spline prediction as part of our linear model
    dtheta = Nx.f64(0.1)
    min_theta = theta |> Nx.reduce_min() |> Nx.new_axis(0)

    dspeed =
      Scholar.Interpolation.BezierSpline.predict(spline_model, Nx.subtract(min_theta, dtheta))

    # Fit {0, 0}, {@dead_zone_angle, 0} and {min_theta, dspeed} as points for the linear "extrapolation"
    dead_zone_thetas = Nx.tensor([0], type: :f64)
    dead_zone_speeds = Nx.tensor([0], type: :f64)

    linear_model =
      Scholar.Interpolation.Linear.fit(
        Nx.concatenate([dead_zone_thetas, min_theta, theta]),
        Nx.concatenate([dead_zone_speeds, dspeed, speed])
      )

    cutoff_angle = Nx.reduce_min(spline_model.k, axes: [0])[0]

    {linear_model, spline_model, cutoff_angle}
  end

  defn speed_from_heading({linear_model, spline_model, cutoff_angle}, angle) do
    # angle is already maintained between 0 and 2pi
    # so we only need to calculate the "absolute value"
    # (as if the angle was between -pi and pi)
    angle = Nx.as_type(wrap_phase(angle), :f64)
    angle = Nx.select(angle > pi(), 2 * pi() - angle, angle)

    linear_pred = Scholar.Interpolation.Linear.predict(linear_model, angle)

    spline_pred =
      Scholar.Interpolation.BezierSpline.predict(
        spline_model,
        angle |> Nx.devectorize() |> Nx.flatten()
      )
      |> Nx.reshape(Nx.devectorize(angle) |> Nx.shape())
      |> Nx.vectorize(angle.vectorized_axes)

    (angle <= @dead_zone_angle)
    |> Nx.select(0, Nx.select(angle <= cutoff_angle, linear_pred, spline_pred))
    |> Nx.clip(0, @speed_over_ground_max)
  end

  defn vmg(node_x, node_y, mark_x, mark_y, headings, speeds, water_current) do
    velocity_vector =
      Nx.stack([speeds * Nx.sin(headings), speeds * Nx.cos(headings)], axis: 1) + water_current

    original_vec_axes = velocity_vector.vectorized_axes

    velocity_vector =
      Nx.revectorize(velocity_vector, [collapsed: :auto], target_shape: {2})

    direction_vector = Nx.stack([mark_x - node_x, mark_y - node_y])
    direction_magnitude = Nx.LinAlg.norm(direction_vector)

    vmgs = Nx.dot(velocity_vector, direction_vector) / direction_magnitude

    Nx.revectorize(vmgs, original_vec_axes, target_shape: {:auto})
  end
end
