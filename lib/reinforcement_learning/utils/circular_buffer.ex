defmodule ReinforcementLearning.Utils.CircularBuffer do
  @moduledoc """
  Circular Buffer utility via Nx Containers.
  """

  import Nx.Defn

  @derive {Nx.Container, containers: [:data, :index, :size], keep: []}
  defstruct [:data, :index, :size]

  def new(shape, opts \\ [init_value: 0, type: :f32]) do
    %__MODULE__{
      data: Nx.broadcast(Nx.tensor(opts[:init_value], type: opts[:type]), shape),
      size: 0,
      index: 0
    }
  end

  defn append(buffer, item) do
    starts = append_start_indices(buffer)
    n = Nx.axis_size(buffer.data, 0)
    index = Nx.remainder(buffer.index + 1, n)
    size = Nx.min(n, buffer.size + 1)

    %{
      buffer
      | data: Nx.put_slice(buffer.data, starts, Nx.new_axis(item, 0)),
        size: size,
        index: index
    }
  end

  deftransformp append_start_indices(buffer) do
    [buffer.index | List.duplicate(0, tuple_size(buffer.data.shape) - 1)]
  end

  @doc """
  Returns the data starting at the current index.

  The oldest persisted entry will be the first entry in
  the result, and so on.
  """
  defn ordered_data(buffer) do
    n = elem(buffer.data.shape, 0)
    indices = Nx.remainder(Nx.iota({n}) + buffer.index, n)
    Nx.take(buffer.data, indices)
  end
end
