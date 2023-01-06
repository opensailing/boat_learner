defmodule BoatLearner.Navigation do
  @moduledoc """
  Behaviour for different navigation strategies and objectives
  """

  defstruct [:id, :state, :epoch]

  @type t :: %__MODULE__{
          id: integer,
          state: any(),
          epoch: integer
        }

  @callback train :: t()
end
