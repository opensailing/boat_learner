defmodule ReinforcementLearning.Utils.Noise.OUProcessTest do
  use ExUnit.Case, async: true

  alias ReinforcementLearning.Utils.Noise.OUProcess

  test "generates samples with given shape" do
    key = Nx.Random.key(1)

    state = OUProcess.init(2)
    range = 1..10

    {values, _key} =
      Enum.map_reduce(range, {key, state}, fn _, {prev_key, state} ->
        {state, key} = OUProcess.sample(prev_key, state)

        refute Nx.backend_copy(key) == Nx.backend_copy(prev_key)

        {state.x, {key, state}}
      end)

    assert Enum.map(values, &Nx.backend_copy/1) == [
             Nx.tensor([-0.161521315574646, -0.04836982861161232], backend: Nx.BinaryBackend),
             Nx.tensor([-0.02224855124950409, -0.040264032781124115], backend: Nx.BinaryBackend),
             Nx.tensor([-0.09898111969232559, 0.007571592926979065], backend: Nx.BinaryBackend),
             Nx.tensor([0.2752320468425751, 0.27117180824279785], backend: Nx.BinaryBackend),
             Nx.tensor([0.19806110858917236, 0.374011367559433], backend: Nx.BinaryBackend),
             Nx.tensor([0.33261623978614807, 0.45093613862991333], backend: Nx.BinaryBackend),
             Nx.tensor([0.5560829043388367, 0.3771272897720337], backend: Nx.BinaryBackend),
             Nx.tensor([0.418714702129364, 0.24803754687309265], backend: Nx.BinaryBackend),
             Nx.tensor([0.043423742055892944, 0.10074643790721893], backend: Nx.BinaryBackend),
             Nx.tensor([-0.3225523829460144, 0.020469389855861664], backend: Nx.BinaryBackend)
           ]
  end
end
