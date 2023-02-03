defmodule BoatLearner.SimulatorTest do
  use ExUnit.Case

  @pi :math.pi()
  describe "unwrap_phase/1" do
    test "property" do
      t =
        Nx.concatenate([
          Nx.tensor([@pi, -@pi]),
          Nx.linspace(0, 4 * @pi, n: 32)
        ])

      u = unwrap_phase(t)

      assert_all_close(Nx.reduce_max(u), @pi)
      assert_all_close(Nx.reduce_min(u), -@pi)

      assert_all_close(Nx.cos(t), Nx.cos(u))
      assert_all_close(Nx.sin(t), Nx.sin(u))
    end
  end

  def assert_all_close(left, right, opts \\ [atol: 1.0e-4, rtol: 1.0e-4])

  def assert_all_close(left, right, opts) when is_number(left),
    do: assert_all_close(Nx.tensor(left), right, opts)

  def assert_all_close(left, right, opts) when is_number(right),
    do: assert_all_close(left, Nx.tensor(right), opts)

  def assert_all_close(%{shape: {}} = left, %{shape: {}} = right, opts) do
    assert_in_delta Nx.to_number(left), Nx.to_number(right), opts[:atol]
  end

  def assert_all_close(left, right, opts) do
    assert Nx.all_close(left, right, opts)
  end
end
