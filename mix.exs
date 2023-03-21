defmodule BoatLearner.MixProject do
  use Mix.Project

  def project do
    [
      app: :boat_learner,
      version: "0.1.0",
      elixir: "~> 1.14",
      elixirc_paths: elixirc_paths(Mix.env()),
      compilers: Mix.compilers(),
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Configuration for the OTP application.
  #
  # Type `mix help compile.app` for more information.
  def application do
    [extra_applications: [:logger, :runtime_tools]]
  end

  # Specifies which paths to compile per environment.
  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  # Specifies your project dependencies.
  #
  # Type `mix help deps` for examples and options.
  defp deps do
    [
      {:jason, "~> 1.2"},
      {:nx, github: "elixir-nx/nx", sparse: "nx", override: true},
      {:scholar, github: "elixir-nx/scholar"},
      {:axon, github: "elixir-nx/axon"},
      {:kino, "~> 0.8"},
      {:kino_vega_lite, "~> 0.1"},
      {:table_rex, "~> 3.1"}
      | backend()
    ]
  end

  defp backend do
    case System.get_env("BOAT_LEARNER_BACKEND") do
      "torchx" ->
        [{:torchx, github: "elixir-nx/nx", sparse: "torchx", override: true}]

      "binary" ->
        []

      _ ->
        [{:exla, github: "elixir-nx/nx", sparse: "exla", override: true}]
    end
  end
end
