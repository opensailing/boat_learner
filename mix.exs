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
      {:nx, path: "../nx/nx", override: true},
      {:scholar, "~> 0.2"},
      {:axon, "~> 0.6"},
      {:kino, "~> 0.9"},
      {:kino_vega_lite, "~> 0.1"},
      {:table_rex, "~> 3.1"},
      {:rein, path: "../rein"}
      | backend()
    ]
  end

  defp backend do
    case System.get_env("BOAT_LEARNER_BACKEND") do
      "torchx" ->
        [{:torchx, "~> 0.6"}]

      "binary" ->
        []

      _ ->
        # [{:exla, "~> 0.6"}]
      [{:exla, path: "../nx/exla", override: true}]
    end
  end
end
