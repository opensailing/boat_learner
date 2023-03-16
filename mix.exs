defmodule BoatLearner.MixProject do
  use Mix.Project

  def project do
    [
      app: :boat_learner,
      version: "0.1.0",
      elixir: "~> 1.12",
      elixirc_paths: elixirc_paths(Mix.env()),
      compilers: Mix.compilers(),
      start_permanent: Mix.env() == :prod,
      aliases: aliases(),
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
      {:exla, github: "elixir-nx/nx", sparse: "exla", override: true},
      {:scholar, github: "elixir-nx/scholar"},
      {:axon, github: "elixir-nx/axon"},
      {:kino, "~> 0.8"},
      {:kino_vega_lite, "~> 0.1"},
      {:table_rex, "~> 3.1"}
    ]
  end

  # Aliases are shortcuts or tasks specific to the current project.
  # For example, to install project dependencies and perform other setup tasks, run:
  #
  #     $ mix setup
  #
  # See the documentation for `Mix` for more info on aliases.
  defp aliases do
    [
      setup: ["deps.get", "ecto.setup"],
      test: ["", "test"],
    ]
  end
end
