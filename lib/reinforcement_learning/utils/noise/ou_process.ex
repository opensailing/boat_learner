defmodule ReinforcementLearning.Utils.Noise.OUProcess do
  @moduledoc """
  Ornstein-Uhlenbeck (OU for short) noise generator
  for temporally correlated noise.
  """

  import Nx.Defn

  @derive {Nx.Container, keep: [], containers: [:size, :theta, :sigma, :mu, :x]}
  defstruct [:size, :theta, :sigma, :mu, :x]

  deftransform init(size, opts \\ []) do
    opts = Keyword.validate!(opts, theta: 0.15, sigma: 0.2, type: :f32, mu: 0)
    theta = opts[:theta]
    sigma = opts[:sigma]
    type = opts[:type]
    mu = opts[:mu]

    x = Nx.broadcast(Nx.as_type(mu, type), {size})
    %__MODULE__{size: size, theta: theta, sigma: sigma, mu: mu, x: x}
  end

  defn reset(process) do
    x = Nx.broadcast(process.mu, process.x)
    %__MODULE__{process | x: x}
  end

  defn sample(random_key, process) do
    %__MODULE__{x: x, sigma: sigma, theta: theta, mu: mu} = process
    {sample, random_key} = Nx.Random.normal(random_key, shape: Nx.shape(x))
    dx = theta * (mu - x) + sigma * sample
    x = x + dx
    {%__MODULE__{process | x: x}, random_key}
  end
end
