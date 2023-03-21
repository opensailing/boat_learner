import Config

backend_env = System.get_env("BOAT_LEARNER_BACKEND")

{backend, defn_opts} =
  case backend_env do
    "torchx" ->
      {Torchx.Backend, []}

    "binary" ->
      {Nx.BinaryBackend, []}

    _ ->
      {EXLA.Backend, compiler: EXLA}
  end

config :nx,
  default_backend: backend,
  global_default_backend: backend,
  default_defn_options: defn_opts
