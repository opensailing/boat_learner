import Config

client =
  if System.get_env("XLA_TARGET", "") =~ ~r"cuda" do
    :cuda
  else
    :host
  end

config :nx,
  default_backend: {EXLA.Backend, client: client},
  global_default_backend: {EXLA.Backend, client: client},
  default_defn_options: [compiler: EXLA, client: client]
