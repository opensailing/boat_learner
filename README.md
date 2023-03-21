# BoatLearner

This system is mostly designed for usage with livebook.

## Environment variables

- BOAT_LEARNER_BACKEND
  If set to "torchx", will use Torchx as the default backend. If "binary", uses plain Nx.BinaryBackend.
  Otherwise, will use EXLA as the defual backend and compiler.

  For EXLA and Torchx, each have their own available environment variables as well.