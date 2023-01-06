defmodule BoatLearner.Repo do
  use Ecto.Repo,
    otp_app: :boat_learner,
    adapter: Ecto.Adapters.Postgres
end
