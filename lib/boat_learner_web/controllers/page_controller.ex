defmodule BoatLearnerWeb.PageController do
  use BoatLearnerWeb, :controller

  def index(conn, _params) do
    render(conn, "index.html")
  end
end
