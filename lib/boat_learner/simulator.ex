defmodule BoatLearner.Simulator do
  @moduledoc """
  Utility functions for updating a boat's
  position and velocity
  """
  import Nx.Defn

  @pi :math.pi()

  @r_by_tws %{
    6 => [4.4, 5.1, 5.59, 5.99, 6.2, 6.37, 6.374, 6.25, 6.02, 5.59, 4.82, 4.11, 3.57, 3.22, 3.08],
    8 => [
      5.41,
      6.05,
      6.37,
      6.54,
      6.72,
      6.88,
      6.99,
      6.98,
      6.77,
      6.42,
      5.93,
      5.24,
      4.65,
      4.23,
      4.07
    ],
    10 => [5.93, 6.38, 6.68, 6.9, 7.1, 7.2, 7.35, 7.48, 7.37, 7, 6.57, 6.12, 5.61, 5.16, 4.97],
    12 => [6.15, 6.56, 6.86, 7.1, 7.3, 7.49, 7.65, 7.85, 7.99, 7.56, 7.12, 6.71, 6.3, 5.97, 5.8],
    16 => [
      6.39,
      6.81,
      7.12,
      7.4,
      7.67,
      7.94,
      8.17,
      8.42,
      8.74,
      9.06,
      8.36,
      7.78,
      7.32,
      7.03,
      6.89
    ],
    20 => [
      6.54,
      6.97,
      7.3,
      7.63,
      7.97,
      8.32,
      8.68,
      9.01,
      9.3,
      9.76,
      10.32,
      9.29,
      8.46,
      8.03,
      7.85
    ],
    24 => [
      6.61,
      7.05,
      7.41,
      7.79,
      8.21,
      8.68,
      9.18,
      9.62,
      9.92,
      10.35,
      11.2,
      11.41,
      10.12,
      9.42,
      9.11
    ]
  }

  @optimal_upwind_and_downwind_by_tws %{
    6 => [{4.62, 42.7}, {5.02, 137.6}],
    8 => [{5.56, 41.6}, {5.64, 144.5}],
    10 => [{5.77, 37.6}, {5.93, 154.2}],
    12 => [{5.88, 35.6}, {6.3, 160.1}],
    16 => [{6.05, 34.4}, {7.01, 170.8}],
    20 => [{6.2, 34.5}, {9.48, 148.1}],
    24 => [{6.28, 34.6}, {11.78, 147.4}]
  }

  @tws 12

  [{r0, theta0}, {r1, theta1}] = @optimal_upwind_and_downwind_by_tws[@tws]
  @r [r0, r1 | @r_by_tws[@tws]]
  @theta [theta0, theta1 | Enum.to_list(40..180//10)]

  # The grid will be represented in meters
  @kts_to_meters_per_sec 0.514444

  defn init, do: init(Nx.tensor(@r), Nx.tensor(@theta) * @pi / 180)

  @doc """
  Returns the `model` tuple for prediction of interpolated TWS
  """
  defn init(r, theta) do
    # theta is in rad!
    r = Nx.as_type(r, :f64) * @kts_to_meters_per_sec
    theta = Nx.as_type(theta, :f64)

    spline_model = Scholar.Interpolation.BezierSpline.fit(theta, r)

    # use the tail-end of the spline prediction as part of our linear model
    dtheta = 0.1
    min_theta = Nx.new_axis(dtheta + Nx.reduce_min(theta), 0)

    dr = Scholar.Interpolation.BezierSpline.predict(spline_model, min_theta + dtheta)

    # Fit {0, 0} and {min_theta, dr} as points for the linear "extrapolation"
    zero = Nx.new_axis(0, 0)

    linear_model =
      Scholar.Interpolation.Linear.fit(
        Nx.concatenate([zero, min_theta, theta]),
        Nx.concatenate([zero, dr, r])
      )

    cutoff_angle = Nx.reduce_min(spline_model.k, axes: [0])[0]

    {linear_model, spline_model, cutoff_angle}
  end

  defn speed({linear_model, spline_model_polar, cutoff_angle}, angle) do
    # Change algorithms where the spline ends at the lower end
    angle = Nx.abs(angle) + Nx.as_type(1.0e-12, :f64)
    angle = Nx.select(angle > @pi, 2 * @pi - angle, angle)

    linear_pred = Scholar.Interpolation.Linear.predict(linear_model, angle)
    spline_pred = Scholar.Interpolation.BezierSpline.predict(spline_model_polar, angle)

    Nx.select(angle <= cutoff_angle, linear_pred, spline_pred)
  end

  @doc """
  Receives a batched tensor in which the last dim is `[x, y]`
  and updates given compatible-shaped tensors of `[speed, angle]` and `[dt]`
  """
  defn update_position(current_position, velocity, dt) do
    r = Nx.slice_along_axis(velocity, 0, 1, axis: 1)
    theta = Nx.slice_along_axis(velocity, 1, 1, axis: 1)

    current_position +
      dt * r * Nx.concatenate([Nx.sin(theta), Nx.cos(theta)], axis: 1)
  end
end
