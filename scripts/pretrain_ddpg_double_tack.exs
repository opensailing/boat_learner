my_app_root = Path.join(__DIR__, "..")

Mix.install(
  [
    {:boat_learner, path: my_app_root, env: :dev}
  ],
  config_path: Path.join(my_app_root, "config/config.exs"),
  lockfile: Path.join(my_app_root, "mix.lock"),
  consolidate_protocols: false,
  system_env: %{"XLA_TARGET" => System.get_env("XLA_TARGET", "cpu")}
)

Nx.default_backend({EXLA.Backend, client: :cuda})
Nx.global_default_backend({EXLA.Backend, client: :cuda})
Nx.Defn.default_options(compiler: EXLA, client: :cuda)
Nx.Defn.global_default_options(compiler: EXLA, client: :cuda)

key = Nx.Random.key(1)

{min_x, max_x, min_y, max_y} = BoatLearner.Environments.DoubleTack.bounding_box()
target_y = 100

actions = Nx.tensor([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])

xs = Nx.linspace(-25, 25, n: 10)
ys = Nx.linspace(0, target_y * 0.75, n: 10)

max_remaining_seconds = 500

remaining_seconds = Nx.linspace(0, max_remaining_seconds, n: 30)
headings = Nx.linspace(0, :math.pi() * 2, n: 12)

space = [actions, xs, ys, remaining_seconds, headings, actions]

IO.puts("Generating cross-product indices")

indices = space |> Enum.with_index(& &1.shape |> Nx.iota() |> Nx.vectorize(:"ax_#{&2}")) |> Nx.stack() |> Nx.devectorize() |> Nx.reshape({:auto, length(space)})

IO.puts("Getting values")

xs = Nx.take(xs, indices[[0..-1//1, 1]])
ys = Nx.take(ys, indices[[0..-1//1, 2]])
remaining_seconds = Nx.take(remaining_seconds, indices[[0..-1//1, 3]])
headings = Nx.take(headings, indices[[0..-1//1, 4]])
next_actions = Nx.take(actions, indices[[0..-1//1, 5]])

actions = Nx.take(actions, indices[[0..-1//1, 0]])

IO.puts("Values taken")

alias BoatLearner.Environments.DoubleTack, as: DT

defmodule GenerateSamples do
  import Nx.Defn

  {min_x, max_x, min_y, _max_y} = DT.bounding_box()

  @min_x min_x
  @max_x max_x
  @min_y min_y
  # @max_y max_y

  defn run(
    key,
    xs,
    ys,
    remaining_seconds,
    headings,
    target_y,
    actions,
    next_actions,
    polar_chart,
    opts \\ []
    ) do

    max_remaining_seconds = opts[:max_remaining_seconds]

    speeds = DT.speed_from_heading(polar_chart, headings)

    rl = %ReinforcementLearning{
      random_key: 1,
      agent_state: 1,
      iteration: 1,
      episode: 1,
      trajectory: 1
    }

    env = %DT{
      x: Nx.vectorize(xs, :i),
      y: Nx.vectorize(ys, :i),
      heading: Nx.vectorize(headings, :i),
      speed: Nx.vectorize(speeds, :i),
      angle_to_target: Nx.vectorize(headings, :i),
      target_y: target_y,
      reward: 0,
      is_terminal: 0,
      polar_chart: polar_chart,
      remaining_seconds: Nx.vectorize(remaining_seconds, :i),
      max_remaining_seconds: max_remaining_seconds,
      vmg: 0,
      tack_count: 0,
      has_tacked: 0
    }

    actions = Nx.vectorize(actions, :i)
    next_actions = Nx.vectorize(next_actions, :i)

    %{environment_state: first_env} = DT.apply_action(%{rl | environment_state: env}, actions)

    %{environment_state: second_env} =
      DT.apply_action(%{rl | environment_state: first_env}, next_actions)

    state_vector = environment_to_state_features(first_env)
    next_state_vector = environment_to_state_features(second_env)

    samples =
      Nx.concatenate([
        state_vector,
        Nx.reshape(first_env.is_terminal, {1}),
        Nx.reshape(first_env.reward, {1}),
        Nx.reshape(actions, {1}),
        Nx.reshape(next_actions, {1}),
        next_state_vector
      ])

    samples = Nx.devectorize(samples)

    {samples, key}
  end

  defnp environment_to_state_features(env_state) do
    # These might seem redundant, but will make more sense for multi-input models
    normalize_x = fn x ->
      (x - @min_x) / (@max_x - @min_x)
    end

    normalize_y = fn y ->
      (y - @min_y) / (env_state.target_y - @min_y)
    end

    normalize_speed = fn s ->
      s / 10
    end

    normalize_angle = fn a ->
      a / (2 * Nx.Constants.pi())
    end

    norm_x = normalize_x.(env_state.x)
    norm_y = normalize_y.(env_state.y)

    distance_squared = Nx.pow(norm_x, 2) |> Nx.add(Nx.pow(norm_y, 2))

    # d**2 = x**2 + y**2, max(d**2) = 1**2 + 1**2 = 2
    distance_squared_norm = Nx.divide(distance_squared, 2)

    [
      norm_x,
      norm_y,
      distance_squared_norm,
      normalize_speed.(env_state.vmg),
      normalize_angle.(env_state.heading),
      normalize_angle.(env_state.angle_to_target),
      normalize_angle.(42.7 / 180 * Nx.Constants.pi()),
      env_state.has_tacked
    ]
    |> Nx.stack()
  end
end

polar_chart = DT.init_polar_chart()

IO.puts("Generating samples")

{samples, key} =
  GenerateSamples.run(
    key,
    xs,
    ys,
    remaining_seconds,
    headings,
    target_y,
    actions,
    next_actions,
    polar_chart,
    max_remaining_seconds: max_remaining_seconds
  )

IO.inspect(samples, label: "samples")

## Pretrain the Critic

state_features_memory_length = 1
state_features_size = 8
num_actions = 1

state_input = Axon.input("state", shape: {nil, state_features_memory_length, state_features_size})
action_input = Axon.input("actions", shape: {nil, num_actions})

critic_net =
  state_input
  |> Axon.nx(fn state_memory ->
    Nx.take(state_memory, state_features_memory_length - 1, axis: 1)
  end)
  |> Axon.dense(64, activation: :linear)
  |> Axon.batch_norm()
  |> Axon.activation(:gelu)
  |> Axon.dropout(rate: 0.25)
  |> Axon.dense(256, activation: :linear)
  |> Axon.batch_norm()
  |> Axon.activation(:gelu)
  |> Axon.dropout(rate: 0.25)
  |> then(&Axon.concatenate([&1, action_input]))
  |> Axon.dense(128, activation: :linear)
  |> Axon.batch_norm()
  |> Axon.activation(:relu)
  |> Axon.dropout()
  |> Axon.dense(64, activation: :linear)
  |> Axon.batch_norm()
  |> Axon.activation(:relu)
  |> Axon.dropout()
  |> Axon.dense(1,
    kernel_initializer: Axon.Initializers.glorot_uniform(scale: 1.0e-3),
    bias_initializer: Axon.Initializers.glorot_uniform(scale: 1.0e-3)
  )

defmodule CriticStepState do
  import Nx.Defn

  @state_features_size state_features_size
  @num_actions num_actions

  defn run(
         batch,
         %{optimizer_state: optimizer_state, model_params: model_params, gamma: gamma},
         predict_fn,
         optimizer_update_fn
       ) do
    state_batch =
      batch
      |> Nx.slice_along_axis(0, @state_features_size, axis: 1)
      |> Nx.reshape({:auto, 1, @state_features_size})

    is_terminal_batch = Nx.slice_along_axis(batch, @state_features_size, 1, axis: 1)
    reward_batch = Nx.slice_along_axis(batch, @state_features_size + 1, 1, axis: 1)

    action_batch = Nx.slice_along_axis(batch, @state_features_size + 2, @num_actions, axis: 1)

    next_action_batch =
      Nx.slice_along_axis(batch, @state_features_size + 2 + @num_actions, @num_actions, axis: 1)

    next_state_batch =
      batch
      |> Nx.slice_along_axis(@state_features_size + 2 * @num_actions + 2, @state_features_size,
        axis: 1
      )
      |> Nx.reshape({:auto, 1, @state_features_size})

    non_final_mask = not is_terminal_batch

    {loss, gradient} =
      value_and_grad(
        model_params,
        fn model_params ->
          q_target = predict_fn.(model_params, next_state_batch, next_action_batch)

          %{shape: {n, 1}} = q = predict_fn.(model_params, state_batch, action_batch)

          %{shape: {m, 1}} = backup = reward_batch + gamma * non_final_mask * q_target

          case {m, n} do
            {m, n} when m != n ->
              raise "shape mismatch for batch values"

            _ ->
              1
          end

          Axon.Losses.mean_squared_error(backup, q, reduction: :mean)
        end
      )

    {updates, optimizer_state} = optimizer_update_fn.(gradient, optimizer_state, model_params)

    model_params = Polaris.Updates.apply_updates(model_params, updates)

    %{loss: loss, gamma: gamma, model_params: model_params, optimizer_state: optimizer_state}
  end
end

{init_fn, predict_fn} = Axon.build(critic_net, seed: 1)

predict_fn = fn params, state_features_memory, action_vector ->
  input = %{
    "state" => Nx.reshape(state_features_memory, {:auto, 1, state_features_size}),
    "actions" => action_vector
  }

  predict_fn.(params, input)
end

contents = File.read!(Path.join(System.fetch_env!("HOME"), "Desktop/double_tack_critic_init.nx"))
model_init_params = Nx.deserialize(contents)
# model_init_params = %{}

model_params =
  init_fn.(
    %{
      "actions" => Nx.template({1, num_actions}, :f32),
      "state" => Nx.template({1, state_features_memory_length, state_features_size}, :f32)
    },
    model_init_params
  )

{optimizer_init_fn, optimizer_update_fn} =
  Polaris.Updates.compose(
    Polaris.Updates.clip(delta: 2),
    Axon.Optimizers.adamw(1.0e-6, decay: 0.005, eps: 1.0e-12)
  )

optimizer_state = optimizer_init_fn.(model_params)

{samples, key} = Nx.Random.shuffle(key, samples, axis: 0)

IO.inspect(samples, label: "shuffled samples")

loop_result =
  Axon.Loop.loop(&CriticStepState.run(&1, &2, predict_fn, optimizer_update_fn))
  |> Axon.Loop.log(
    &"Epoch: #{&1.epoch} Loss: #{Nx.to_number(&1.step_state.loss)}\n",
    event: :epoch_completed
  )
  |> Axon.Loop.run(
    Nx.to_batched(samples, 2500),
    %{
      gamma: 0.95,
      model_params: model_params,
      optimizer_state: optimizer_state,
      loss: 0.0
    },
    epochs: 500,
    client: :cuda
  )

contents = Nx.serialize(loop_result.step_state.model_params)
File.write!(Path.join(System.fetch_env!("HOME"), "Desktop/double_tack_critic_init.nx"), contents)
