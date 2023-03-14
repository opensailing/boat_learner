defmodule ReinforcementLearning.Agents.DQN do
  import Nx.Defn

  @type state :: ReinforcementLearning.t()
  @type tensor :: Nx.Tensor.t()

  @learning_rate 1.0e-5
  @adamw_decay 1.0e-2
  @eps 1.0e-7
  @experience_replay_buffer_num_entries 10_000

  @eps_start 0.996
  @eps_end 0.01

  @train_every_steps 64
  @adamw_decay 0.01

  @batch_size 256

  @gamma 0.99

  @derive {Nx.Container,
           containers: [
             :q_policy,
             :q_policy_optimizer_state,
             :loss,
             :experience_replay_buffer,
             :experience_replay_buffer_index,
             :persisted_experience_replay_buffer_entries,
             :total_reward,
             :eps_max_iter
           ],
           keep: [:optimizer_update_fn, :policy_predict_fn, :state_vector_size, :num_actions]}
  defstruct [
    :state_vector_size,
    :num_actions,
    :q_policy,
    :q_policy_optimizer_state,
    :policy_predict_fn,
    :optimizer_update_fn,
    :experience_replay_buffer,
    :experience_replay_buffer_index,
    :persisted_experience_replay_buffer_entries,
    :loss,
    :total_reward,
    :eps_max_iter
  ]

  @type t :: %__MODULE__{}

  @spec init(
          random_key :: tensor(),
          state_vector_size :: pos_integer(),
          num_actions :: pos_integer(),
          opts :: keyword()
        ) :: {t(), random_key :: tensor()}
  def init(random_key, state_vector_size, num_actions, opts \\ []) do
    opts =
      Keyword.validate!(opts, [
        :q_policy,
        :experience_replay_buffer,
        :experience_replay_buffer_index,
        :persisted_experience_replay_buffer_entries,
        :eps_max_iter
      ])

    policy_net = dqn(state_vector_size, num_actions)

    {policy_init_fn, policy_predict_fn} = Axon.build(policy_net, seed: 0)

    {optimizer_init_fn, optimizer_update_fn} =
      Axon.Updates.clip()
      |> Axon.Updates.compose(
        Axon.Optimizers.adamw(@learning_rate, eps: @eps, decay: @adamw_decay)
      )

    input = %{"state" => Nx.template({1, state_vector_size}, :f32)}

    initial_q_policy_state = opts[:q_policy] || raise "missing initial q_policy"
    eps_max_iter = opts[:eps_max_iter] || raise "missing :eps_max_iter"

    q_policy = policy_init_fn.(input, initial_q_policy_state)

    q_policy_optimizer_state = optimizer_init_fn.(q_policy)

    reset(random_key, %__MODULE__{
      eps_max_iter: eps_max_iter,
      state_vector_size: state_vector_size,
      num_actions: num_actions,
      q_policy: q_policy,
      q_policy_optimizer_state: q_policy_optimizer_state,
      policy_predict_fn: policy_predict_fn,
      optimizer_update_fn: optimizer_update_fn,
      # prev_state_vector, target_x, target_y, action, reward, is_terminal, next_state_vector
      experience_replay_buffer:
        opts[:experience_replay_buffer] ||
          Nx.broadcast(
            Nx.tensor(:nan, type: :f32),
            {@experience_replay_buffer_num_entries, 2 * state_vector_size + 3}
          ),
      experience_replay_buffer_index:
        opts[:experience_replay_buffer_index] || Nx.tensor(0, type: :s64),
      persisted_experience_replay_buffer_entries:
        opts[:persisted_experience_replay_buffer_entries] || Nx.tensor(0, type: :s64)
    })
  end

  @spec reset(random_key :: tensor(), state :: t()) :: {t(), random_key :: tensor()}
  def reset(random_key, state) do
    total_reward = Nx.tensor(0, type: :f32)
    loss = Nx.broadcast(Nx.tensor(0, type: :f32), {@batch_size, 1})

    {%{state | loss: loss, total_reward: total_reward}, random_key}
  end

  @spec select_action(
          state,
          iteration :: non_neg_integer(),
          as_state_vector_fn :: (map() -> tensor())
        ) ::
          {action :: tensor, state :: state}
  defn select_action(
         %ReinforcementLearning{random_key: random_key, agent_state: agent_state} = state,
         iteration,
         as_state_vector_fn
       ) do
    %{
      q_policy: q_policy,
      policy_predict_fn: policy_predict_fn,
      num_actions: num_actions,
      eps_max_iter: eps_max_iter
    } = agent_state

    {sample, random_key} = Nx.Random.uniform(random_key)

    eps_threshold = @eps_end + (@eps_start - @eps_end) * Nx.exp(-5 * iteration / eps_max_iter)

    {action, random_key} =
      if sample > eps_threshold do
        action =
          q_policy
          |> policy_predict_fn.(%{
            "state" => as_state_vector_fn.(state.environment_state)
          })
          |> Nx.argmax()

        {action, random_key}
      else
        Nx.Random.randint(random_key, 0, num_actions, type: :s64)
      end

    {action, %{state | random_key: random_key}}
  end

  @spec record_observation(
          state :: state,
          action :: tensor,
          reward :: tensor,
          is_terminal :: tensor,
          next_state :: state,
          as_state_vector_fn :: (map -> tensor)
        ) :: state
  defn record_observation(
         %{environment_state: env_state},
         action,
         reward,
         is_terminal,
         %{environment_state: next_env_state} = state,
         as_state_vector_fn
       ) do
    state_vector = as_state_vector_fn.(env_state)
    next_state_vector = as_state_vector_fn.(next_env_state)

    idx = Nx.stack([state.agent_state.experience_replay_buffer_index, 0]) |> Nx.new_axis(0)

    shape = {Nx.size(state_vector) + 3 + Nx.size(next_state_vector), 1}

    index_template = Nx.concatenate([Nx.broadcast(0, shape), Nx.iota(shape, axis: 0)], axis: 1)

    updates =
      Nx.concatenate([
        Nx.flatten(state_vector),
        Nx.stack([action, reward, is_terminal]),
        Nx.flatten(next_state_vector)
      ])

    experience_replay_buffer =
      Nx.indexed_put(state.agent_state.experience_replay_buffer, idx + index_template, updates)

    experience_replay_buffer_index =
      Nx.remainder(
        state.agent_state.experience_replay_buffer_index + 1,
        @experience_replay_buffer_num_entries
      )

    entries = state.agent_state.persisted_experience_replay_buffer_entries

    persisted_experience_replay_buffer_entries =
      Nx.select(
        entries < @experience_replay_buffer_num_entries,
        entries + 1,
        entries
      )

    %{
      state
      | agent_state: %{
          state.agent_state
          | experience_replay_buffer: experience_replay_buffer,
            experience_replay_buffer_index: experience_replay_buffer_index,
            persisted_experience_replay_buffer_entries:
              persisted_experience_replay_buffer_entries,
            total_reward: state.agent_state.total_reward + reward
        }
    }
  end

  @spec optimize_model(state :: state) :: state
  defn optimize_model(state) do
    %{persisted_experience_replay_buffer_entries: persisted_experience_replay_buffer_entries} =
      state.agent_state

    if persisted_experience_replay_buffer_entries > @batch_size and
         rem(persisted_experience_replay_buffer_entries, @train_every_steps) == 0 do
      do_optimize_model(state)
    else
      state
    end
  end

  defnp do_optimize_model(state) do
    %{
      agent_state: %{
        q_policy: q_policy,
        q_policy_optimizer_state: q_policy_optimizer_state,
        policy_predict_fn: policy_predict_fn,
        optimizer_update_fn: optimizer_update_fn,
        state_vector_size: state_vector_size
      },
      random_key: random_key
    } = state

    {batch, random_key} =
      Nx.Random.choice(random_key, slice_experience_replay_buffer(state.agent_state),
        samples: @batch_size,
        replace: false,
        axis: 0
      )

    state_batch = Nx.slice_along_axis(batch, 0, state_vector_size, axis: 1)
    action_batch = Nx.slice_along_axis(batch, state_vector_size, 1, axis: 1)
    reward_batch = Nx.slice_along_axis(batch, state_vector_size + 1, 1, axis: 1)
    is_terminal_batch = Nx.slice_along_axis(batch, state_vector_size + 2, 1, axis: 1)

    next_state_batch =
      Nx.slice_along_axis(batch, state_vector_size + 3, state_vector_size, axis: 1)

    non_final_mask = not is_terminal_batch

    {loss, gradient} =
      value_and_grad(q_policy, fn q_policy ->
        action_idx = Nx.as_type(action_batch, :s64)

        %{shape: {m, 1}} =
          state_action_values =
          q_policy
          |> policy_predict_fn.(%{"state" => state_batch})
          |> Nx.take_along_axis(action_idx, axis: 1)

        %{shape: {n, 1}} =
          expected_state_action_values =
          q_policy
          |> policy_predict_fn.(%{"state" => next_state_batch})
          |> Nx.multiply(@gamma)
          |> Nx.multiply(non_final_mask)
          |> Nx.add(reward_batch)
          |> Nx.reduce_max(axes: [-1], keep_axes: true)

        case {m, n} do
          {m, n} when m != n ->
            raise "shape mismatch for batch values"

          _ ->
            1
        end

        huber_loss(expected_state_action_values, state_action_values)
      end)

    {scaled_updates, optimizer_state} =
      optimizer_update_fn.(gradient, q_policy_optimizer_state, q_policy)

    q_policy = Axon.Updates.apply_updates(q_policy, scaled_updates)

    %{
      state
      | agent_state: %{
          state.agent_state
          | q_policy: q_policy,
            q_policy_optimizer_state: optimizer_state,
            loss: loss
        },
        random_key: random_key
    }
  end

  defnp slice_experience_replay_buffer(state) do
    %{
      experience_replay_buffer: experience_replay_buffer,
      persisted_experience_replay_buffer_entries: entries
    } = state

    if entries < @experience_replay_buffer_num_entries do
      t = Nx.iota({@experience_replay_buffer_num_entries})
      idx = Nx.select(t <= entries, t, 0)
      Nx.take(experience_replay_buffer, idx)
    else
      experience_replay_buffer
    end
  end

  defnp huber_loss(y_true, y_pred, opts \\ [delta: 1.0]) do
    delta = opts[:delta]

    abs_diff = Nx.abs(y_pred - y_true)

    (abs_diff <= delta)
    |> Nx.select(0.5 * abs_diff ** 2, delta * abs_diff - 0.5 * delta ** 2)
    |> Nx.mean()
  end

  def dqn(num_observations, num_actions) do
    Axon.input("state", shape: {nil, num_observations})
    |> Axon.dense(64, activation: :relu)
    |> Axon.dense(64, activation: :relu)
    |> Axon.dense(num_actions)
  end
end

7 * 64 * 64 * 4
