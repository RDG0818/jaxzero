import jax
import jax.numpy as jnp
from jax import lax
import optax
import numpy as np
from typing import Any
from jaxzero.config import MAZeroConfig
from jaxzero.model.transforms import h, phi, phi_inv
from jaxzero.reanalyze import BatchData


def awpo_sharp_loss(
    policy_logits: jnp.ndarray,   # (B, N, A)
    sampled_actions: jnp.ndarray, # (B, K, N)
    visit_counts: jnp.ndarray,    # (B, K) normalized (sum=1)
    advantages: jnp.ndarray,      # (B, K)
    masks: jnp.ndarray,           # (B, K)
    alpha: float,
) -> jnp.ndarray:                 # (B,)
    N = policy_logits.shape[1]
    log_probs = jax.nn.log_softmax(policy_logits, axis=-1)  # (B, N, A)

    def gather_joint_logprob(log_p, actions):
        # log_p: (N, A), actions: (K, N) -> (K,)
        # For each k and n, get log_p[n, actions[k, n]], then sum over n
        return jnp.sum(log_p[jnp.arange(N)[None, :], actions], axis=-1)

    joint_log_probs = jax.vmap(gather_joint_logprob)(log_probs, sampled_actions)  # (B, K)

    masked_adv_mean = (advantages * masks).sum(-1, keepdims=True) / (masks.sum(-1, keepdims=True) + 1e-8)
    adv_norm = advantages - masked_adv_mean
    adv_weights = jnp.exp(adv_norm / alpha)

    return -(joint_log_probs * visit_counts * adv_weights * masks).sum(axis=-1)


def categorical_cross_entropy(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Cross-entropy between target distribution and model logits, summed over support.

    Args:
        logits: (B, support_size) unnormalized logits
        targets: (B, support_size) target probabilities

    Returns:
        (B,) per-sample cross-entropy
    """
    return -(targets * jax.nn.log_softmax(logits, axis=-1)).sum(axis=-1)


def _batch_phi_h(vals: jnp.ndarray, S: int) -> jnp.ndarray:
    """Encode (B,) scalars to (B, 2S+1) support distributions via h-transform."""
    return phi(h(vals), S)


def make_update_fn(model, config: MAZeroConfig):
    """Create a JIT-able pure loss function (params, batch) -> scalar_loss."""
    S_v = config.value_support_size
    S_r = config.reward_support_size

    @jax.jit
    def update_fn(params: Any, batch: BatchData) -> jnp.ndarray:
        obs = jnp.array(batch.obs)                         # (B, U+1, N, obs_dim)
        actions = jnp.array(batch.actions)                 # (B, U, N)
        target_rewards = jnp.array(batch.target_rewards)   # (B, U)
        target_values = jnp.array(batch.target_values)     # (B, U+1)
        target_policies = jnp.array(batch.target_policies)  # (B, U+1, K)
        target_qvalues = jnp.array(batch.target_qvalues)    # (B, U+1, K)
        target_masks = jnp.array(batch.target_masks)        # (B, U+1, K)
        sampled_acts = jnp.array(batch.sampled_actions)     # (B, U+1, K, N)
        weights = jnp.array(batch.weights)                  # (B,)

        U = config.unroll_steps

        # Initial inference: obs[:, 0] shape (B, N, obs_dim)
        out0 = model.apply(params, obs[:, 0])
        target_v0 = _batch_phi_h(target_values[:, 0], S_v)  # (B, 2*S_v+1)

        value_loss = categorical_cross_entropy(out0.value_logits, target_v0)   # (B,)
        reward_loss = jnp.zeros(obs.shape[0])                                   # (B,)
        policy_loss = awpo_sharp_loss(
            out0.policy_logits,      # (B, N, A)
            sampled_acts[:, 0],      # (B, K, N)
            target_policies[:, 0],   # (B, K)
            target_qvalues[:, 0],    # (B, K)
            target_masks[:, 0],      # (B, K)
            config.awpo_alpha,
        )  # (B,)

        hidden = out0.hidden_state  # (B, N, D)

        for k in range(1, U + 1):
            # Half-gradient trick: scale gradient contribution from hidden state
            hidden = 0.5 * hidden + 0.5 * lax.stop_gradient(hidden)
            out_k = model.apply(
                params, hidden, actions[:, k - 1],
                method=model.recurrent_inference,
            )

            target_r = _batch_phi_h(target_rewards[:, k - 1], S_r)  # (B, 2*S_r+1)
            target_v = _batch_phi_h(target_values[:, k], S_v)        # (B, 2*S_v+1)

            reward_loss = reward_loss + categorical_cross_entropy(out_k.reward_logits, target_r)
            value_loss = value_loss + categorical_cross_entropy(out_k.value_logits, target_v)
            policy_loss = policy_loss + awpo_sharp_loss(
                out_k.policy_logits,     # (B, N, A)
                sampled_acts[:, k],      # (B, K, N)
                target_policies[:, k],   # (B, K)
                target_qvalues[:, k],    # (B, K)
                target_masks[:, k],      # (B, K)
                config.awpo_alpha,
            )

            hidden = out_k.hidden_state

        total = (
            config.reward_loss_coeff * reward_loss
            + config.value_loss_coeff * value_loss
            + config.policy_loss_coeff * policy_loss
        )
        return (weights * total).mean() / U

    return update_fn


def _store_search_stats(game, result, idx, config):
    """Pad ctree results to fixed K and store in game."""
    visits = result.sampled_visit_counts[idx]
    actions_pool = result.sampled_actions[idx]
    K_full = config.sampled_action_times
    K_actual = len(visits)
    pad = K_full - K_actual
    if pad > 0:
        N_agents = actions_pool.shape[1]
        actions_padded = np.concatenate(
            [actions_pool, np.zeros((pad, N_agents), dtype=actions_pool.dtype)], axis=0
        )
        visits_padded = np.concatenate([visits, np.zeros(pad, dtype=visits.dtype)])
        qvals_padded = np.concatenate(
            [result.sampled_qvalues[idx], np.zeros(pad, dtype=np.float32)]
        )
        mask_padded = np.concatenate(
            [np.ones(K_actual, dtype=bool), np.zeros(pad, dtype=bool)]
        )
    else:
        actions_padded = actions_pool
        visits_padded = visits
        qvals_padded = result.sampled_qvalues[idx]
        mask_padded = np.ones(K_full, dtype=bool)
    game.store_search_stats(
        sampled_actions=actions_padded,
        visit_counts=visits_padded.astype(np.float32) / visits_padded.sum(),
        qvalues=qvals_padded.astype(np.float32),
        mask=mask_padded,
    )


def collect_episodes_parallel(envs, mcts, params, config: MAZeroConfig, rng_key):
    """Collect one episode per env using constant B-batch MCTS.

    Batch size stays fixed at B=len(envs) every step — JAX never recompiles
    mid-collection. Done envs are skipped in result storage but kept in the
    batch so the shape doesn't change.
    """
    from jaxzero.game import GameHistory
    import jax.random as jr

    B = len(envs)
    np_rng = np.random.default_rng(int(jax.random.randint(rng_key, (), 0, 2**31 - 1)))
    raw_obs_dim = config.obs_size // config.stacked_observations

    env_rngs = jr.split(rng_key, B + 1)
    step_rng = env_rngs[0]

    obss, states = [], []
    games = []
    for i, env in enumerate(envs):
        obs, state = env.reset(env_rngs[i + 1])
        obss.append(obs)
        states.append(state)
        game = GameHistory(
            num_agents=config.num_agents,
            obs_dim=raw_obs_dim,
            action_space_size=config.action_space_size,
            stacked_observations=config.stacked_observations,
        )
        game.store_observation(obs[:, :raw_obs_dim])
        games.append(game)

    dones = [False] * B

    for _ in range(config.max_episode_steps):
        if all(dones):
            break

        # Always full B batch — constant shape, no JAX recompilation
        obs_batch = np.stack(obss)
        legal_batch = np.stack([envs[i].get_legal_actions(states[i]) for i in range(B)])
        result = mcts.search(params, obs_batch, legal_batch, np_rng)

        step_rng, sub = jr.split(step_rng)
        sub_rngs = jr.split(sub, B)

        for i in range(B):
            if dones[i]:
                continue
            visits = result.sampled_visit_counts[i]
            actions_pool = result.sampled_actions[i]
            probs = visits / visits.sum()
            action = actions_pool[np_rng.choice(len(probs), p=probs)]

            obs_next, state, reward, done, _ = envs[i].step(sub_rngs[i], states[i], action)

            games[i].store_observation(obs_next[:, :raw_obs_dim])
            games[i].store_action(action)
            games[i].store_reward(reward)
            games[i].store_legal_actions(legal_batch[i])
            games[i].store_root_value(float(result.root_value[i]))
            games[i].store_pred_value(0.0)
            _store_search_stats(games[i], result, i, config)

            obss[i] = obs_next
            states[i] = state
            if done:
                dones[i] = True

    return games


def collect_episode(env, mcts, params, config: MAZeroConfig, rng_key):
    """Collect a single episode. Used in tests."""
    return collect_episodes_parallel([env], mcts, params, config, rng_key)[0]


def train(config: MAZeroConfig, env_fn):
    """Training loop. env_fn is a callable () -> EnvWrapper used to create parallel envs."""
    import jax.random as jr
    from jaxzero.model.networks import MAMuZeroNet
    from jaxzero.replay_buffer import PrioritizedReplayBuffer
    from jaxzero.reanalyze import ReanalyzeWorker

    net = MAMuZeroNet(config=config)
    rng = jr.PRNGKey(config.seed)
    obs_init = jnp.ones((1, config.num_agents, config.obs_size))
    rng, init_rng = jr.split(rng)
    params = net.init(init_rng, obs_init)
    params = jax.device_put(params)

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.learning_rate, eps=config.adam_eps),
    )
    opt_state = optimizer.init(params)

    replay_buffer = PrioritizedReplayBuffer(config)
    reanalyze_worker = ReanalyzeWorker(config=config, model=net)
    update_fn = make_update_fn(net, config)

    # Create persistent pool of parallel envs and one MCTS instance (JIT compiled once)
    envs = [env_fn() for _ in range(config.num_envs_parallel)]
    from jaxzero.mcts.sampled_mcts import SampledMCTS
    mcts = SampledMCTS(config=config, model=net)

    # Warmup JIT: compile initial + recurrent inference at the collection batch size
    _B = config.num_envs_parallel
    _obs_dummy = jnp.ones((_B, config.num_agents, config.obs_size))
    _legal_dummy = np.ones((_B, config.num_agents, config.action_space_size), dtype=bool)
    print("Compiling JAX model (one-time)...")
    mcts.search(params, np.array(_obs_dummy), _legal_dummy, np.random.default_rng(0))
    print("Compilation done.")

    beta_fn = lambda step: min(
        1.0,
        config.priority_beta_start + (1.0 - config.priority_beta_start) * step / config.training_steps,
    )

    step = 0
    collection_round = 0
    recent_returns = []
    while step < config.training_steps:
        rng, ep_rng = jr.split(rng)
        games = collect_episodes_parallel(envs, mcts, params, config, ep_rng)
        for game in games:
            replay_buffer.add(game)
            recent_returns.append(sum(game.rewards))
            if len(recent_returns) > 100:
                recent_returns.pop(0)

        if not replay_buffer.can_sample(config.batch_size):
            collection_round += 1
            if collection_round % 5 == 0:
                print(f"[filling buffer] round {collection_round}, size={replay_buffer.size}")
            continue

        for _ in range(config.updates_per_collection):
            beta = beta_fn(step)
            ctx = replay_buffer.prepare_batch_context(config.batch_size, beta)
            batch = reanalyze_worker.make_batch(ctx, params)

            loss, grads = jax.value_and_grad(update_fn)(params, batch)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            if step % config.log_interval == 0:
                mean_ret = np.mean(recent_returns) if recent_returns else float("nan")
                print(f"Step {step}: loss={float(loss):.4f} | ep_return={mean_ret:.2f}")

            step += 1
            if step >= config.training_steps:
                break

    return params
