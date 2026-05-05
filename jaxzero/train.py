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
    clip: float = 3.0,
) -> jnp.ndarray:                 # (B,)
    N = policy_logits.shape[1]
    log_probs = jax.nn.log_softmax(policy_logits, axis=-1)  # (B, N, A)

    def gather_joint_logprob(log_p, actions):
        # log_p: (N, A), actions: (K, N) -> (K,)
        # For each k and n, get log_p[n, actions[k, n]], then sum over n
        return jnp.sum(log_p[jnp.arange(N)[None, :], actions], axis=-1)

    joint_log_probs = jax.vmap(gather_joint_logprob)(log_probs, sampled_actions)  # (B, K)

    n_valid = masks.sum(-1, keepdims=True) + 1e-8
    masked_adv_mean = (advantages * masks).sum(-1, keepdims=True) / n_valid
    adv_centered = (advantages - masked_adv_mean) * masks  # zero invalid → no exp(huge) NaN
    masked_adv_var = (adv_centered ** 2 * masks).sum(-1, keepdims=True) / n_valid
    masked_adv_std = jnp.sqrt(masked_adv_var + 1e-10)
    adv_norm = adv_centered / (masked_adv_std + 1e-5)
    adv_weights = jnp.exp(adv_norm / alpha)
    adv_weights = jnp.clip(adv_weights, -clip, clip)

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


def cosine_similarity_loss(p: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
    """Negative cosine similarity between p and z.
    p: (B, D)
    z: (B, D)
    """
    p_norm = jnp.sqrt(jnp.sum(p**2, axis=-1, keepdims=True) + 1e-8)
    z_norm = jnp.sqrt(jnp.sum(z**2, axis=-1, keepdims=True) + 1e-8)
    p = p / p_norm
    z = z / z_norm
    return -(p * z).sum(axis=-1)


def _batch_phi_h(vals: jnp.ndarray, S: int) -> jnp.ndarray:
    """Encode (B,) scalars to (B, 2S+1) support distributions via h-transform."""
    return phi(h(vals), S)


def make_update_fn(model, config: MAZeroConfig):
    """Create a pure (params, batch) -> (loss, grads, aux) update function."""
    S_v = config.value_support_size
    S_r = config.reward_support_size

    def _loss_fn(params: Any, batch: BatchData):
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

        out0 = model.apply(params, obs[:, 0])
        target_v0 = _batch_phi_h(target_values[:, 0], S_v)

        value_loss = categorical_cross_entropy(out0.value_logits, target_v0)
        reward_loss = jnp.zeros(obs.shape[0])
        policy_loss = awpo_sharp_loss(
            out0.policy_logits,
            sampled_acts[:, 0],
            target_policies[:, 0],
            target_qvalues[:, 0],
            target_masks[:, 0],
            config.awpo_alpha,
            config.adv_clip,
        )
        consistency_loss = jnp.zeros(obs.shape[0])

        hidden = out0.hidden_state

        for k in range(1, U + 1):
            hidden = 0.5 * hidden + 0.5 * lax.stop_gradient(hidden)
            out_k = model.apply(
                params, hidden, actions[:, k - 1],
                method=model.recurrent_inference,
            )

            target_r = _batch_phi_h(target_rewards[:, k - 1], S_r)
            target_v = _batch_phi_h(target_values[:, k], S_v)

            reward_loss = reward_loss + categorical_cross_entropy(out_k.reward_logits, target_r)
            value_loss = value_loss + categorical_cross_entropy(out_k.value_logits, target_v)
            policy_loss = policy_loss + awpo_sharp_loss(
                out_k.policy_logits,
                sampled_acts[:, k],
                target_policies[:, k],
                target_qvalues[:, k],
                target_masks[:, k],
                config.awpo_alpha,
                config.adv_clip,
            )

            if config.consistency_coeff > 0:
                # Online: project + predict from unrolled state
                dynamic_proj = model.apply(params, out_k.hidden_state, method=model.project_online)
                # Target: project only from re-encoded actual observation
                target_out = model.apply(params, obs[:, k])
                represet_proj = model.apply(params, target_out.hidden_state, method=model.project_target)
                consistency_loss = consistency_loss + cosine_similarity_loss(
                    dynamic_proj, lax.stop_gradient(represet_proj)
                )

            hidden = out_k.hidden_state

        r_term = config.reward_loss_coeff * reward_loss
        v_term = config.value_loss_coeff * value_loss
        p_term = config.policy_loss_coeff * policy_loss
        c_term = config.consistency_coeff * consistency_loss
        total = r_term + v_term + p_term + c_term
        scalar = (weights * total).mean() / U
        aux = {
            "reward_loss": (weights * r_term).mean() / U,
            "value_loss": (weights * v_term).mean() / U,
            "policy_loss": (weights * p_term).mean() / U,
            "consistency_loss": (weights * c_term).mean() / U,
        }
        return scalar, (aux, total)

    _grad_fn = jax.jit(jax.value_and_grad(_loss_fn, has_aux=True))

    def update_fn(params: Any, batch: BatchData):
        (loss, (aux, priorities)), grads = _grad_fn(params, batch)
        return loss, grads, aux, priorities

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
    target_params = params  # Initialize target network

    while step < config.training_steps:
        rng, ep_rng = jr.split(rng)
        # Collection always uses latest params for best data
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
            # Periodically update target network
            if step % config.target_model_interval == 0:
                target_params = params

            beta = beta_fn(step)
            ctx = replay_buffer.prepare_batch_context(config.batch_size, beta)
            # Targets are computed using target_params to stabilize bootstrap
            batch = reanalyze_worker.make_batch(ctx, target_params)

            loss, grads, aux, priorities = update_fn(params, batch)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            # Update priorities in buffer for PER
            replay_buffer.update_priorities(batch.indices, priorities)

            if step % config.log_interval == 0:
                mean_ret = np.mean(recent_returns) if recent_returns else float("nan")
                print(
                    f"Step {step}: loss={float(loss):.4f}"
                    f" | r={float(aux['reward_loss']):.3f}"
                    f" v={float(aux['value_loss']):.3f}"
                    f" p={float(aux['policy_loss']):.3f}"
                    f" c={float(aux['consistency_loss']):.3f}"
                    f" | ep_return={mean_ret:.2f}"
                )

            step += 1
            if step >= config.training_steps:
                break

    return params
