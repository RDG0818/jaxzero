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


def collect_episode(env, params, model, config: MAZeroConfig, rng_key):
    """Collect a single episode. Used in tests; training uses collect_episodes_parallel."""
    games = collect_episodes_parallel([env], params, model, config, rng_key)
    return games[0]


def collect_episodes_parallel(envs, params, model, config: MAZeroConfig, rng_key):
    """Collect one episode per env in parallel using B-batched MCTS inference.

    Each simulation step issues one model call of shape (B, ...) across all
    active environments simultaneously, maximising GPU utilisation.

    Args:
        envs: list of EnvWrapper instances (len = B).
        params: current model parameters.
        model: MAMuZeroNet.
        config: MAZeroConfig.
        rng_key: JAX PRNGKey.

    Returns:
        list of GameHistory, one per env.
    """
    from jaxzero.game import GameHistory
    from jaxzero.mcts.sampled_mcts import SampledMCTS
    import jax.random as jr

    n_envs = len(envs)
    mcts = SampledMCTS(config=config, model=model)
    np_rng = np.random.default_rng(int(jax.random.randint(rng_key, (), 0, 2**31 - 1)))
    raw_obs_dim = config.obs_size // config.stacked_observations

    games = [
        GameHistory(
            num_agents=config.num_agents,
            obs_dim=raw_obs_dim,
            action_space_size=config.action_space_size,
            stacked_observations=config.stacked_observations,
        )
        for _ in range(n_envs)
    ]

    env_rngs = jr.split(rng_key, n_envs + 1)
    step_rng = env_rngs[0]

    obss, states = [], []
    for i, env in enumerate(envs):
        obs, state = env.reset(env_rngs[i + 1])
        obss.append(obs)
        states.append(state)
        games[i].store_observation(obs[:, :raw_obs_dim])

    dones = [False] * n_envs

    for _ in range(config.max_episode_steps):
        if all(dones):
            break

        active = [i for i in range(n_envs) if not dones[i]]
        obs_batch = np.stack([obss[i] for i in active])
        legal_batch = np.stack([envs[i].get_legal_actions(states[i]) for i in active])

        # One MCTS search with B = len(active): model calls batch all active envs
        result = mcts.search(params, obs_batch, legal_batch, np_rng)

        step_rng, sub = jr.split(step_rng)
        sub_rngs = jr.split(sub, n_envs)

        for idx, i in enumerate(active):
            visits = result.sampled_visit_counts[idx]
            actions_pool = result.sampled_actions[idx]
            probs = visits / visits.sum()
            chosen = np_rng.choice(len(probs), p=probs)
            action = actions_pool[chosen]

            obs_next, state, reward, done, _ = envs[i].step(sub_rngs[i], states[i], action)

            games[i].store_observation(obs_next[:, :raw_obs_dim])
            games[i].store_action(action)
            games[i].store_reward(reward)
            games[i].store_legal_actions(legal_batch[idx])
            games[i].store_root_value(float(result.root_value[idx]))
            games[i].store_pred_value(0.0)
            games[i].store_search_stats(
                sampled_actions=result.sampled_actions[idx],
                visit_counts=visits.astype(np.float32) / visits.sum(),
                qvalues=result.sampled_qvalues[idx].astype(np.float32),
                mask=np.ones(len(visits), dtype=bool),
            )

            obss[i] = obs_next
            states[i] = state
            if done:
                dones[i] = True

    return games


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

    # Create persistent pool of parallel envs
    envs = [env_fn() for _ in range(config.num_envs_parallel)]

    beta_fn = lambda step: min(
        1.0,
        config.priority_beta_start + (1.0 - config.priority_beta_start) * step / config.training_steps,
    )

    step = 0
    while step < config.training_steps:
        rng, ep_rng = jr.split(rng)
        # Collect num_envs_parallel episodes simultaneously with batched MCTS
        games = collect_episodes_parallel(envs, params, net, config, ep_rng)
        for game in games:
            replay_buffer.add(game)

        if replay_buffer.can_sample(config.batch_size):
            beta = beta_fn(step)
            ctx = replay_buffer.prepare_batch_context(config.batch_size, beta)
            batch = reanalyze_worker.make_batch(ctx, params)

            loss, grads = jax.value_and_grad(update_fn)(params, batch)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            if step % config.log_interval == 0:
                print(f"Step {step}: loss={float(loss):.4f}")

            step += 1

    return params
