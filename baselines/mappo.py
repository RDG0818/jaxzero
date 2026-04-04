# baselines/mappo.py
"""
Multi-Agent PPO (MAPPO) for cooperative multi-agent environments.

Extends IPPO with a centralized critic (CTDE — centralized training,
decentralized execution). The critic receives the global state (all agents'
observations concatenated) and outputs a single team value estimate, which
is used to compute shared GAE advantages.

Key differences from IPPO:
  - Separate actor and critic networks with separate optimizers.
  - Critic input: global_state = concat(obs_agent_0, ..., obs_agent_N-1), shape (B, N*obs_size).
  - Single shared advantage signal for all agents per timestep.

Pure JAX implementation — no Ray, no replay buffer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
import jaxmarl
from flax.training.train_state import TrainState
from jaxmarl.wrappers.baselines import LogWrapper

from baselines.networks import ActorCritic, CentralizedCritic
from utils.logging_utils import logger


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MAPPOConfig:
    env_name: str = "MPE_simple_spread_v3"
    num_agents: int = 3
    num_envs: int = 32
    num_steps: int = 128
    num_minibatches: int = 4
    update_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    actor_lr: float = 2.5e-4
    critic_lr: float = 1e-3
    max_grad_norm: float = 0.5
    total_timesteps: int = 10_000_000
    hidden_sizes: tuple = (64, 64)
    log_interval: int = 10
    anneal_lr: bool = True
    wandb_mode: str = "disabled"

    @property
    def minibatch_size(self) -> int:
        return self.num_envs * self.num_steps // self.num_minibatches

    @property
    def num_updates(self) -> int:
        return self.total_timesteps // (self.num_envs * self.num_steps)


# ---------------------------------------------------------------------------
# Helpers (shared with IPPO)
# ---------------------------------------------------------------------------

def _obs_with_agent_id(obs: jnp.ndarray, num_agents: int) -> jnp.ndarray:
    """(B, N, obs_size) → (B, N, obs_size + N)."""
    B, N, _ = obs.shape
    ids = jnp.broadcast_to(jnp.eye(num_agents, dtype=obs.dtype)[None], (B, N, N))
    return jnp.concatenate([obs, ids], axis=-1)


def _global_state(obs: jnp.ndarray) -> jnp.ndarray:
    """(B, N, obs_size) → (B, N*obs_size)."""
    B, N, O = obs.shape
    return obs.reshape(B, N * O)


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

def make_rollout_fn(env, actor: ActorCritic, critic: CentralizedCritic, config: MAPPOConfig):
    num_agents = config.num_agents

    def _step(carry, _):
        actor_state, critic_state, obs, env_state, rng = carry
        rng, key_action, key_env = jax.random.split(rng, 3)

        obs_aug = _obs_with_agent_id(obs, num_agents)    # (B, N, obs+N)
        global_st = _global_state(obs)                   # (B, N*obs)

        logits, _ = actor.apply(actor_state.params, obs_aug)  # (B, N, A)
        values = critic.apply(critic_state.params, global_st)  # (B,)

        keys_per_agent = jax.random.split(key_action, num_agents)
        actions = jax.vmap(
            lambda l, k: jax.random.categorical(k, l), in_axes=(0, 0)
        )(jnp.moveaxis(logits, 1, 0), keys_per_agent)  # (N, B)
        actions = jnp.moveaxis(actions, 0, 1)  # (B, N)

        log_probs = jax.nn.log_softmax(logits)
        action_log_probs = jnp.take_along_axis(
            log_probs, actions[..., None], axis=-1
        ).squeeze(-1)  # (B, N)

        action_dict = {a: actions[:, i] for i, a in enumerate(env.agents)}
        key_envs = jax.random.split(key_env, config.num_envs)
        next_obs_dict, next_env_state, rewards_dict, dones_dict, info = jax.vmap(env.step)(
            key_envs, env_state, action_dict
        )

        next_obs = jnp.stack(
            [next_obs_dict[a].astype(jnp.float32) for a in env.agents], axis=1
        )
        team_reward = jnp.stack([rewards_dict[a] for a in env.agents], axis=1).sum(axis=1)
        done = jnp.stack([dones_dict[a] for a in env.agents], axis=1).all(axis=1)

        transition = (obs, actions, action_log_probs, values, team_reward, done)
        return (actor_state, critic_state, next_obs, next_env_state, rng), (transition, info)

    return _step


# ---------------------------------------------------------------------------
# GAE (scalar value → shared advantage)
# ---------------------------------------------------------------------------

def compute_gae_shared(
    rewards: jnp.ndarray,  # (T, B)
    values: jnp.ndarray,   # (T+1, B)
    dones: jnp.ndarray,    # (T, B)
    gamma: float,
    gae_lambda: float,
) -> tuple:
    """Returns advantages and targets, both shape (T, B)."""
    def _step(gae, inputs):
        reward, value, next_value, done = inputs
        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * gae_lambda * (1 - done) * gae
        return gae, gae

    _, advantages = jax.lax.scan(
        _step,
        jnp.zeros_like(values[-1]),
        (rewards, values[:-1], values[1:], dones),
        reverse=True,
    )
    targets = advantages + values[:-1]
    return advantages, targets


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def actor_loss(actor_params, actor, minibatch, config: MAPPOConfig):
    obs, actions, old_log_probs, advantages = minibatch
    logits, _ = actor.apply(actor_params, obs)
    log_probs = jax.nn.log_softmax(logits)
    new_log_probs = jnp.take_along_axis(
        log_probs, actions[..., None], axis=-1
    ).squeeze(-1)
    ratio = jnp.exp(new_log_probs - old_log_probs)
    adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    # Broadcast scalar advantage to (B, N)
    adv_n = jnp.broadcast_to(adv_norm[..., None], ratio.shape)
    pg1 = -adv_n * ratio
    pg2 = -adv_n * jnp.clip(ratio, 1 - config.clip_eps, 1 + config.clip_eps)
    pg_loss = jnp.maximum(pg1, pg2).mean()
    entropy = -jnp.sum(jax.nn.softmax(logits) * log_probs, axis=-1).mean()
    total = pg_loss - config.ent_coef * entropy
    return total, {"pg_loss": pg_loss, "entropy": entropy}


def critic_loss(critic_params, critic, minibatch):
    global_state, targets = minibatch
    values = critic.apply(critic_params, global_state)
    loss = 0.5 * jnp.square(values - targets).mean()
    return loss, {"vf_loss": loss}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_training_loop(config: MAPPOConfig, rng_seed: int = 0):
    rng = jax.random.PRNGKey(rng_seed)

    _env = jaxmarl.make(config.env_name, num_agents=config.num_agents)
    env = LogWrapper(_env)

    obs_size = _env.observation_space(_env.agents[0]).shape[0]
    action_size = _env.action_space(_env.agents[0]).n
    aug_obs_size = obs_size + config.num_agents
    global_state_size = obs_size * config.num_agents

    actor = ActorCritic(action_size=action_size, hidden_sizes=config.hidden_sizes)
    critic = CentralizedCritic(hidden_sizes=config.hidden_sizes)

    num_updates = config.num_updates

    if config.anneal_lr:
        actor_lr = optax.linear_schedule(config.actor_lr, 0.0, num_updates)
        critic_lr = optax.linear_schedule(config.critic_lr, 0.0, num_updates)
    else:
        actor_lr, critic_lr = config.actor_lr, config.critic_lr

    actor_opt = optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(actor_lr, eps=1e-5))
    critic_opt = optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(critic_lr, eps=1e-5))

    rng, a_key, c_key = jax.random.split(rng, 3)
    actor_params = actor.init(a_key, jnp.zeros((1, config.num_agents, aug_obs_size)))
    critic_params = critic.init(c_key, jnp.zeros((1, global_state_size)))

    actor_state = TrainState.create(apply_fn=actor.apply, params=actor_params, tx=actor_opt)
    critic_state = TrainState.create(apply_fn=critic.apply, params=critic_params, tx=critic_opt)

    rng, *reset_keys = jax.random.split(rng, config.num_envs + 1)
    reset_keys = jnp.stack(reset_keys)
    obs_dict, env_state = jax.vmap(env.reset)(reset_keys)
    obs = jnp.stack([obs_dict[a].astype(jnp.float32) for a in env.agents], axis=1)

    rollout_fn = make_rollout_fn(env, actor, critic, config)
    actor_loss_grad = jax.value_and_grad(actor_loss, has_aux=True)
    critic_loss_grad = jax.value_and_grad(critic_loss, has_aux=True)

    @jax.jit
    def collect_rollout(carry):
        return jax.lax.scan(rollout_fn, carry, None, length=config.num_steps)

    @jax.jit
    def update_epoch(actor_state, critic_state, trajectories, rng):
        obs_t, actions_t, log_probs_t, advantages_t, targets_t, global_state_t = trajectories

        def _flat(x):
            return x.reshape(-1, *x.shape[2:])

        obs_f    = _flat(obs_t)            # (T*B, N, obs+N)
        acts_f   = _flat(actions_t)        # (T*B, N)
        lp_f     = _flat(log_probs_t)      # (T*B, N)
        adv_f    = _flat(advantages_t)     # (T*B,)
        tgt_f    = _flat(targets_t)        # (T*B,)
        gs_f     = _flat(global_state_t)   # (T*B, N*obs)

        n = obs_f.shape[0]
        rng, perm_key = jax.random.split(rng)
        perm = jax.random.permutation(perm_key, n)
        mb_size = config.minibatch_size

        def _update_mb(states, mb_idx):
            a_state, c_state = states
            idx = perm[mb_idx * mb_size:(mb_idx + 1) * mb_size]
            a_mb = (obs_f[idx], acts_f[idx], lp_f[idx], adv_f[idx])
            c_mb = (gs_f[idx], tgt_f[idx])
            (a_loss, a_aux), a_grads = actor_loss_grad(a_state.params, actor, a_mb, config)
            (c_loss, c_aux), c_grads = critic_loss_grad(c_state.params, critic, c_mb)
            a_state = a_state.apply_gradients(grads=a_grads)
            c_state = c_state.apply_gradients(grads=c_grads)
            return (a_state, c_state), {**a_aux, **c_aux, "total": a_loss + config.vf_coef * c_loss}

        (actor_state, critic_state), metrics = jax.lax.scan(
            _update_mb, (actor_state, critic_state), jnp.arange(config.num_minibatches)
        )
        return actor_state, critic_state, {k: v.mean() for k, v in metrics.items()}, rng

    logger.info(
        f"MAPPO | env={config.env_name} N={config.num_agents} "
        f"B={config.num_envs} T={config.num_steps} updates={num_updates}"
    )

    ep_returns = []
    t_start = time.monotonic()

    for update in range(num_updates):
        rng, rollout_rng = jax.random.split(rng)
        carry = (actor_state, critic_state, obs, env_state, rollout_rng)
        (actor_state, critic_state, obs, env_state, _), (transitions, infos) = collect_rollout(carry)

        obs_t, actions_t, log_probs_t, values_t, rewards_t, dones_t = transitions

        # Bootstrap
        global_st_boot = _global_state(obs)
        bootstrap_v = critic.apply(critic_state.params, global_st_boot)  # (B,)
        all_values = jnp.concatenate([values_t, bootstrap_v[None]], axis=0)  # (T+1, B)

        advantages, targets = compute_gae_shared(
            rewards_t, all_values, dones_t, config.gamma, config.gae_lambda
        )

        obs_aug_t = jax.vmap(_obs_with_agent_id, in_axes=(0, None))(obs_t, config.num_agents)
        global_state_t = jax.vmap(_global_state)(obs_t)  # (T, B, N*obs)

        trajectories = (obs_aug_t, actions_t, log_probs_t, advantages, targets, global_state_t)

        for _ in range(config.update_epochs):
            rng, epoch_rng = jax.random.split(rng)
            actor_state, critic_state, metrics, _ = update_epoch(
                actor_state, critic_state, trajectories, epoch_rng
            )

        if "returned_episode_returns" in infos:
            ret_vals = infos["returned_episode_returns"]
            if isinstance(ret_vals, dict):
                team_ret = jnp.stack(list(ret_vals.values()), axis=-1).sum(-1)
            else:
                team_ret = ret_vals
            mask = infos.get("returned_episode", jnp.ones_like(team_ret, dtype=bool))
            if isinstance(mask, dict):
                mask = jnp.stack(list(mask.values()), axis=-1).any(-1)
            finished = team_ret[mask]
            if finished.size > 0:
                ep_returns.extend(finished.tolist())

        if (update + 1) % config.log_interval == 0:
            elapsed = time.monotonic() - t_start
            sps = (update + 1) * config.num_envs * config.num_steps / elapsed
            mean_ret = float(jnp.mean(jnp.array(ep_returns[-100:]))) if ep_returns else 0.0
            logger.info(
                f"MAPPO update {update + 1}/{num_updates} | "
                f"total={float(metrics['total']):.4f} | "
                f"pg={float(metrics['pg_loss']):.4f} vf={float(metrics['vf_loss']):.4f} "
                f"ent={float(metrics['entropy']):.4f} | "
                f"return={mean_ret:.2f} | SPS={sps:.0f}"
            )

    logger.info("MAPPO training complete.")
    return actor_state, critic_state
