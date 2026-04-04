# baselines/ippo.py
"""
Independent PPO (IPPO) for cooperative multi-agent environments.

Each agent independently runs PPO on its own observation, using the summed
team reward. All agents share one policy network; a one-hot agent ID is
appended to each observation to break symmetry.

Training loop:
  1. Collect T-step rollout across B parallel environments.
  2. Compute GAE advantages and discounted returns.
  3. Run K epochs of minibatch PPO updates.
  4. Repeat until total_timesteps reached.

Pure JAX implementation — no Ray, no replay buffer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import optax
import jaxmarl
from flax.training.train_state import TrainState
from jaxmarl.wrappers.baselines import LogWrapper

from baselines.networks import ActorCritic
from utils.logging_utils import logger


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class IPPOConfig:
    env_name: str = "MPE_simple_spread_v3"
    num_agents: int = 3
    num_envs: int = 32
    num_steps: int = 128        # rollout length T per update
    num_minibatches: int = 4
    update_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    lr: float = 2.5e-4
    max_grad_norm: float = 0.5
    total_timesteps: int = 10_000_000
    hidden_sizes: tuple = (64, 64)
    log_interval: int = 10      # log every N updates
    anneal_lr: bool = True
    wandb_mode: str = "disabled"

    @property
    def minibatch_size(self) -> int:
        return self.num_envs * self.num_steps // self.num_minibatches

    @property
    def num_updates(self) -> int:
        return self.total_timesteps // (self.num_envs * self.num_steps)


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

def _obs_with_agent_id(obs: jnp.ndarray, num_agents: int) -> jnp.ndarray:
    """Append one-hot agent ID to each agent's observation.

    obs: (B, N, obs_size)  →  (B, N, obs_size + N)
    """
    B, N, _ = obs.shape
    agent_ids = jnp.eye(num_agents, dtype=obs.dtype)      # (N, N)
    agent_ids = jnp.broadcast_to(agent_ids[None], (B, N, N))
    return jnp.concatenate([obs, agent_ids], axis=-1)


def make_rollout_fn(env, network: ActorCritic, config: IPPOConfig):
    """Returns a jax.lax.scan-compatible rollout step."""
    num_agents = config.num_agents

    def _step(carry, _):
        train_state, obs, env_state, rng = carry

        rng, key_action, key_env = jax.random.split(rng, 3)

        # obs: (B, N, obs_size) → augment with agent ID
        obs_aug = _obs_with_agent_id(obs, num_agents)  # (B, N, obs_size+N)

        logits, values = network.apply(train_state.params, obs_aug)
        # logits: (B, N, A),  values: (B, N)

        # Sample actions per agent
        def sample_agent(logits_n, key):
            return jax.random.categorical(key, logits_n)  # (B,)

        keys_per_agent = jax.random.split(key_action, num_agents)
        # logits transposed to (N, B, A) for vmap over agents
        actions = jax.vmap(sample_agent, in_axes=(0, 0))(
            jnp.moveaxis(logits, 1, 0), keys_per_agent
        )  # (N, B)
        actions = jnp.moveaxis(actions, 0, 1)  # (B, N)

        log_probs = jax.nn.log_softmax(logits)  # (B, N, A)
        action_log_probs = jnp.take_along_axis(
            log_probs, actions[..., None], axis=-1
        ).squeeze(-1)  # (B, N)

        # Step all envs
        action_dict = {
            agent: actions[:, i] for i, agent in enumerate(env.agents)
        }
        key_envs = jax.random.split(key_env, config.num_envs)
        next_obs_dict, next_env_state, rewards_dict, dones_dict, info = jax.vmap(
            env.step
        )(key_envs, env_state, action_dict)

        # Stack obs: (B, N, obs_size)
        next_obs = jnp.stack(
            [next_obs_dict[a].astype(jnp.float32) for a in env.agents], axis=1
        )
        # Team reward: sum over agents → (B,)
        team_reward = jnp.stack(
            [rewards_dict[a] for a in env.agents], axis=1
        ).sum(axis=1)
        # Done when all agents done: (B,)
        done = jnp.stack(
            [dones_dict[a] for a in env.agents], axis=1
        ).all(axis=1)

        transition = (obs, actions, action_log_probs, values, team_reward, done)
        return (train_state, next_obs, next_env_state, rng), (transition, info)

    return _step


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: jnp.ndarray,   # (T, B)
    values: jnp.ndarray,    # (T+1, B, N)  — last entry is bootstrap value
    dones: jnp.ndarray,     # (T, B)
    gamma: float,
    gae_lambda: float,
    num_agents: int,
) -> tuple:
    """Returns (advantages, targets) both shape (T, B, N)."""
    # Broadcast scalar reward/done to per-agent shape
    rewards_n = jnp.broadcast_to(rewards[..., None], rewards.shape + (num_agents,))  # (T, B, N)
    dones_n   = jnp.broadcast_to(dones[..., None],   dones.shape   + (num_agents,))  # (T, B, N)

    def _step(gae, inputs):
        reward, value, next_value, done = inputs
        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * gae_lambda * (1 - done) * gae
        return gae, gae

    _, advantages = jax.lax.scan(
        _step,
        jnp.zeros_like(values[-1]),
        (rewards_n, values[:-1], values[1:], dones_n),
        reverse=True,
    )
    targets = advantages + values[:-1]
    return advantages, targets


# ---------------------------------------------------------------------------
# PPO loss
# ---------------------------------------------------------------------------

def ppo_loss(params, network, minibatch, config: IPPOConfig):
    obs, actions, old_log_probs, advantages, targets = minibatch

    logits, values = network.apply(params, obs)

    log_probs = jax.nn.log_softmax(logits)
    new_log_probs = jnp.take_along_axis(
        log_probs, actions[..., None], axis=-1
    ).squeeze(-1)

    ratio = jnp.exp(new_log_probs - old_log_probs)
    adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    pg_loss1 = -adv_norm * ratio
    pg_loss2 = -adv_norm * jnp.clip(ratio, 1 - config.clip_eps, 1 + config.clip_eps)
    pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

    vf_loss = 0.5 * jnp.square(values - targets).mean()

    entropy = -jnp.sum(
        jax.nn.softmax(logits) * log_probs, axis=-1
    ).mean()

    total = pg_loss + config.vf_coef * vf_loss - config.ent_coef * entropy
    return total, {"pg_loss": pg_loss, "vf_loss": vf_loss, "entropy": entropy}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_training_loop(config: IPPOConfig, rng_seed: int = 0):
    """Full IPPO training loop."""
    rng = jax.random.PRNGKey(rng_seed)

    # Build env with logging wrapper
    _env = jaxmarl.make(config.env_name, num_agents=config.num_agents)
    env = LogWrapper(_env)

    obs_size = _env.observation_space(_env.agents[0]).shape[0]
    action_size = _env.action_space(_env.agents[0]).n
    aug_obs_size = obs_size + config.num_agents

    # Network and optimizer
    network = ActorCritic(
        action_size=action_size, hidden_sizes=config.hidden_sizes
    )

    if config.anneal_lr:
        lr_schedule = optax.linear_schedule(
            init_value=config.lr,
            end_value=0.0,
            transition_steps=config.num_updates,
        )
    else:
        lr_schedule = config.lr

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(lr_schedule, eps=1e-5),
    )

    rng, init_rng = jax.random.split(rng)
    dummy_obs = jnp.zeros((1, config.num_agents, aug_obs_size))
    params = network.init(init_rng, dummy_obs)
    train_state = TrainState.create(apply_fn=network.apply, params=params, tx=optimizer)

    # Reset all envs
    rng, *reset_keys = jax.random.split(rng, config.num_envs + 1)
    reset_keys = jnp.stack(reset_keys)
    obs_dict, env_state = jax.vmap(env.reset)(reset_keys)
    obs = jnp.stack([obs_dict[a].astype(jnp.float32) for a in env.agents], axis=1)  # (B, N, obs_size)

    rollout_fn = make_rollout_fn(env, network, config)
    loss_and_grad = jax.value_and_grad(ppo_loss, has_aux=True)

    @jax.jit
    def collect_rollout(carry):
        return jax.lax.scan(rollout_fn, carry, None, length=config.num_steps)

    @jax.jit
    def update_epoch(train_state, trajectories, rng):
        obs_t, actions_t, log_probs_t, advantages_t, targets_t = trajectories
        # Flatten (T, B, ...) → (T*B, ...)
        def _flat(x):
            return x.reshape(-1, *x.shape[2:])
        obs_f       = _flat(obs_t)
        actions_f   = _flat(actions_t)
        log_probs_f = _flat(log_probs_t)
        advantages_f= _flat(advantages_t)
        targets_f   = _flat(targets_t)

        n = obs_f.shape[0]
        rng, perm_key = jax.random.split(rng)
        perm = jax.random.permutation(perm_key, n)

        def _update_minibatch(state, mb_idx):
            idx = perm[mb_idx * config.minibatch_size:(mb_idx + 1) * config.minibatch_size]
            mb = (obs_f[idx], actions_f[idx], log_probs_f[idx], advantages_f[idx], targets_f[idx])
            (loss, aux), grads = loss_and_grad(state.params, network, mb, config)
            state = state.apply_gradients(grads=grads)
            return state, (loss, aux)

        train_state, (losses, auxes) = jax.lax.scan(
            _update_minibatch,
            train_state,
            jnp.arange(config.num_minibatches),
        )
        return train_state, (losses.mean(), {k: v.mean() for k, v in auxes.items()}), rng

    logger.info(
        f"IPPO | env={config.env_name} N={config.num_agents} "
        f"B={config.num_envs} T={config.num_steps} "
        f"updates={config.num_updates}"
    )

    ep_returns = []
    t_start = time.monotonic()

    for update in range(config.num_updates):
        rng, rollout_rng = jax.random.split(rng)
        carry = (train_state, obs, env_state, rollout_rng)
        (train_state, obs, env_state, _), (transitions, infos) = collect_rollout(carry)

        obs_t, actions_t, log_probs_t, values_t, rewards_t, dones_t = transitions

        # Bootstrap value for last obs
        obs_aug = _obs_with_agent_id(obs, config.num_agents)
        _, bootstrap_v = network.apply(train_state.params, obs_aug)  # (B, N)
        all_values = jnp.concatenate([values_t, bootstrap_v[None]], axis=0)  # (T+1, B, N)

        advantages, targets = compute_gae(
            rewards_t, all_values, dones_t,
            config.gamma, config.gae_lambda, config.num_agents,
        )

        # Broadcast obs per-agent already includes augmentation from rollout;
        # re-augment collected obs for training
        obs_aug_t = jax.vmap(_obs_with_agent_id, in_axes=(0, None))(obs_t, config.num_agents)  # (T, B, N, obs+N)

        trajectories = (obs_aug_t, actions_t, log_probs_t, advantages, targets)

        for _ in range(config.update_epochs):
            rng, epoch_rng = jax.random.split(rng)
            train_state, (loss, aux), _ = update_epoch(train_state, trajectories, epoch_rng)

        # Collect episode returns from LogWrapper info
        if "returned_episode_returns" in infos:
            # infos["returned_episode_returns"]: dict agent→(T, B) or (T, B) for log wrapper
            # LogWrapper returns per-agent; sum team returns
            ret_vals = infos["returned_episode_returns"]
            if isinstance(ret_vals, dict):
                team_ret = jnp.stack(list(ret_vals.values()), axis=-1).sum(-1)  # (T, B)
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
                f"IPPO update {update + 1}/{config.num_updates} | "
                f"loss={float(loss):.4f} | "
                f"pg={float(aux['pg_loss']):.4f} vf={float(aux['vf_loss']):.4f} "
                f"ent={float(aux['entropy']):.4f} | "
                f"return={mean_ret:.2f} | "
                f"SPS={sps:.0f}"
            )

    logger.info("IPPO training complete.")
    return train_state
