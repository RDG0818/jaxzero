# train_v2.py
"""
Asynchronous actor-learner training for Multi-Agent MuZero.

Architecture
------------
  LearnerActor  (1×, GPU)  trains model parameters, serves them to actors
  DataActor     (N×, CPU)  runs MCTS episodes, ships replay items to buffer
  ReplayBufferActor (1×)   stores and samples prioritized experience

JAX + Ray constraint
--------------------
  All JAX imports MUST be inside actor __init__ / methods — never at module
  level. Ray spawns isolated subprocesses; importing JAX at module level in an
  actor file triggers GPU allocation before CUDA_VISIBLE_DEVICES is set, causing
  SEGFAULTs or driver crashes.
"""

import os
import time
import numpy as np
from collections import deque
from typing import List, Tuple

import ray

from config import CONFIG
from utils.logging_utils import logger


# ---------------------------------------------------------------------------
# ReplayBufferActor
# ---------------------------------------------------------------------------

@ray.remote
class ReplayBufferActor:
    """Manages prioritized experience replay in a dedicated process."""

    def __init__(self, obs_size: int, action_size: int):
        from utils.replay_buffer import ReplayBuffer

        self.buffer = ReplayBuffer(
            capacity=CONFIG.train.replay_buffer_size,
            observation_shape=(obs_size,),
            action_space_size=action_size,
            num_agents=CONFIG.train.num_agents,
            unroll_steps=CONFIG.train.unroll_steps,
            alpha=CONFIG.train.replay_buffer_alpha,
            beta_start=CONFIG.train.replay_buffer_beta_start,
            beta_frames=CONFIG.train.replay_buffer_beta_frames,
        )

    def add(self, items: list, priorities: List[float]):
        for item, priority in zip(items, priorities):
            self.buffer.add(item, priority)

    def sample(self, batch_size: int):
        return self.buffer.sample(batch_size)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        self.buffer.update_priorities(indices, priorities)

    def get_size(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# LearnerActor
# ---------------------------------------------------------------------------

@ray.remote(num_gpus=1)
class LearnerActor:
    """
    Trains the MuZero model on GPU.

    Pulls batches from the replay buffer, runs a JIT-compiled training step,
    and serves updated parameters to DataActors on request.

    The training step is built as a closure so that model, optimizer, and
    support objects are captured at construction time. This avoids static_argnames
    and the associated retracing pitfalls.
    """

    def __init__(self, obs_size: int, action_size: int, replay_buffer_actor):
        # Must be set before importing JAX so XLA picks up these settings.
        # Ray automatically sets CUDA_VISIBLE_DEVICES for num_gpus=1 actors.
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.70"
        # Suppress XLA GEMM autotuner "all configs filtered" spam.
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "") + " --xla_gpu_autotune_level=0"
        )

        import jax
        import jax.numpy as jnp
        import optax
        from utils.utils import DiscreteSupport, scalar_to_support, support_to_scalar
        from model.model import FlaxMAMuZeroNet

        logger.info(f"(Learner pid={os.getpid()}) Initializing on GPU...")

        self.replay_buffer = replay_buffer_actor
        self.train_step_count = 0
        self.rng_key = jax.random.PRNGKey(0)

        value_support = DiscreteSupport(
            min=-CONFIG.model.value_support_size,
            max=CONFIG.model.value_support_size,
        )
        reward_support = DiscreteSupport(
            min=-CONFIG.model.reward_support_size,
            max=CONFIG.model.reward_support_size,
        )

        # ---- Model ----
        model = FlaxMAMuZeroNet(CONFIG.model, action_size)
        dummy_obs = jnp.ones((1, CONFIG.train.num_agents, obs_size))
        self.rng_key, init_key = jax.random.split(self.rng_key)
        self.params = model.init(init_key, dummy_obs)["params"]

        # ---- Optimizer ----
        lr = CONFIG.train.learning_rate
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr,
            warmup_steps=CONFIG.train.lr_warmup_steps,
            decay_steps=CONFIG.train.num_episodes - CONFIG.train.lr_warmup_steps,
            end_value=lr * CONFIG.train.end_lr_factor,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(CONFIG.train.gradient_clip_norm),
            optax.adamw(learning_rate=lr_schedule),
        )
        self.opt_state = optimizer.init(self.params)
        self.lr_schedule = lr_schedule

        # ---- Build JIT-compiled training step as a closure ----
        # Capturing model/optimizer/supports here means JIT only traces once,
        # with no static_argnames needed.
        U = CONFIG.train.unroll_steps
        value_scale = CONFIG.train.value_scale
        consistency_scale = CONFIG.train.consistency_scale

        def train_step(params, opt_state, batch, weights, rng_key):
            # Pre-compute categorical support targets outside loss_fn so they
            # are treated as constants by jax.value_and_grad (zero gradient).
            value_target_dist = scalar_to_support(
                batch.value_target.mean(axis=2), value_support
            )   # (B, U+1, Sv)
            reward_target_dist = scalar_to_support(
                batch.reward_target.mean(axis=2), reward_support
            )   # (B, U, Sr)

            def loss_fn(p):
                rng_init, _, rng_unroll = jax.random.split(rng_key, 3)
                unroll_keys = jax.random.split(rng_unroll, U)  # (U, 2)

                # ---- Step-0: initial inference ----
                init_out = model.apply(
                    {"params": p}, batch.observation, rngs={"dropout": rng_init}
                )
                hidden = init_out.hidden_state  # (B, N, D)

                # Policy loss: mean over agents N to get (B,)
                p0_loss = optax.softmax_cross_entropy(
                    init_out.policy_logits, batch.policy_target[:, 0]
                ).mean(axis=-1)   # (B, N, A) × (B, N, A) → (B, N) → (B,)

                # Value loss: centralized, so (B, Sv) directly → (B,)
                v0_loss = optax.softmax_cross_entropy(
                    init_out.value_logits, value_target_dist[:, 0]
                )

                # ---- Steps 1..U: unroll via scan ----
                def scan_step(hidden, inputs):
                    ai, ri_dist, pi_target, vi_dist, step_key = inputs

                    # Online projection before advancing the state.
                    online_proj = model.apply(
                        {"params": p}, hidden, method=model.project_online
                    )   # (B, N, proj_dim)

                    out = model.apply(
                        {"params": p},
                        hidden,
                        ai,
                        method=model.recurrent_inference,
                        rngs={"dropout": step_key},
                    )
                    next_hidden = out.hidden_state

                    # Target projection: stop gradient so the online branch
                    # is the only one that updates toward the target.
                    target_proj = jax.lax.stop_gradient(
                        model.apply(
                            {"params": p}, next_hidden, method=model.project_target
                        )
                    )

                    ri_loss = optax.softmax_cross_entropy(out.reward_logits, ri_dist)
                    # (B,)
                    pi_loss = optax.softmax_cross_entropy(
                        out.policy_logits, pi_target
                    ).mean(axis=-1)
                    # (B, N) → (B,)
                    vi_loss = optax.softmax_cross_entropy(out.value_logits, vi_dist)
                    # (B,)

                    # Cosine consistency loss: flatten agents into batch dim.
                    B_, N_, D_ = online_proj.shape
                    sim = optax.cosine_similarity(
                        online_proj.reshape(B_ * N_, D_),
                        target_proj.reshape(B_ * N_, D_),
                    ).reshape(B_, N_).mean(axis=-1)   # (B,)
                    cons_loss = -sim

                    return next_hidden, (ri_loss, pi_loss, vi_loss, cons_loss)

                # Transpose batch-major → step-major for scan.
                xs = (
                    jnp.moveaxis(batch.actions, 1, 0),                    # (U, B, N)
                    jnp.moveaxis(reward_target_dist, 1, 0),               # (U, B, Sr)
                    jnp.moveaxis(batch.policy_target[:, 1:], 1, 0),       # (U, B, N, A)
                    jnp.moveaxis(value_target_dist[:, 1:], 1, 0),         # (U, B, Sv)
                    unroll_keys,                                           # (U, 2)
                )
                _, (ri_losses, pi_losses, vi_losses, cons_losses) = jax.lax.scan(
                    scan_step, hidden, xs
                )
                # Each output: (U, B)

                # Average unroll losses, normalizing step-0 and unroll together
                # where applicable.
                reward_loss      = ri_losses.mean(axis=0)                          # (B,)
                policy_loss      = (p0_loss + pi_losses.sum(axis=0)) / (U + 1)    # (B,)
                value_loss       = (v0_loss + vi_losses.sum(axis=0)) / (U + 1)    # (B,)
                consistency_loss = cons_losses.mean(axis=0)                        # (B,)

                loss = (
                    reward_loss
                    + policy_loss
                    + value_loss * value_scale
                    + consistency_loss * consistency_scale
                )
                # Importance-sampling weighted mean for PER.
                total_loss = (loss * weights).mean()

                # TD error for priority updates: |V_pred - V_target| at step 0.
                td_error = jnp.abs(
                    support_to_scalar(init_out.value_logits, value_support)
                    - batch.value_target[:, 0].mean(axis=1)
                )   # (B,)

                metrics = {
                    "total_loss": total_loss,
                    "reward_loss": reward_loss.mean(),
                    "policy_loss": policy_loss.mean(),
                    "value_loss": value_loss.mean(),
                    "consistency_loss": consistency_loss.mean(),
                }
                return total_loss, (metrics, td_error)

            (_, (metrics, td_error)), grads = jax.value_and_grad(
                loss_fn, has_aux=True
            )(params)
            metrics["grad_norm"] = optax.global_norm(grads)

            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            new_priorities = td_error + 1e-6

            return new_params, new_opt_state, metrics, new_priorities

        self.train_step = jax.jit(train_step)
        logger.info(f"(Learner pid={os.getpid()}) Setup complete.")

    def train(self):
        """
        Samples a batch, runs one training step, updates priorities.

        Returns a metrics dict, or None if the buffer is empty.
        """
        import jax

        batch, weights, indices = ray.get(
            self.replay_buffer.sample.remote(CONFIG.train.batch_size)
        )
        if batch is None:
            return None

        self.rng_key, train_key = jax.random.split(self.rng_key)
        jax_batch = jax.tree_util.tree_map(jax.device_put, batch)
        jax_weights = jax.device_put(np.array(weights, dtype=np.float32))

        self.params, self.opt_state, metrics, new_priorities = self.train_step(
            self.params, self.opt_state, jax_batch, jax_weights, train_key
        )
        self.train_step_count += 1

        self.replay_buffer.update_priorities.remote(indices, np.array(new_priorities))

        metrics = {k: float(v) for k, v in metrics.items()}
        metrics["learning_rate"] = float(self.lr_schedule(self.train_step_count))
        return metrics

    def get_params(self):
        return self.params

    def get_train_step_count(self) -> int:
        return self.train_step_count


# ---------------------------------------------------------------------------
# DataActor
# ---------------------------------------------------------------------------

@ray.remote(num_cpus=1)
class DataActor:
    """
    Generates experience on CPU via MCTS planning.

    Runs episodes, converts them to ReplayItems, and ships them to
    the ReplayBufferActor. Periodically syncs parameters from LearnerActor.
    """

    def __init__(
        self,
        actor_id: int,
        obs_size: int,
        action_size: int,
        learner_actor,
        replay_buffer_actor,
    ):
        # Ray sets CUDA_VISIBLE_DEVICES="" for CPU workers, causing the JAX CUDA
        # plugin to crash on cuInit(0). Pop it so the device is visible and cuInit
        # succeeds, then set JAX_PLATFORMS=cpu so JAX never actually uses the GPU.
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["JAX_PLATFORMS"] = "cpu"

        import jax
        from model.model import FlaxMAMuZeroNet
        from mcts.mcts_independent import MCTSIndependentPlanner
        from mcts.mcts_joint import MCTSJointPlanner
        from utils.mpe_env_wrapper import MPEEnvWrapper

        self.actor_id = actor_id
        self.learner = learner_actor
        self.replay_buffer = replay_buffer_actor
        self.episodes_since_update = 0

        # Deterministic but non-colliding seeds across actors.
        self.rng_key = jax.random.PRNGKey(actor_id * 1337 + 7)

        self.env = MPEEnvWrapper(
            CONFIG.train.env_name,
            CONFIG.train.num_agents,
            CONFIG.train.max_episode_steps,
        )

        model = FlaxMAMuZeroNet(CONFIG.model, action_size)
        planner_map = {
            "independent": MCTSIndependentPlanner,
            "joint": MCTSJointPlanner,
        }
        if CONFIG.mcts.planner_mode not in planner_map:
            raise ValueError(
                f"Unknown planner_mode '{CONFIG.mcts.planner_mode}'. "
                f"Choose from: {list(planner_map)}"
            )
        planner = planner_map[CONFIG.mcts.planner_mode](model=model, config=CONFIG)

        # Single JIT boundary: DataActor owns compilation of the plan function.
        # Planners do not self-JIT; this is the only jax.jit call for data collection.
        self.plan_fn = jax.jit(planner.plan)

        self.params = ray.get(learner_actor.get_params.remote())
        logger.info(f"(DataActor {actor_id} pid={os.getpid()}) Setup complete.")

    def run_episode(self) -> float:
        """
        Runs one episode with MCTS planning, processes it into ReplayItems,
        and ships them to the buffer.

        Returns the undiscounted episode return.
        """
        import jax
        from utils.replay_buffer import Episode, Transition, process_episode

        self.rng_key, reset_key = jax.random.split(self.rng_key)
        # MPEEnvWrapper returns (1, N, obs_size) — the leading 1 is the batch dim
        # needed by the model. We keep it for plan_fn and strip it for storage.
        observation, state = self.env.reset(reset_key)
        episode = Episode()

        for _ in range(CONFIG.train.max_episode_steps):
            self.rng_key, plan_key, step_key = jax.random.split(self.rng_key, 3)
            plan_output = self.plan_fn(self.params, plan_key, observation)

            action_np = np.array(plan_output.joint_action)
            next_obs, next_state, reward, done = self.env.step(step_key, state, action_np)

            episode.add_step(
                Transition(
                    observation=observation[0],   # (N, obs_size) — drop batch dim for storage
                    action=action_np,
                    reward=reward,
                    done=done,
                    policy_target=np.array(plan_output.policy_targets),
                    value_target=float(plan_output.root_value),
                    agent_order=np.array(plan_output.agent_order),
                )
            )
            observation = next_obs
            state = next_state
            if done:
                break

        items = process_episode(
            episode,
            CONFIG.train.unroll_steps,
            CONFIG.train.n_step,
            CONFIG.train.discount_gamma,
            CONFIG.train.num_agents,
        )

        if items:
            self.replay_buffer.add.remote(items, [1.0] * len(items))

        self.episodes_since_update += 1
        if self.episodes_since_update >= CONFIG.train.param_update_interval:
            self.params = ray.get(self.learner.get_params.remote())
            self.episodes_since_update = 0

        return episode.episode_return


# ---------------------------------------------------------------------------
# System initialization helpers
# ---------------------------------------------------------------------------

@ray.remote
def _fetch_env_metadata() -> Tuple[int, int]:
    """
    Retrieves observation and action sizes from the configured environment.

    Runs as a Ray task so that JAX (imported transitively by MPEEnvWrapper)
    is never initialized in the main process, preserving GPU memory for the
    LearnerActor.
    """
    # Ray sets CUDA_VISIBLE_DEVICES="" for workers with no GPU allocation, which
    # causes the JAX CUDA plugin to crash on cuInit(0). Pop the restriction so
    # the GPU is visible (cuInit succeeds), then use JAX_PLATFORMS=cpu so JAX
    # selects CPU as its backend and never actually allocates GPU memory.
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ["JAX_PLATFORMS"] = "cpu"
    from utils.mpe_env_wrapper import MPEEnvWrapper

    env = MPEEnvWrapper(
        CONFIG.train.env_name,
        CONFIG.train.num_agents,
        CONFIG.train.max_episode_steps,
    )
    return env.observation_size, env.action_space_size


def initialize_actors(obs_size: int, action_size: int):
    """Instantiates and returns all Ray actors."""
    replay_buffer = ReplayBufferActor.options(num_cpus=4).remote(obs_size, action_size)
    learner = LearnerActor.remote(obs_size, action_size, replay_buffer)
    data_actors = [
        DataActor.remote(i, obs_size, action_size, learner, replay_buffer)
        for i in range(CONFIG.train.num_actors)
    ]
    return replay_buffer, learner, data_actors


# ---------------------------------------------------------------------------
# Training phases
# ---------------------------------------------------------------------------

def run_warmup(data_actors: list, replay_buffer) -> dict:
    """
    Fills the replay buffer to `warmup_episodes` items before training begins.

    Returns the {future: actor} dict of still-running actor tasks so the
    training loop can continue from where warmup left off.
    """
    target = CONFIG.train.warmup_episodes
    logger.info(f"Warmup: filling buffer to {target} items...")
    start = time.time()

    actor_tasks = {actor.run_episode.remote(): actor for actor in data_actors}

    while True:
        buffer_size = ray.get(replay_buffer.get_size.remote())
        print(f"  Buffer: {buffer_size}/{target}", end="\r")
        if buffer_size >= target:
            break
        done_refs, _ = ray.wait(list(actor_tasks.keys()), num_returns=1)
        done_ref = done_refs[0]
        finished_actor = actor_tasks.pop(done_ref)
        ray.get(done_ref)  # consume result; return value not needed during warmup
        actor_tasks[finished_actor.run_episode.remote()] = finished_actor

    logger.info(f"\nWarmup complete in {time.time() - start:.1f}s.")
    return actor_tasks


def run_training_loop(
    learner,
    data_actors: list,
    replay_buffer,
    actor_tasks: dict,
):
    """
    Main training loop.

    The learner and actors run fully asynchronously. ray.wait dispatches on
    whichever task finishes first — the learner or an actor — so neither
    blocks the other. The learner fires a new training step immediately after
    each step completes, maximising GPU utilisation.
    """
    if CONFIG.train.wandb_mode != "disabled":
        import wandb

    episodes_processed = 0
    returns: deque = deque(maxlen=CONFIG.train.log_interval)
    train_losses: deque = deque(maxlen=CONFIG.train.log_interval)
    start_time = time.time()

    # Kick off the first learner training step.
    learner_task = learner.train.remote()

    logger.info("Starting main training loop...")

    while episodes_processed < CONFIG.train.num_episodes:
        all_pending = list(actor_tasks.keys()) + [learner_task]
        done_refs, _ = ray.wait(all_pending, num_returns=1)
        done_ref = done_refs[0]

        if done_ref == learner_task:
            # ---- Learner finished a training step ----
            metrics = ray.get(learner_task)
            if metrics is not None:
                train_losses.append(metrics["total_loss"])
                if CONFIG.train.wandb_mode != "disabled":
                    wandb.log(metrics, step=episodes_processed)
            # Immediately queue the next training step.
            learner_task = learner.train.remote()

        else:
            # ---- An actor finished an episode ----
            ep_return = ray.get(done_ref)
            returns.append(ep_return)
            episodes_processed += 1

            finished_actor = actor_tasks.pop(done_ref)
            actor_tasks[finished_actor.run_episode.remote()] = finished_actor

            if episodes_processed % CONFIG.train.log_interval == 0 and returns:
                avg_return = float(np.mean(returns))
                avg_loss = float(np.mean(train_losses)) if train_losses else 0.0
                elapsed = time.time() - start_time
                logger.info(
                    f"Episodes: {episodes_processed:6d} | "
                    f"Avg Return: {avg_return:8.2f} | "
                    f"Avg Loss: {avg_loss:.4f} | "
                    f"Elapsed: {elapsed:.1f}s"
                )
                if CONFIG.train.wandb_mode != "disabled":
                    wandb.log(
                        {
                            "avg_return": avg_return,
                            "avg_loss": avg_loss,
                            "episodes": episodes_processed,
                        },
                        step=episodes_processed,
                    )
                start_time = time.time()

    logger.info("Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # Opt into future Ray behavior: don't override CUDA_VISIBLE_DEVICES for
    # zero-GPU workers. We manage device visibility ourselves in each actor.
    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
    ray.init(ignore_reinit_error=True)
    logger.info(f"Ray resources: {ray.available_resources()}")

    if CONFIG.train.wandb_mode != "disabled":
        import wandb
        wandb.init(project=CONFIG.train.project_name, config=CONFIG)

    # Fetch env metadata in a Ray task to avoid initializing JAX in the main
    # process (which would compete with the LearnerActor for GPU memory).
    obs_size, action_size = ray.get(_fetch_env_metadata.remote())
    logger.info(f"Env: obs_size={obs_size}, action_size={action_size}")

    replay_buffer, learner, data_actors = initialize_actors(obs_size, action_size)

    # Run one episode per actor upfront to trigger JIT compilation before the
    # timed warmup phase begins.
    logger.info("Compiling plan_fn (one warm-up episode per actor)...")
    ray.get([actor.run_episode.remote() for actor in data_actors])

    actor_tasks = run_warmup(data_actors, replay_buffer)
    run_training_loop(learner, data_actors, replay_buffer, actor_tasks)

    if CONFIG.train.wandb_mode != "disabled":
        import wandb
        wandb.finish()

    ray.shutdown()


if __name__ == "__main__":
    main()
