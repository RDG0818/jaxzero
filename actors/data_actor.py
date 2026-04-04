import os
import time
import numpy as np

import ray

from config import ExperimentConfig
from utils.logging_utils import logger


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
        config: ExperimentConfig,
    ):
        # Ray sets CUDA_VISIBLE_DEVICES="" for CPU workers, causing the JAX
        # CUDA plugin to crash on cuInit(0). Pop the restriction so the device
        # is visible (cuInit succeeds), then set JAX_PLATFORMS=cpu so JAX
        # never allocates GPU memory.
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["JAX_PLATFORMS"] = "cpu"

        import jax
        from model import FlaxMAMuZeroNet
        from mcts import MCTSIndependentPlanner, MCTSJointPlanner
        from envs import MPEEnvWrapper

        self.actor_id = actor_id
        self.config = config
        self.learner = learner_actor
        self.replay_buffer = replay_buffer_actor
        self.episodes_since_update = 0

        self.rng_key = jax.random.PRNGKey(actor_id * 1337 + 7)

        self.env = MPEEnvWrapper(
            config.train.env_name,
            config.train.num_agents,
            config.train.max_episode_steps,
        )

        model = FlaxMAMuZeroNet(config.model, action_size)
        planner_map = {
            "independent": MCTSIndependentPlanner,
            "joint": MCTSJointPlanner,
        }
        if config.mcts.planner_mode not in planner_map:
            raise ValueError(
                f"Unknown planner_mode '{config.mcts.planner_mode}'. "
                f"Choose from: {list(planner_map)}"
            )
        planner = planner_map[config.mcts.planner_mode](model=model, config=config)

        # Single JIT boundary: DataActor owns compilation of the plan function.
        self.plan_fn = jax.jit(planner.plan)
        self.params = ray.get(learner_actor.get_params.remote())
        logger.info(f"(DataActor {actor_id} pid={os.getpid()}) Setup complete.")

    def run_episode(self) -> float:
        """
        Runs one episode, processes it into ReplayItems, ships to buffer.
        Returns the undiscounted episode return.
        """
        import jax
        from utils.replay_buffer import Episode, Transition, process_episode

        debug = self.config.train.debug
        ep_start = time.monotonic()

        self.rng_key, reset_key = jax.random.split(self.rng_key)
        observation, state = self.env.reset(reset_key)
        episode = Episode()

        plan_times = []
        root_values = []

        for _ in range(self.config.train.max_episode_steps):
            self.rng_key, plan_key, step_key = jax.random.split(self.rng_key, 3)

            t_plan = time.monotonic()
            plan_output = self.plan_fn(self.params, plan_key, observation)
            plan_times.append(time.monotonic() - t_plan)

            action_np = np.array(plan_output.joint_action)

            if debug:
                root_values.append(float(plan_output.root_value))
                policy_targets = np.array(plan_output.policy_targets)
                if np.isnan(policy_targets).any() or np.isnan(action_np).any():
                    logger.warning(
                        f"(DataActor {self.actor_id}) NaN detected in plan output at step {len(episode.trajectory)}"
                    )

            next_obs, next_state, reward, done = self.env.step(step_key, state, action_np)

            episode.add_step(
                Transition(
                    observation=observation[0],  # (N, obs_size) — drop batch dim
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

        if debug:
            ep_time = time.monotonic() - ep_start
            ep_len = len(episode.trajectory)
            mean_plan_ms = np.mean(plan_times) * 1000
            mean_root_value = np.mean(root_values)
            # Policy entropy: mean over steps and agents
            all_targets = np.stack([t.policy_target for t in episode.trajectory])  # (T, N, A)
            p = np.clip(all_targets, 1e-8, None)
            policy_entropy = float(-np.sum(p * np.log(p), axis=-1).mean())
            logger.info(
                f"(DataActor {self.actor_id}) "
                f"ep_len={ep_len} return={episode.episode_return:.3f} "
                f"mean_root_value={mean_root_value:.3f} "
                f"policy_entropy={policy_entropy:.3f} "
                f"mean_plan={mean_plan_ms:.1f}ms "
                f"ep_time={ep_time:.2f}s"
            )

        items = process_episode(
            episode,
            self.config.train.unroll_steps,
            self.config.train.n_step,
            self.config.train.discount_gamma,
            self.config.train.num_agents,
        )

        if items:
            self.replay_buffer.add.remote(items, [1.0] * len(items))

        self.episodes_since_update += 1
        if self.episodes_since_update >= self.config.train.param_update_interval:
            self.params = ray.get(self.learner.get_params.remote())
            self.episodes_since_update = 0
            logger.info(f"(DataActor {self.actor_id}) Synced params from learner.")

        return episode.episode_return
