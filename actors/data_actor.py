import os
import numpy as np

import ray

from config import CONFIG
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
        self.learner = learner_actor
        self.replay_buffer = replay_buffer_actor
        self.episodes_since_update = 0

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

        self.rng_key, reset_key = jax.random.split(self.rng_key)
        observation, state = self.env.reset(reset_key)
        episode = Episode()

        for _ in range(CONFIG.train.max_episode_steps):
            self.rng_key, plan_key, step_key = jax.random.split(self.rng_key, 3)
            plan_output = self.plan_fn(self.params, plan_key, observation)

            action_np = np.array(plan_output.joint_action)
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
