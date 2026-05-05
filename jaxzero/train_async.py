import numpy as np
from collections import deque

import ray
from jaxzero.config import MAZeroConfig


def train_async(config: MAZeroConfig) -> dict:
    """
    Async actor-learner training loop.

    DataActors collect on CPU in parallel; LearnerActor trains on GPU.
    ray.wait processes whichever finishes first — GPU never waits for collection.

    Returns final params as numpy dict.
    """
    from jaxzero.actors import ReplayBufferActor, LearnerActor, DataActor

    ray.init(ignore_reinit_error=True)
    print(f"Ray resources: {ray.available_resources()}")

    replay_buffer = ReplayBufferActor.remote(config)
    learner = LearnerActor.remote(config, replay_buffer)
    data_actors = [
        DataActor.remote(i, config, learner, replay_buffer)
        for i in range(config.num_actors)
    ]
    reanalyze_actors = [
        ReanalyzeActor.remote(i, config, learner, replay_buffer)
        for i in range(config.num_reanalyze_actors)
    ]

    # -- Warmup: fill buffer before training starts --
    target = config.min_replay_size
    print(f"Warming up: filling buffer to {target} games...")
    actor_tasks = {actor.run_episode.remote(): actor for actor in data_actors}
    reanalyze_tasks = {actor.run_reanalyze.remote(): actor for actor in reanalyze_actors}

    while True:
        buf_size = ray.get(replay_buffer.get_size.remote())
        print(f"  Buffer: {buf_size}/{target}", end="\r", flush=True)
        if buf_size >= target:
            break
        done_refs, _ = ray.wait(list(actor_tasks.keys()), num_returns=1)
        done_ref = done_refs[0]
        finished_actor = actor_tasks.pop(done_ref)
        ray.get(done_ref)
        actor_tasks[finished_actor.run_episode.remote()] = finished_actor
    print(f"\nWarmup complete.")

    # -- Async training loop --
    learner_task = learner.run_training_loop.remote(config.learner_steps_per_call)
    total_steps = 0
    recent_returns: deque = deque(maxlen=100)

    print(f"Starting async training loop with {len(data_actors)} data actors and {len(reanalyze_actors)} reanalyze actors...")
    while total_steps < config.training_steps:
        all_pending = list(actor_tasks.keys()) + list(reanalyze_tasks.keys()) + [learner_task]
        done_refs, _ = ray.wait(all_pending, num_returns=1)
        done_ref = done_refs[0]

        if done_ref == learner_task:
            metrics = ray.get(learner_task)
            if metrics is not None:
                total_steps = metrics["step"]
                if total_steps % config.log_interval < config.learner_steps_per_call:
                    mean_ret = float(np.mean(recent_returns)) if recent_returns else float("nan")
                    print(
                        f"Step {total_steps}: loss={metrics['total_loss']:.4f}"
                        f" | r={metrics['reward_loss']:.3f}"
                        f" v={metrics['value_loss']:.3f}"
                        f" p={metrics['policy_loss']:.3f}"
                        f" | ep_return={mean_ret:.2f}"
                    )
            if total_steps < config.training_steps:
                learner_task = learner.run_training_loop.remote(config.learner_steps_per_call)

        elif done_ref in reanalyze_tasks:
            finished_reanalyze = reanalyze_tasks.pop(done_ref)
            reanalyze_tasks[finished_reanalyze.run_reanalyze.remote()] = finished_reanalyze

        else:
            ep_return = ray.get(done_ref)
            recent_returns.append(ep_return)
            finished_actor = actor_tasks.pop(done_ref)
            actor_tasks[finished_actor.run_episode.remote()] = finished_actor

    params = ray.get(learner.get_params.remote())
    ray.shutdown()
    return params
