import time
import numpy as np
from collections import deque

import ray

from config import CONFIG
from utils.logging_utils import logger


def run_warmup(data_actors: list, replay_buffer) -> dict:
    """
    Fills the replay buffer to `warmup_episodes` items before training begins.
    Returns the {future: actor} dict so the training loop can pick up where
    warmup left off without restarting all actors.
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
        ray.get(done_ref)
        actor_tasks[finished_actor.run_episode.remote()] = finished_actor

    logger.info(f"\nWarmup complete in {time.time() - start:.1f}s.")
    return actor_tasks


def run_training_loop(learner, data_actors: list, replay_buffer, actor_tasks: dict):
    """
    Main training loop.

    The learner and actors run fully asynchronously. ray.wait dispatches on
    whichever task finishes first so neither blocks the other. The learner
    fires a new training step immediately after each step completes.
    """
    if CONFIG.train.wandb_mode != "disabled":
        import wandb

    episodes_processed = 0
    returns: deque = deque(maxlen=CONFIG.train.log_interval)
    train_losses: deque = deque(maxlen=CONFIG.train.log_interval)
    start_time = time.time()

    learner_task = learner.train.remote()
    logger.info("Starting main training loop...")

    while episodes_processed < CONFIG.train.num_episodes:
        all_pending = list(actor_tasks.keys()) + [learner_task]
        done_refs, _ = ray.wait(all_pending, num_returns=1)
        done_ref = done_refs[0]

        if done_ref == learner_task:
            metrics = ray.get(learner_task)
            if metrics is not None:
                train_losses.append(metrics["total_loss"])
                if CONFIG.train.wandb_mode != "disabled":
                    wandb.log(metrics, step=episodes_processed)
            learner_task = learner.train.remote()

        else:
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
