import time
import numpy as np
from collections import deque

import ray

from config import ExperimentConfig
from utils.logging_utils import logger


def _fmt_buf_stats(stats: dict) -> str:
    return (
        f"buf={stats['size']}/{stats['capacity']} ({stats['fill_pct']:.1f}%) "
        f"pri=[{stats['priority_min']:.3f},{stats['priority_max']:.3f}] "
        f"beta={stats['beta']:.3f}"
    )


def run_warmup(data_actors: list, replay_buffer, config: ExperimentConfig) -> dict:
    """
    Fills the replay buffer to `warmup_episodes` items before training begins.
    Returns the {future: actor} dict so the training loop can pick up where
    warmup left off without restarting all actors.
    """
    target = config.train.warmup_episodes
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


def run_training_loop(
    learner,
    data_actors: list,
    replay_buffer,
    actor_tasks: dict,
    config: ExperimentConfig,
    reanalyze_actors: list = None,
):
    """
    Main training loop.

    The learner, data actors, and (optionally) reanalyze actors run fully
    asynchronously. ray.wait dispatches on whichever task finishes first.
    The learner fires a new training step immediately after each step completes.
    Reanalyze actors continuously re-run MCTS on stored observations to freshen
    policy/value targets with the latest model params.
    """
    if config.train.wandb_mode != "disabled":
        import wandb

    episodes_processed = 0
    train_steps = 0
    returns: deque = deque(maxlen=config.train.log_interval)
    train_losses: deque = deque(maxlen=config.train.log_interval)
    interval_start = time.monotonic()

    reanalyze_tasks = {}
    if reanalyze_actors:
        reanalyze_tasks = {actor.run_reanalyze.remote(): actor for actor in reanalyze_actors}

    # Steps per learner call: run the learner for this many steps before
    # returning to the main loop. Higher values reduce Ray round-trip overhead
    # at the cost of coarser-grained metrics reporting.
    learner_steps_per_call = max(1, config.train.log_interval // 10)
    learner_task = learner.run_training_loop.remote(learner_steps_per_call)
    logger.info("Starting main training loop...")

    while episodes_processed < config.train.num_episodes:
        all_pending = list(actor_tasks.keys()) + list(reanalyze_tasks.keys()) + [learner_task]
        done_refs, _ = ray.wait(all_pending, num_returns=1)
        done_ref = done_refs[0]

        if done_ref == learner_task:
            metrics = ray.get(learner_task)
            if metrics is not None:
                train_losses.append(metrics["total_loss"])
                train_steps += learner_steps_per_call
                if config.train.wandb_mode != "disabled":
                    wandb.log(metrics, step=episodes_processed)
            learner_task = learner.run_training_loop.remote(learner_steps_per_call)

        elif done_ref in reanalyze_tasks:
            ray.get(done_ref)
            finished_reanalyze = reanalyze_tasks.pop(done_ref)
            reanalyze_tasks[finished_reanalyze.run_reanalyze.remote()] = finished_reanalyze

        else:
            ep_return = ray.get(done_ref)
            returns.append(ep_return)
            episodes_processed += config.train.num_envs_per_actor

            finished_actor = actor_tasks.pop(done_ref)
            actor_tasks[finished_actor.run_episode.remote()] = finished_actor

            prev = episodes_processed - config.train.num_envs_per_actor
            if episodes_processed // config.train.log_interval > prev // config.train.log_interval and returns:
                avg_return = float(np.mean(returns))
                avg_loss = float(np.mean(train_losses)) if train_losses else 0.0
                elapsed = time.monotonic() - interval_start
                eps_per_sec = config.train.log_interval / elapsed
                steps_per_sec = train_steps / elapsed

                logger.info(
                    f"Episodes: {episodes_processed:6d} | "
                    f"Avg Return: {avg_return:8.2f} | "
                    f"Avg Loss: {avg_loss:.4f} | "
                    f"eps/s: {eps_per_sec:.1f} | "
                    f"train steps/s: {steps_per_sec:.1f}"
                )

                if config.train.debug:
                    buf_stats = ray.get(replay_buffer.get_stats.remote())
                    logger.info(f"  {_fmt_buf_stats(buf_stats)}")

                if config.train.wandb_mode != "disabled":
                    wandb.log(
                        {
                            "avg_return": avg_return,
                            "avg_loss": avg_loss,
                            "episodes": episodes_processed,
                            "eps_per_sec": eps_per_sec,
                            "train_steps_per_sec": steps_per_sec,
                        },
                        step=episodes_processed,
                    )

                train_steps = 0
                interval_start = time.monotonic()

    logger.info("Training complete.")


def run_training_loop_sync(
    learner, data_actors: list, replay_buffer, actor_tasks: dict, config: ExperimentConfig,
):
    """
    Synchronous training loop.

    All actors complete one episode batch, then the learner runs one training
    step — no overlap. Simpler to reason about and debug than the async loop,
    at the cost of ~2-3x lower throughput (actors idle while learner trains
    and vice versa). The existing actor_tasks dict is discarded; all actors
    are re-submitted each round.
    """
    if config.train.wandb_mode != "disabled":
        import wandb

    # Drain the in-flight tasks from warmup before taking control.
    if actor_tasks:
        ray.get(list(actor_tasks.keys()))

    episodes_processed = 0
    train_steps = 0
    returns: deque = deque(maxlen=config.train.log_interval)
    train_losses: deque = deque(maxlen=config.train.log_interval)
    interval_start = time.monotonic()

    logger.info("Starting synchronous training loop...")

    while episodes_processed < config.train.num_episodes:
        # Step 1: all actors run one episode batch (blocking).
        ep_returns = ray.get([actor.run_episode.remote() for actor in data_actors])
        for r in ep_returns:
            returns.append(r)
        episodes_processed += config.train.num_envs_per_actor * len(data_actors)

        # Step 2: learner trains one step (blocking).
        metrics = ray.get(learner.train.remote())
        if metrics is not None:
            train_losses.append(metrics["total_loss"])
            train_steps += 1
            if config.train.wandb_mode != "disabled":
                wandb.log(metrics, step=episodes_processed)

        # Step 3: log on interval.
        prev = episodes_processed - config.train.num_envs_per_actor * len(data_actors)
        if episodes_processed // config.train.log_interval > prev // config.train.log_interval and returns:
            avg_return = float(np.mean(returns))
            avg_loss = float(np.mean(train_losses)) if train_losses else 0.0
            elapsed = time.monotonic() - interval_start
            eps_per_sec = config.train.log_interval / elapsed
            steps_per_sec = train_steps / elapsed

            logger.info(
                f"Episodes: {episodes_processed:6d} | "
                f"Avg Return: {avg_return:8.2f} | "
                f"Avg Loss: {avg_loss:.4f} | "
                f"eps/s: {eps_per_sec:.1f} | "
                f"train steps/s: {steps_per_sec:.1f}"
            )

            if config.train.debug:
                buf_stats = ray.get(replay_buffer.get_stats.remote())
                logger.info(f"  {_fmt_buf_stats(buf_stats)}")

            if config.train.wandb_mode != "disabled":
                wandb.log(
                    {
                        "avg_return": avg_return,
                        "avg_loss": avg_loss,
                        "episodes": episodes_processed,
                        "eps_per_sec": eps_per_sec,
                        "train_steps_per_sec": steps_per_sec,
                    },
                    step=episodes_processed,
                )

            train_steps = 0
            interval_start = time.monotonic()

    logger.info("Training complete (sync).")
