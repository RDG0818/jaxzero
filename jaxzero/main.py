import argparse
from jaxzero.config import MAZeroConfig


def main():
    """Entry point for MAZero training and evaluation."""
    parser = argparse.ArgumentParser(description="MAZero training on SMAX or MPE environments")
    parser.add_argument(
        "--env",
        default="3m",
        choices=["3m", "mpe"],
        help="Environment: '3m' (SMAX) or 'mpe' (MPE SimpleSpreads)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--training_steps",
        type=int,
        default=100_000,
        help="Number of training steps",
    )
    parser.add_argument(
        "--no_reanalyze",
        action="store_true",
        help="Disable reanalyze (use raw MCTS targets)",
    )
    parser.add_argument(
        "--num_simulations",
        type=int,
        default=None,
        help="MCTS simulations per step (default: 100; use 10-20 for quick tests)",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="Parallel envs for collection (default: 8)",
    )
    parser.add_argument(
        "--updates_per_collection",
        type=int,
        default=None,
        help="Gradient updates per collection round (default: 10)",
    )
    parser.add_argument(
        "--async_training",
        action="store_true",
        help="Use Ray async actor-learner training (DataActors on CPU + LearnerActor on GPU)",
    )
    parser.add_argument(
        "--num_actors",
        type=int,
        default=None,
        help="Number of parallel DataActors for async training (default: 3)",
    )
    args = parser.parse_args()

    # For async training, init Ray before JAX is imported (via env probe below).
    # Ray workers fork from main process; if JAX is already initialized in main, the
    # fork copies JAX state into workers → deadlock. Initializing Ray first ensures
    # workers fork from a JAX-free parent.
    if args.async_training:
        import ray
        ray.init(ignore_reinit_error=True)

    # Build env factory and probe one instance for dimensions
    if args.env == "3m":
        from jaxzero.envs.smax_wrapper import SMAXWrapper
        env_fn = lambda: SMAXWrapper(map_name="3m", stacked_observations=4)
    else:
        from jaxzero.envs.mpe_wrapper import MPEWrapper
        env_fn = lambda: MPEWrapper()

    probe = env_fn()
    overrides = {}
    if args.num_simulations is not None:
        overrides["num_simulations"] = args.num_simulations
    if args.num_envs is not None:
        overrides["num_envs_parallel"] = args.num_envs
    if args.updates_per_collection is not None:
        overrides["updates_per_collection"] = args.updates_per_collection
    if args.num_actors is not None:
        overrides["num_actors"] = args.num_actors
    config = MAZeroConfig(
        env_name=args.env,
        num_agents=probe.num_agents,
        obs_size=probe.obs_size,
        action_space_size=probe.action_space_size,
        stacked_observations=probe.stacked_observations,
        seed=args.seed,
        training_steps=args.training_steps,
        use_reanalyze=not args.no_reanalyze,
        **overrides,
    )

    if args.async_training:
        from jaxzero.train_async import train_async
        params = train_async(config)
    else:
        from jaxzero.train import train
        params = train(config, env_fn)
    print("Training complete.")


if __name__ == "__main__":
    main()
