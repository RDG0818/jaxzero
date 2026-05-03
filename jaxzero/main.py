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
    args = parser.parse_args()

    # Instantiate environment
    if args.env == "3m":
        from jaxzero.envs.smax_wrapper import SMAXWrapper
        env = SMAXWrapper(map_name="3m", stacked_observations=4)
    else:
        from jaxzero.envs.mpe_wrapper import MPEWrapper
        env = MPEWrapper()

    # Create config from environment dimensions
    config = MAZeroConfig(
        env_name=args.env,
        num_agents=env.num_agents,
        obs_size=env.obs_size,
        action_space_size=env.action_space_size,
        stacked_observations=env.stacked_observations,
        seed=args.seed,
        training_steps=args.training_steps,
        use_reanalyze=not args.no_reanalyze,
    )

    # Run training
    from jaxzero.train import train
    params = train(config, env)
    print("Training complete.")


if __name__ == "__main__":
    main()
