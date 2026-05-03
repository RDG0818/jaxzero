import numpy as np
import jax
import jax.numpy as jnp
from jaxzero.config import MAZeroConfig
from jaxzero.mcts.sampled_mcts import SampledMCTS


def evaluate(env, params, model, config: MAZeroConfig) -> dict:
    """Evaluate the trained model on the environment using MCTS.

    Args:
        env: Environment wrapper with reset, step, get_legal_actions.
        params: Model parameters (pytree).
        model: The neural network model (e.g., MAMuZeroNet).
        config: MAZeroConfig with eval_episodes, max_episode_steps, etc.

    Returns:
        Dictionary with:
            - win_rate: fraction of episodes with positive return.
            - avg_return: mean episode return.
            - action_histogram: normalized action counts per agent.
    """
    mcts = SampledMCTS(config=config, model=model)
    rng = np.random.default_rng(config.seed + 999)

    wins = 0
    returns = []
    action_counts = np.zeros((config.num_agents, config.action_space_size))

    for ep in range(config.eval_episodes):
        import jax.random as jr
        env_rng = jr.PRNGKey(ep)
        obs, state = env.reset(env_rng)
        ep_return = 0.0
        done = False
        step = 0

        while not done and step < config.max_episode_steps:
            legal = env.get_legal_actions(state)
            result = mcts.search(params, obs[np.newaxis], legal[np.newaxis], rng)
            visit_counts = result.sampled_visit_counts[0]
            actions_pool = result.sampled_actions[0]
            chosen_idx = np.argmax(visit_counts)  # greedy at eval
            action = actions_pool[chosen_idx]

            for i, a in enumerate(action):
                action_counts[i, a] += 1

            env_rng, step_rng = jr.split(env_rng)
            obs, state, reward, done, won = env.step(step_rng, state, action)
            ep_return += reward
            step += 1

        returns.append(ep_return)
        if done and ep_return > 0:
            wins += 1

    return {
        "win_rate": wins / config.eval_episodes,
        "avg_return": float(np.mean(returns)),
        "action_histogram": (action_counts / (action_counts.sum(axis=-1, keepdims=True) + 1e-8)).tolist(),
    }
