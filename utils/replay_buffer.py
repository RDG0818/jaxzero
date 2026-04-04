# replay_buffer.py
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
from jax import tree_util


@dataclass
class Transition:
    """
    Holds all the data for a single step (or transition) in an environment.
    """
    observation: np.ndarray
    action: np.ndarray
    reward: float
    done: bool
    policy_target: np.ndarray
    value_target: float
    agent_order: np.ndarray


@dataclass
class Episode:
    """
    A container for a full episode's trajectory and metadata.
    """
    trajectory: List[Transition] = field(default_factory=list)
    episode_return: float = 0.0

    def add_step(self, transition: Transition):
        self.trajectory.append(transition)
        self.episode_return += transition.reward


@dataclass
class ReplayItem:
    """
    A single, self-contained training sample for the MuZero model.

    Shapes (using shorthand):
        B = batch_size, U = unroll_steps, N = num_agents, A = action_space_size
    """
    observation: np.ndarray    # (B, N, obs_size)
    actions: np.ndarray        # (B, U, N)
    policy_target: np.ndarray  # (B, U+1, N, A)
    value_target: np.ndarray   # (B, U+1, N)
    reward_target: np.ndarray  # (B, U, N)
    agent_order: np.ndarray    # (B, N)


def flatten_replay_item(item: ReplayItem):
    children = (
        item.observation,
        item.actions,
        item.policy_target,
        item.value_target,
        item.reward_target,
        item.agent_order,
    )
    return children, None


def unflatten_replay_item(static_data, children):
    return ReplayItem(
        observation=children[0],
        actions=children[1],
        policy_target=children[2],
        value_target=children[3],
        reward_target=children[4],
        agent_order=children[5],
    )


tree_util.register_pytree_node(
    ReplayItem,
    flatten_replay_item,
    unflatten_replay_item,
)


# ---------------------------------------------------------------------------
# Episode processing
# ---------------------------------------------------------------------------

def process_episode(
    episode: "Episode",
    unroll_steps: int,
    n_step: int,
    discount_gamma: float,
    num_agents: int,
) -> list:
    """
    Converts a completed episode into ReplayItems via a sliding window.

    For each valid start position, computes n-step bootstrapped value targets
    and packs `unroll_steps` transitions into a single ReplayItem.

    Args:
        episode: Completed episode containing a trajectory of Transitions.
        unroll_steps: Number of steps per training sample (U).
        n_step: Lookahead horizon for bootstrapped value targets.
        discount_gamma: Discount factor γ.
        num_agents: Number of agents N (for broadcasting scalar targets).

    Returns:
        List of ReplayItems, one per valid start index.
        Empty list if the episode is too short to produce any samples.
    """
    trajectory = episode.trajectory
    ep_len = len(trajectory)

    if ep_len <= unroll_steps:
        return []

    # Extract arrays once to avoid repeated attribute lookups in the loop.
    observations = np.stack([t.observation for t in trajectory])      # (T, N, obs_size)
    actions = np.stack([t.action for t in trajectory])                # (T, N)
    policy_targets = np.stack([t.policy_target for t in trajectory])  # (T, N, A)
    rewards = np.array([t.reward for t in trajectory], dtype=np.float32)      # (T,)
    mcts_values = np.array([t.value_target for t in trajectory], dtype=np.float32)  # (T,)
    agent_orders = np.stack([t.agent_order for t in trajectory])      # (T, N)

    # Pre-compute discount coefficients [γ^0, γ^1, ..., γ^(n-1)] for np.dot.
    discount_vec = discount_gamma ** np.arange(n_step, dtype=np.float32)

    replay_items = []
    for start in range(ep_len - unroll_steps):
        # Compute n-step bootstrapped value target for each of the U+1 positions.
        value_targets = np.empty(unroll_steps + 1, dtype=np.float32)
        for i in range(unroll_steps + 1):
            t = start + i
            window = rewards[t : t + n_step]
            value_targets[i] = np.dot(window, discount_vec[: len(window)])
            bootstrap_idx = t + n_step
            if bootstrap_idx < ep_len:
                value_targets[i] += mcts_values[bootstrap_idx] * (discount_gamma ** n_step)

        # Broadcast scalar value/reward targets to per-agent arrays.
        # np.broadcast_to returns a read-only view; .copy() makes it writable.
        value_target_per_agent = np.broadcast_to(
            value_targets[:, None], (unroll_steps + 1, num_agents)
        ).copy().astype(np.float32)  # (U+1, N)

        reward_target_per_agent = np.broadcast_to(
            rewards[start : start + unroll_steps, None], (unroll_steps, num_agents)
        ).copy().astype(np.float32)  # (U, N)

        replay_items.append(
            ReplayItem(
                observation=observations[start],                                    # (N, obs_size)
                actions=actions[start : start + unroll_steps],                     # (U, N)
                policy_target=policy_targets[start : start + unroll_steps + 1],   # (U+1, N, A)
                value_target=value_target_per_agent,                               # (U+1, N)
                reward_target=reward_target_per_agent,                             # (U, N)
                agent_order=agent_orders[start],                                   # (N,)
            )
        )

    return replay_items
