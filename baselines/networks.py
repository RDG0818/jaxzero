# baselines/networks.py
"""
Flax modules for IPPO and MAPPO baselines.

ActorCritic     — shared policy + decentralized value head (used by IPPO).
CentralizedCritic — global-state value head (used by MAPPO in addition to
                    the actor from ActorCritic).

All networks use parameter sharing: every agent sees the same weights.
Agent identity is disambiguated by a one-hot ID vector appended to the obs.
"""

import flax.linen as nn
import jax.numpy as jnp
from typing import Sequence


class ActorCritic(nn.Module):
    """
    Shared actor-critic network for IPPO (and the actor branch of MAPPO).

    Input obs shape: (B, N, obs_size + N)   [obs concatenated with agent one-hot]
    Output:
        logits: (B, N, action_size)          [per-agent policy logits]
        values: (B, N)                       [per-agent value estimates]
    """
    action_size: int
    hidden_sizes: Sequence[int] = (64, 64)

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> tuple:
        # obs: (..., obs_dim)  — works for any leading batch dims
        x = obs
        for h in self.hidden_sizes:
            x = nn.Dense(h)(x)
            x = nn.tanh(x)

        logits = nn.Dense(self.action_size)(x)   # (..., A)
        values = nn.Dense(1)(x).squeeze(-1)      # (...,)
        return logits, values


class CentralizedCritic(nn.Module):
    """
    Centralized value function for MAPPO.

    Takes the global state (all agents' observations concatenated) and
    produces a single scalar team value estimate.

    Input global_state shape: (B, N * obs_size)
    Output value: (B,)
    """
    hidden_sizes: Sequence[int] = (64, 64)

    @nn.compact
    def __call__(self, global_state: jnp.ndarray) -> jnp.ndarray:
        x = global_state
        for h in self.hidden_sizes:
            x = nn.Dense(h)(x)
            x = nn.tanh(x)
        return nn.Dense(1)(x).squeeze(-1)  # (B,)
