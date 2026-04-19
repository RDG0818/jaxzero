# mcts/mcts_joint_osla.py

import functools

import chex
import jax
import jax.numpy as jnp

from model import FlaxMAMuZeroNet
import utils.transforms as utils
from config import ExperimentConfig
from mcts.base import MCTSPlanner, MCTSPlanOutput


def compute_osla_value(
    sim_depths: chex.Array,    # [K] int32 — simulation leaf depths
    sim_values: chex.Array,    # [K] float32 — backup values from root
    rho: float = 0.75,         # keep top (1-rho) fraction; rho=0.75 → top 25%
    lam: float = 0.8,          # depth decay weight
) -> chex.Array:
    """
    OS(λ) value estimate: weighted mean of the top (1-rho) quantile of simulations,
    each weighted by lambda^depth.

    rho=0.75 keeps the top 25% of simulations by backup value — amplifying rare
    high-value paths (e.g. wins) that mean backup would dilute.
    """
    K = sim_values.shape[0]
    # Keep the top (1-rho) fraction of simulations; clamp to [1, K].
    # rho=0.75 → keep top 25%; rho=1.0 → keep top 0% → clamped to all K.
    k_top = K if rho == 1.0 else max(1, int((1.0 - rho) * K))

    # Sort simulations by backup value descending; take top k_top indices.
    sorted_idx = jnp.argsort(sim_values)[::-1]  # descending
    top_idx = sorted_idx[:k_top]                 # [k_top]

    top_values = sim_values[top_idx]   # [k_top]
    top_depths = sim_depths[top_idx]   # [k_top]
    weights = lam ** top_depths.astype(jnp.float32)  # [k_top]

    return (top_values * weights).sum() / (weights.sum() + 1e-8)
