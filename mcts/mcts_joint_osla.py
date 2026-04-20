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


# ─── Tree data structure ──────────────────────────────────────────────────────

@chex.dataclass
class OSLATree:
    """Static-shaped search tree for one environment instance (no batch dim).

    max_nodes = num_simulations + 1 (root + 1 new leaf per simulation).
    K = num_gumbel_samples (sampled joint actions per node).
    """
    visit_counts:     chex.Array  # [max_nodes] int32
    value_sum:        chex.Array  # [max_nodes] float32
    reward:           chex.Array  # [max_nodes] float32 — reward to enter this node
    embedding:        chex.Array  # [max_nodes, N, D] float32
    depth:            chex.Array  # [max_nodes] int32
    parent:           chex.Array  # [max_nodes] int32 (-1 = root)
    child_actions:    chex.Array  # [max_nodes, K] int32 — flat joint action indices
    child_node_idx:   chex.Array  # [max_nodes, K] int32 (-1 = not yet expanded)
    child_prior_prob: chex.Array  # [max_nodes, K] float32


@chex.dataclass
class SimCarry:
    """Outer fori_loop state across all simulations."""
    tree:       OSLATree
    next_free:  chex.Array   # [] int32 — next available node index
    rng:        chex.Array   # PRNGKey
    sim_depths: chex.Array   # [num_simulations] int32
    sim_values: chex.Array   # [num_simulations] float32


@chex.dataclass
class SelectCarry:
    """Inner while_loop state during tree selection."""
    node_idx:     chex.Array  # [] int32 — current node
    depth:        chex.Array  # [] int32
    path_nodes:   chex.Array  # [max_depth+1] int32
    path_k:       chex.Array  # [max_depth+1] int32
    path_rewards: chex.Array  # [max_depth+1] float32 — reward[path_nodes[i]]
    best_k:       chex.Array  # [] int32 — UCB-best child of current node
    child_idx:    chex.Array  # [] int32 — child_node_idx[node, best_k]


# ─── UCB helper ───────────────────────────────────────────────────────────────

def compute_ucb_scores(
    child_q:       chex.Array,  # [K] float32 — mean Q-value per child
    child_visits:  chex.Array,  # [K] float32 — visit counts (0 = unvisited)
    prior_probs:   chex.Array,  # [K] float32 — prior probabilities
    parent_visits: chex.Array,  # [] float32 — total visits at parent
    c_puct: float = 1.25,
) -> chex.Array:
    """PUCT formula: Q(a) + c_puct * P(a) * sqrt(N_parent + 1) / (1 + N(a))."""
    exploration = c_puct * prior_probs * jnp.sqrt(parent_visits + 1.0) / (1.0 + child_visits)
    return child_q + exploration
