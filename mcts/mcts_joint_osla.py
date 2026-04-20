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


# ─── Action sampling helper ───────────────────────────────────────────────────

def _sample_k_actions(
    rng: chex.Array,
    logits: chex.Array,   # [A_N] joint action logits
    K: int,               # number of actions to sample (static)
    A_N: int,             # joint action space size (static)
) -> tuple[chex.Array, chex.Array]:
    """Sample K joint actions proportional to prior; return (actions [K] int32, probs [K] float32)."""
    probs = jax.nn.softmax(logits)
    replace = K >= A_N
    actions = jax.random.choice(rng, A_N, shape=(K,), replace=replace, p=probs)
    return actions, probs[actions]


# ─── Single simulation step ───────────────────────────────────────────────────

def _run_single_sim(
    carry: SimCarry,
    sim_idx: chex.Array,          # [] int32
    params,
    recurrent_fn,                 # (params, rng, flat_action[1], embedding[1,N,D]) -> (RecurrentFnOutput, next_embedding[1,N,D])
    K: int,                       # static
    A_N: int,                     # static
    max_depth: int,               # static
    gamma: float,
    c_puct: float = 1.25,
) -> SimCarry:
    """Run one MCTS simulation: selection → expansion → backup → record (depth, value)."""
    tree = carry.tree
    rng, expand_rng = jax.random.split(carry.rng)

    # ── 1. Helper: best UCB child index for a given node ─────────────────────
    def _best_ucb(node_idx):
        child_node_idxs = tree.child_node_idx[node_idx]   # [K]
        # For unvisited children (idx=-1), treat as node 0 (safe gather) but with 0 visits
        safe_idxs = jnp.maximum(child_node_idxs, 0)
        child_visits = jnp.where(
            child_node_idxs >= 0,
            tree.visit_counts[safe_idxs].astype(jnp.float32),
            0.0,
        )
        child_q = jnp.where(
            child_node_idxs >= 0,
            tree.value_sum[safe_idxs] / jnp.maximum(child_visits, 1.0),
            0.0,
        )
        prior_probs = tree.child_prior_prob[node_idx]
        parent_visits = tree.visit_counts[node_idx].astype(jnp.float32)
        ucb = compute_ucb_scores(child_q, child_visits, prior_probs, parent_visits, c_puct)
        return jnp.argmax(ucb).astype(jnp.int32)

    # ── 2. Selection via while_loop ───────────────────────────────────────────
    init_bk = _best_ucb(jnp.array(0, jnp.int32))
    init_cidx = tree.child_node_idx[0, init_bk]

    init_sc = SelectCarry(
        node_idx=jnp.array(0, jnp.int32),
        depth=jnp.array(0, jnp.int32),
        path_nodes=jnp.full(max_depth + 1, 0, jnp.int32),
        path_k=jnp.full(max_depth + 1, 0, jnp.int32),
        path_rewards=jnp.zeros(max_depth + 1, jnp.float32),
        best_k=init_bk,
        child_idx=init_cidx,
    )

    def _select_cond(sc: SelectCarry) -> bool:
        return (sc.depth < max_depth) & (sc.child_idx >= 0)

    def _select_body(sc: SelectCarry) -> SelectCarry:
        # Record current node in path
        path_nodes = sc.path_nodes.at[sc.depth].set(sc.node_idx)
        path_k = sc.path_k.at[sc.depth].set(sc.best_k)
        child_reward = tree.reward[sc.child_idx]
        path_rewards = sc.path_rewards.at[sc.depth + 1].set(child_reward)
        # Move to child, compute its best UCB child
        new_bk = _best_ucb(sc.child_idx)
        new_cidx = tree.child_node_idx[sc.child_idx, new_bk]
        return SelectCarry(
            node_idx=sc.child_idx,
            depth=sc.depth + 1,
            path_nodes=path_nodes,
            path_k=path_k,
            path_rewards=path_rewards,
            best_k=new_bk,
            child_idx=new_cidx,
        )

    sc = jax.lax.while_loop(_select_cond, _select_body, init_sc)

    # After selection: sc.node_idx is the deepest expanded node (the parent of the new leaf).
    # sc.best_k is which child to expand. sc.depth is depth of sc.node_idx.
    parent_node = sc.node_idx
    expand_k = sc.best_k
    leaf_depth = sc.depth + 1  # new leaf will be at this depth

    # Record parent in path at sc.depth
    path_nodes = sc.path_nodes.at[sc.depth].set(parent_node)
    path_k = sc.path_k.at[sc.depth].set(expand_k)

    # ── 3. Expansion ─────────────────────────────────────────────────────────
    parent_embedding = tree.embedding[parent_node]          # [N, D]
    flat_action = tree.child_actions[parent_node, expand_k] # [] int32

    rec_out, new_embedding = recurrent_fn(
        params, expand_rng,
        flat_action[None],           # [1] — batched
        parent_embedding[None],      # [1, N, D] — batched
    )
    new_embedding = new_embedding[0]     # [N, D]
    leaf_reward = rec_out.reward[0]      # scalar
    leaf_value = rec_out.value[0]        # scalar
    leaf_prior_logits = rec_out.prior_logits[0]  # [A_N]

    # Sample K children for the new node
    sample_rng, _ = jax.random.split(expand_rng)
    new_child_actions, new_child_probs = _sample_k_actions(sample_rng, leaf_prior_logits, K, A_N)

    new_node_idx = carry.next_free

    # Write new node into tree
    tree = tree.replace(
        embedding=tree.embedding.at[new_node_idx].set(new_embedding),
        reward=tree.reward.at[new_node_idx].set(leaf_reward),
        depth=tree.depth.at[new_node_idx].set(leaf_depth),
        parent=tree.parent.at[new_node_idx].set(parent_node),
        child_actions=tree.child_actions.at[new_node_idx].set(new_child_actions),
        child_node_idx=tree.child_node_idx.at[new_node_idx].set(
            jnp.full(K, -1, jnp.int32)
        ),
        child_prior_prob=tree.child_prior_prob.at[new_node_idx].set(new_child_probs),
    )
    # Link parent → new node
    tree = tree.replace(
        child_node_idx=tree.child_node_idx.at[parent_node, expand_k].set(new_node_idx),
    )

    # Include new leaf in path
    path_nodes = path_nodes.at[leaf_depth].set(new_node_idx)
    path_rewards = sc.path_rewards.at[leaf_depth].set(leaf_reward)

    # ── 4. Backup ─────────────────────────────────────────────────────────────
    # Walk from leaf (leaf_depth) up to root (depth 0), updating visit_counts and value_sum.
    # V at each node = reward_into_that_node + gamma * V_below.
    # We go from leaf upward: at step k=0, update leaf (depth=leaf_depth) with V=leaf_value.
    # At step k=1, update parent (depth=leaf_depth-1) with V = reward[leaf] + gamma * leaf_value.
    # etc.
    def backup_step(k, bcarry):
        btree, V = bcarry
        i = leaf_depth - k   # depth index: leaf_depth, leaf_depth-1, ..., 0
        valid = k <= leaf_depth
        node_i = jnp.where(valid, path_nodes[i], 0)  # safe index (0 when invalid)
        new_vc = btree.visit_counts.at[node_i].add(jnp.where(valid, 1, 0))
        new_vs = btree.value_sum.at[node_i].add(jnp.where(valid, V, 0.0))
        btree = btree.replace(visit_counts=new_vc, value_sum=new_vs)
        # V for next (shallower) node: V_parent = reward_into_current + gamma * V_current
        r_i = path_rewards[jnp.maximum(i, 0)]
        V_new = r_i + gamma * V
        V = jnp.where(valid & (k < leaf_depth), V_new, V)
        return btree, V

    tree, _ = jax.lax.fori_loop(0, max_depth + 1, backup_step, (tree, leaf_value))

    # ── 5. Record OS(λ) data ───────────────────────────────────────────────────
    # Root backup value: cumulative discounted rewards from root to leaf, plus gamma^depth * V_leaf
    def accum_step(k, acc):
        # k=0 → d=1 (first child reward), k=leaf_depth-1 → d=leaf_depth
        d = k + 1
        valid = d <= leaf_depth
        return acc + jnp.where(valid, (gamma ** k) * path_rewards[d], 0.0)

    root_cum_reward = jax.lax.fori_loop(0, max_depth, accum_step, 0.0)
    root_backup_value = root_cum_reward + (gamma ** leaf_depth) * leaf_value

    sim_depths = carry.sim_depths.at[sim_idx].set(leaf_depth)
    sim_values = carry.sim_values.at[sim_idx].set(root_backup_value)

    return SimCarry(
        tree=tree,
        next_free=carry.next_free + 1,
        rng=rng,
        sim_depths=sim_depths,
        sim_values=sim_values,
    )
