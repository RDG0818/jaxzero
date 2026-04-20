"""
Unit tests for mcts/mcts_independent.py and mcts/mcts_joint.py.

Run with:
    conda run -n mazero pytest tests/test_mcts.py -v

Tests are split into:
  - Shared output contract tests (shapes, validity, determinism) for both planners.
  - Independent-planner-specific tests (argmax vs sample mode).
  - Joint-planner-specific tests (helper functions: logit factorization, marginalization).

Module-scoped fixtures compile JAX/JIT once for the whole file to keep
the total run time reasonable.
"""

import dataclasses
import pytest
import jax
import jax.numpy as jnp
import numpy as np

from config import ExperimentConfig, ModelConfig, MCTSConfig, TrainConfig
from model import FlaxMAMuZeroNet
from mcts import MCTSPlanOutput, MCTSIndependentPlanner, MCTSJointPlanner

# ─── Test constants ────────────────────────────────────────────────────────────

N   = 3     # num agents
OBS = 18    # observation dim
A   = 5     # action space size
D   = 32    # hidden state size
S   = 10    # support size


# ─── Module-scoped fixtures (built once, shared across all tests) ──────────────

@pytest.fixture(scope="module")
def test_config():
    """Minimal config designed for fast test execution."""
    return ExperimentConfig(
        train=TrainConfig(
            env_name="MPE_simple_spread_v3",
            num_agents=N,
            num_episodes=100,
            warmup_episodes=10,
            log_interval=10,
            num_actors=1,
            max_episode_steps=25,
            replay_buffer_size=1000,
            replay_buffer_alpha=0.6,
            replay_buffer_beta_start=0.4,
            replay_buffer_beta_frames=1000,
            batch_size=16,
            learning_rate=1e-3,
            param_update_interval=1,
            end_lr_factor=0.1,
            lr_warmup_steps=100,
            value_scale=0.25,
            consistency_scale=2.0,
            consistency_horizon=1,
            gradient_clip_norm=5.0,
            unroll_steps=5,
            n_step=10,
            discount_gamma=0.99,
            wandb_mode="disabled",
            project_name="test",
            checkpoint_dir="checkpoints",
            checkpoint_interval=100,
            num_envs_per_actor=1,
            sync=True,
            ema_decay=0.999,
            num_reanalyze_actors=0,
            reanalyze_batch_size=32,
            debug=False,
            debug_interval=100,
        ),
        model=ModelConfig(
            hidden_state_size=D,
            value_support_size=S,
            reward_support_size=S,
            fc_representation_layers=(D,),
            fc_dynamic_layers=(D,),
            fc_reward_layers=(16,),
            fc_value_layers=(16,),
            fc_policy_layers=(16,),
            attention_type="none",   # skip transformer for speed
            attention_layers=1,
            attention_heads=1,
            dropout_rate=0.0,        # deterministic for reproducibility
            proj_hid=16,
            proj_out=16,
            pred_hid=16,
            pred_out=16,
            use_obs_normalization=False,
        ),
        mcts=MCTSConfig(
            planner_mode="independent",
            num_simulations=8,           # minimum viable for gumbel (>= num_gumbel_samples)
            max_depth_gumbel_search=3,
            num_gumbel_samples=4,
            dirichlet_alpha=0.3,
            dirichlet_fraction=0.25,
            independent_argmax=True,
            use_root_communication=False,
            mcts_rho=0.75,
            mcts_lambda=0.8,
        ),
    )


@pytest.fixture(scope="module")
def model_and_params(test_config):
    """Initialize the world model once for the whole module."""
    net = FlaxMAMuZeroNet(test_config.model, A)
    rng = jax.random.PRNGKey(0)
    dummy_obs = jnp.ones((1, N, OBS))
    params = net.init(rng, dummy_obs)["params"]
    return net, params


@pytest.fixture(scope="module")
def independent_plan_fn(model_and_params, test_config):
    """JIT-compiled plan function for the independent planner (compiled once)."""
    net, _ = model_and_params
    planner = MCTSIndependentPlanner(model=net, config=test_config)
    return jax.jit(planner.plan), planner


@pytest.fixture(scope="module")
def joint_plan_fn(model_and_params, test_config):
    """JIT-compiled plan function for the joint planner (compiled once)."""
    net, _ = model_and_params
    planner = MCTSJointPlanner(model=net, config=test_config)
    return jax.jit(planner.plan), planner


@pytest.fixture
def params(model_and_params):
    _, p = model_and_params
    return p


@pytest.fixture
def obs():
    """Single observation as used during actor rollouts (B=1)."""
    return jnp.ones((1, N, OBS))


# ─── Independent planner tests ────────────────────────────────────────────────

class TestMCTSIndependentPlanner:

    def test_returns_plan_output(self, independent_plan_fn, params, obs):
        plan_fn, _ = independent_plan_fn
        out = plan_fn(params, jax.random.PRNGKey(0), obs)
        assert isinstance(out, MCTSPlanOutput)

    def test_joint_action_shape(self, independent_plan_fn, params, obs):
        plan_fn, _ = independent_plan_fn
        out = plan_fn(params, jax.random.PRNGKey(0), obs)
        assert out.joint_action.shape == (1, N), \
            f"Expected (B=1, N)={(1, N)}, got {out.joint_action.shape}"

    def test_policy_targets_shape(self, independent_plan_fn, params, obs):
        plan_fn, _ = independent_plan_fn
        out = plan_fn(params, jax.random.PRNGKey(0), obs)
        assert out.policy_targets.shape == (1, N, A), \
            f"Expected (B=1, N, A)={(1, N, A)}, got {out.policy_targets.shape}"

    def test_actions_in_valid_range(self, independent_plan_fn, params, obs):
        plan_fn, _ = independent_plan_fn
        out = plan_fn(params, jax.random.PRNGKey(0), obs)
        assert jnp.all(out.joint_action >= 0), "Actions must be non-negative"
        assert jnp.all(out.joint_action < A), f"Actions must be < A={A}"

    def test_policy_targets_sum_to_one(self, independent_plan_fn, params, obs):
        """Gumbel MuZero action_weights are a probability distribution over actions."""
        plan_fn, _ = independent_plan_fn
        out = plan_fn(params, jax.random.PRNGKey(0), obs)
        sums = jnp.sum(out.policy_targets, axis=-1)  # (1, N)
        assert jnp.allclose(sums, jnp.ones((1, N)), atol=1e-5), \
            f"Policy targets must sum to 1 per agent, got: {sums}"

    def test_agent_order_is_sequential(self, independent_plan_fn, params, obs):
        plan_fn, _ = independent_plan_fn
        out = plan_fn(params, jax.random.PRNGKey(0), obs)
        assert jnp.array_equal(out.agent_order, jnp.arange(N)), \
            f"Expected sequential order {jnp.arange(N)}, got {out.agent_order}"

    def test_root_value_shape(self, independent_plan_fn, params, obs):
        plan_fn, _ = independent_plan_fn
        out = plan_fn(params, jax.random.PRNGKey(0), obs)
        assert out.root_value.shape == (1,), \
            f"Expected (B=1,), got {out.root_value.shape}"

    def test_deterministic_with_same_key(self, independent_plan_fn, params, obs):
        plan_fn, _ = independent_plan_fn
        out1 = plan_fn(params, jax.random.PRNGKey(42), obs)
        out2 = plan_fn(params, jax.random.PRNGKey(42), obs)
        assert jnp.array_equal(out1.joint_action, out2.joint_action)
        assert jnp.allclose(out1.policy_targets, out2.policy_targets)

    def test_stochastic_with_different_keys(self, independent_plan_fn, params, obs):
        """Different rng keys should generally produce different search results."""
        plan_fn, _ = independent_plan_fn
        results = [
            plan_fn(params, jax.random.PRNGKey(i), obs).joint_action
            for i in range(10)
        ]
        # At least some results should differ (not all identical)
        all_same = all(jnp.array_equal(results[0], r) for r in results[1:])
        assert not all_same, "10 different keys all produced the same joint action"

    def test_single_agent(self, test_config, model_and_params):
        """N=1 edge case: one agent searching over its own actions."""
        net, _ = model_and_params
        cfg = ExperimentConfig(
            train=dataclasses.replace(test_config.train, num_agents=1),
            model=test_config.model,
            mcts=test_config.mcts,
        )
        planner = MCTSIndependentPlanner(model=net, config=cfg)
        plan_fn = jax.jit(planner.plan)
        obs_1 = jnp.ones((1, 1, OBS))
        params = net.init(jax.random.PRNGKey(0), obs_1)["params"]
        out = plan_fn(params, jax.random.PRNGKey(0), obs_1)
        assert out.joint_action.shape == (1, 1)   # (B=1, N=1)
        assert out.policy_targets.shape == (1, 1, A)
        assert int(out.joint_action[0, 0]) < A

    def test_argmax_mode(self, model_and_params, test_config):
        """independent_argmax=True: other agents fixed to argmax(prior)."""
        net, params = model_and_params
        cfg = ExperimentConfig(
            train=test_config.train,
            model=test_config.model,
            mcts=MCTSConfig(**{**test_config.mcts.__dict__, "independent_argmax": True}),
        )
        planner = MCTSIndependentPlanner(model=net, config=cfg)
        out = jax.jit(planner.plan)(params, jax.random.PRNGKey(0), jnp.ones((1, N, OBS)))
        assert out.joint_action.shape == (1, N)
        assert jnp.all(out.joint_action < A)

    def test_sample_mode(self, model_and_params, test_config):
        """independent_argmax=False: other agents sampled from their policy."""
        net, params = model_and_params
        cfg = ExperimentConfig(
            train=test_config.train,
            model=test_config.model,
            mcts=MCTSConfig(**{**test_config.mcts.__dict__, "independent_argmax": False}),
        )
        planner = MCTSIndependentPlanner(model=net, config=cfg)
        out = jax.jit(planner.plan)(params, jax.random.PRNGKey(0), jnp.ones((1, N, OBS)))
        assert out.joint_action.shape == (1, N)
        assert jnp.all(out.joint_action < A)


# ─── Joint planner tests ──────────────────────────────────────────────────────

class TestMCTSJointPlanner:

    def test_returns_plan_output(self, joint_plan_fn, params, obs):
        plan_fn, _ = joint_plan_fn
        out = plan_fn(params, jax.random.PRNGKey(0), obs)
        assert isinstance(out, MCTSPlanOutput)

    def test_joint_action_shape(self, joint_plan_fn, params, obs):
        plan_fn, _ = joint_plan_fn
        out = plan_fn(params, jax.random.PRNGKey(0), obs)
        assert out.joint_action.shape == (1, N)

    def test_policy_targets_shape(self, joint_plan_fn, params, obs):
        plan_fn, _ = joint_plan_fn
        out = plan_fn(params, jax.random.PRNGKey(0), obs)
        assert out.policy_targets.shape == (1, N, A)

    def test_actions_in_valid_range(self, joint_plan_fn, params, obs):
        plan_fn, _ = joint_plan_fn
        out = plan_fn(params, jax.random.PRNGKey(0), obs)
        assert jnp.all(out.joint_action >= 0)
        assert jnp.all(out.joint_action < A)

    def test_policy_targets_sum_to_one(self, joint_plan_fn, params, obs):
        """Marginal policy targets should sum to 1 per agent."""
        plan_fn, _ = joint_plan_fn
        out = plan_fn(params, jax.random.PRNGKey(0), obs)
        sums = jnp.sum(out.policy_targets, axis=-1)  # (1, N)
        assert jnp.allclose(sums, jnp.ones((1, N)), atol=1e-5), \
            f"Marginal policy targets must sum to 1 per agent, got: {sums}"

    def test_root_value_shape(self, joint_plan_fn, params, obs):
        plan_fn, _ = joint_plan_fn
        out = plan_fn(params, jax.random.PRNGKey(0), obs)
        assert out.root_value.shape == (1,), \
            f"Expected (B=1,), got {out.root_value.shape}"

    def test_deterministic_with_same_key(self, joint_plan_fn, params, obs):
        plan_fn, _ = joint_plan_fn
        out1 = plan_fn(params, jax.random.PRNGKey(7), obs)
        out2 = plan_fn(params, jax.random.PRNGKey(7), obs)
        assert jnp.array_equal(out1.joint_action, out2.joint_action)

    def test_stochastic_with_different_keys(self, joint_plan_fn, params, obs):
        plan_fn, _ = joint_plan_fn
        results = [
            plan_fn(params, jax.random.PRNGKey(i), obs).joint_action
            for i in range(10)
        ]
        all_same = all(jnp.array_equal(results[0], r) for r in results[1:])
        assert not all_same, "10 different keys all produced the same joint action"

    def test_agent_order_is_sequential(self, joint_plan_fn, params, obs):
        plan_fn, _ = joint_plan_fn
        out = plan_fn(params, jax.random.PRNGKey(0), obs)
        assert jnp.array_equal(out.agent_order, jnp.arange(N))

    # ── _logits_to_joint_logits ───────────────────────────────────────────────

    def test_logits_to_joint_shape(self, joint_plan_fn):
        _, planner = joint_plan_fn
        logits = jnp.zeros((1, N, A))
        joint = planner._logits_to_joint_logits(logits)
        assert joint.shape == (1, A ** N), \
            f"Expected (1, A^N)=(1, {A**N}), got {joint.shape}"

    def test_logits_to_joint_uniform_input(self, joint_plan_fn):
        """Uniform per-agent logits should produce uniform joint log-probs."""
        _, planner = joint_plan_fn
        logits = jnp.zeros((1, N, A))   # log-uniform after softmax
        joint = planner._logits_to_joint_logits(logits)
        # All joint log-probs should be identical.
        assert jnp.allclose(joint, joint[0, 0] * jnp.ones_like(joint), atol=1e-5)

    def test_logits_to_joint_sums_to_one(self, joint_plan_fn):
        """exp(joint_logits) should be a valid probability distribution."""
        _, planner = joint_plan_fn
        logits = jax.random.normal(jax.random.PRNGKey(0), (1, N, A))
        joint = planner._logits_to_joint_logits(logits)
        total_prob = jnp.sum(jnp.exp(joint))
        assert jnp.allclose(total_prob, 1.0, atol=1e-5), \
            f"Joint probabilities must sum to 1, got {total_prob}"

    def test_logits_to_joint_batch(self, joint_plan_fn):
        """Should work for B > 1."""
        _, planner = joint_plan_fn
        logits = jnp.zeros((4, N, A))
        joint = planner._logits_to_joint_logits(logits)
        assert joint.shape == (4, A ** N)

    # ── _joint_policy_to_marginal ─────────────────────────────────────────────

    def test_marginal_shape(self, joint_plan_fn):
        _, planner = joint_plan_fn
        joint = jnp.ones((1, A ** N)) / (A ** N)
        marginals = planner._joint_policy_to_marginal(joint)
        assert marginals.shape == (1, N, A), \
            f"Expected (1, N, A)=(1, {N}, {A}), got {marginals.shape}"

    def test_marginals_sum_to_one(self, joint_plan_fn):
        _, planner = joint_plan_fn
        # Use a non-uniform joint to make this a meaningful test.
        key = jax.random.PRNGKey(1)
        raw = jax.random.uniform(key, (1, A ** N))
        joint = raw / raw.sum()
        marginals = planner._joint_policy_to_marginal(joint)
        sums = jnp.sum(marginals, axis=-1)  # (1, N)
        assert jnp.allclose(sums, jnp.ones((1, N)), atol=1e-5), \
            f"Each agent's marginal must sum to 1, got: {sums}"

    def test_uniform_joint_gives_uniform_marginals(self, joint_plan_fn):
        """For a uniform joint distribution, each agent's marginal should be 1/A."""
        _, planner = joint_plan_fn
        joint = jnp.ones((1, A ** N)) / (A ** N)
        marginals = planner._joint_policy_to_marginal(joint)
        expected = jnp.full((1, N, A), 1.0 / A)
        assert jnp.allclose(marginals, expected, atol=1e-5)

    def test_factorization_roundtrip(self, joint_plan_fn):
        """
        Roundtrip: per-agent logits → joint distribution → marginals.

        Under the independence assumption, marginalizing the joint should
        exactly recover each agent's individual softmax distribution.
        """
        _, planner = joint_plan_fn
        logits = jax.random.normal(jax.random.PRNGKey(2), (1, N, A))
        joint_logits = planner._logits_to_joint_logits(logits)
        joint_probs = jax.nn.softmax(joint_logits, axis=-1)
        marginals = planner._joint_policy_to_marginal(joint_probs)  # (1, N, A)

        per_agent_probs = jax.nn.softmax(logits, axis=-1)  # (1, N, A)
        assert jnp.allclose(marginals, per_agent_probs, atol=1e-5), \
            "Marginalizing the factored joint should recover per-agent softmax distributions"

    def test_single_agent_joint(self, test_config, model_and_params):
        """N=1: joint action space is just A, no combinatorics needed."""
        net, _ = model_and_params
        cfg = ExperimentConfig(
            train=dataclasses.replace(test_config.train, num_agents=1),
            model=test_config.model,
            mcts=test_config.mcts,
        )
        planner = MCTSJointPlanner(model=net, config=cfg)
        assert planner.joint_action_shape == (A,)

        plan_fn = jax.jit(planner.plan)
        obs_1 = jnp.ones((1, 1, OBS))
        params = net.init(jax.random.PRNGKey(0), obs_1)["params"]
        out = plan_fn(params, jax.random.PRNGKey(0), obs_1)
        assert out.joint_action.shape == (1, 1)   # (B=1, N=1)
        assert int(out.joint_action[0, 0]) < A


# ─── MCTSConfig field tests ───────────────────────────────────────────────────

def test_mcts_config_has_osla_fields():
    from config import MCTSConfig
    cfg = MCTSConfig(
        planner_mode="joint",
        num_simulations=8,
        max_depth_gumbel_search=3,
        num_gumbel_samples=4,
        dirichlet_alpha=0.3,
        dirichlet_fraction=0.25,
        independent_argmax=True,
        use_root_communication=False,
        mcts_rho=0.75,
        mcts_lambda=0.8,
    )
    assert cfg.mcts_rho == 0.75
    assert cfg.mcts_lambda == 0.8


# ─── OS(λ) value aggregation tests ───────────────────────────────────────────

class TestComputeOslaValue:
    """Tests for the OS(λ) value aggregation function."""

    def test_all_same_depth_rho_one_equals_mean(self):
        """When all sims reach same depth, OS(λ) with rho=1.0 = plain mean."""
        from mcts.mcts_joint_osla import compute_osla_value
        depths = jnp.array([1, 1, 1, 1], dtype=jnp.int32)
        values = jnp.array([0.0, 0.0, 0.0, 1.0], dtype=jnp.float32)
        # rho=1.0: keep all 4, mean = 0.25
        v = compute_osla_value(depths, values, rho=1.0, lam=0.8)
        assert jnp.allclose(v, 0.25, atol=1e-4)

    def test_top_rho_amplifies_rare_wins(self):
        """rho=0.75 keeps top 25% (1 out of 4), result = the win value (1.0)."""
        from mcts.mcts_joint_osla import compute_osla_value
        depths = jnp.array([1, 1, 1, 1], dtype=jnp.int32)
        values = jnp.array([0.0, 0.0, 0.0, 1.0], dtype=jnp.float32)
        # rho=0.75: keep top 25% = top 1 = value 1.0; weight = lambda^1 = 0.8
        v = compute_osla_value(depths, values, rho=0.75, lam=0.8)
        assert jnp.allclose(v, 1.0, atol=1e-4)

    def test_depth_weighting_discounts_deeper_sims(self):
        """Deeper sims get less weight (lam^depth); verifies weighted mean is computed correctly."""
        from mcts.mcts_joint_osla import compute_osla_value
        depths = jnp.array([1, 5], dtype=jnp.int32)
        values = jnp.array([1.0, 0.5], dtype=jnp.float32)
        # rho=1.0: keep both. weights = [0.8^1, 0.8^5] = [0.8, 0.32768]
        # weighted mean = (0.8*1.0 + 0.32768*0.5) / (0.8 + 0.32768)
        w1, w2 = 0.8 ** 1, 0.8 ** 5
        expected = (w1 * 1.0 + w2 * 0.5) / (w1 + w2)
        v = compute_osla_value(depths, values, rho=1.0, lam=0.8)
        assert jnp.allclose(v, expected, atol=1e-4)

    def test_jit_compatible(self):
        """Must be JIT-compilable."""
        from mcts.mcts_joint_osla import compute_osla_value
        fn = jax.jit(compute_osla_value, static_argnames=("rho", "lam"))
        depths = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
        values = jnp.array([0.1, 0.5, 0.2, 0.9], dtype=jnp.float32)
        v = fn(depths, values, rho=0.75, lam=0.8)
        assert v.shape == ()
        assert jnp.isfinite(v)


# ─── OSLATree dataclass and UCB helper tests ──────────────────────────────────

class TestOSLAHelpers:

    def test_compute_ucb_prefers_unvisited(self):
        """Unvisited children (visit_count=0) should have higher UCB than visited ones."""
        from mcts.mcts_joint_osla import compute_ucb_scores
        child_visits = jnp.array([5.0, 0.0, 3.0, 0.0])
        child_q = jnp.array([0.5, 0.0, 0.3, 0.0])
        prior_probs = jnp.array([0.25, 0.25, 0.25, 0.25])
        parent_visits = jnp.array(8.0)
        ucb = compute_ucb_scores(child_q, child_visits, prior_probs, parent_visits, c_puct=1.25)
        # Unvisited children (indices 1, 3) should have higher UCB than visited ones
        assert ucb[1] > ucb[0]
        assert ucb[3] > ucb[2]

    def test_compute_ucb_shape(self):
        from mcts.mcts_joint_osla import compute_ucb_scores
        K = 10
        ucb = compute_ucb_scores(
            jnp.zeros(K), jnp.zeros(K), jnp.ones(K) / K,
            jnp.array(1.0), c_puct=1.25
        )
        assert ucb.shape == (K,)

    def test_dataclasses_are_pytrees(self):
        """OSLATree, SimCarry, SelectCarry must be registered as JAX pytrees."""
        from mcts.mcts_joint_osla import OSLATree, SimCarry, SelectCarry
        # chex.dataclass auto-registers as pytree; verify leaves/treedef work
        tree = OSLATree(
            visit_counts=jnp.zeros(5, jnp.int32),
            value_sum=jnp.zeros(5),
            reward=jnp.zeros(5),
            embedding=jnp.zeros((5, 3, 32)),
            depth=jnp.zeros(5, jnp.int32),
            parent=jnp.full(5, -1, jnp.int32),
            child_actions=jnp.zeros((5, 4), jnp.int32),
            child_node_idx=jnp.full((5, 4), -1, jnp.int32),
            child_prior_prob=jnp.zeros((5, 4)),
        )
        leaves, treedef = jax.tree_util.tree_flatten(tree)
        assert len(leaves) == 9  # 9 fields
        tree2 = treedef.unflatten(leaves)
        assert jnp.array_equal(tree2.visit_counts, tree.visit_counts)


# ─── _sample_k_actions tests ──────────────────────────────────────────────────

class TestSampleKActions:

    def test_output_shapes(self):
        from mcts.mcts_joint_osla import _sample_k_actions
        rng = jax.random.PRNGKey(0)
        actions, probs = _sample_k_actions(rng, jnp.zeros(729), K=10, A_N=729)
        assert actions.shape == (10,)
        assert probs.shape == (10,)

    def test_actions_in_range(self):
        from mcts.mcts_joint_osla import _sample_k_actions
        rng = jax.random.PRNGKey(1)
        actions, probs = _sample_k_actions(rng, jnp.zeros(729), K=10, A_N=729)
        assert jnp.all(actions >= 0) and jnp.all(actions < 729)

    def test_probs_are_subset_of_softmax(self):
        """Returned probs must equal softmax(logits)[actions]."""
        from mcts.mcts_joint_osla import _sample_k_actions
        logits = jax.random.normal(jax.random.PRNGKey(2), (25,))
        actions, probs = _sample_k_actions(jax.random.PRNGKey(3), logits, K=5, A_N=25)
        expected_probs = jax.nn.softmax(logits)[actions]
        assert jnp.allclose(probs, expected_probs, atol=1e-6)

    def test_k_ge_an_uses_replacement(self):
        """K >= A_N must not crash (uses replacement)."""
        from mcts.mcts_joint_osla import _sample_k_actions
        actions, probs = _sample_k_actions(jax.random.PRNGKey(0), jnp.zeros(3), K=10, A_N=3)
        assert actions.shape == (10,)


# ─── _run_single_sim backup correctness tests ────────────────────────────────

class TestRunSingleSimBackup:
    """Verify _run_single_sim produces correct backup values."""

    def _make_fake_recurrent_fn(self, fixed_reward: float, fixed_value: float, A_N: int, N: int, D: int):
        """Returns a recurrent_fn that always outputs fixed reward and value."""
        import mctx
        def fake_recurrent_fn(params, rng, flat_action, embedding):
            B = flat_action.shape[0]
            return (
                mctx.RecurrentFnOutput(
                    reward=jnp.full((B,), fixed_reward),
                    discount=jnp.ones((B,)),
                    prior_logits=jnp.zeros((B, A_N)),
                    value=jnp.full((B,), fixed_value),
                ),
                embedding,  # pass through embedding unchanged
            )
        return fake_recurrent_fn

    def test_root_backup_value_single_step(self):
        """After 1 sim reaching depth 1: root_backup_value = r + gamma * V."""
        from mcts.mcts_joint_osla import _run_single_sim, OSLATree, SimCarry, _sample_k_actions

        K, A_N, N, D, max_depth, gamma = 4, 25, 2, 8, 3, 0.99
        r, v = 0.5, 1.0  # fixed reward and value from the fake model

        # Build a minimal root node with K sampled children
        rng = jax.random.PRNGKey(0)
        root_logits = jnp.zeros(A_N)
        child_actions, child_probs = _sample_k_actions(rng, root_logits, K, A_N)

        max_nodes = 5
        tree = OSLATree(
            visit_counts=jnp.array([1] + [0] * (max_nodes - 1), jnp.int32),
            value_sum=jnp.zeros(max_nodes),
            reward=jnp.zeros(max_nodes),
            embedding=jnp.zeros((max_nodes, N, D)),
            depth=jnp.zeros(max_nodes, jnp.int32),
            parent=jnp.full(max_nodes, -1, jnp.int32),
            child_actions=jnp.zeros((max_nodes, K), jnp.int32).at[0].set(child_actions),
            child_node_idx=jnp.full((max_nodes, K), -1, jnp.int32),
            child_prior_prob=jnp.zeros((max_nodes, K)).at[0].set(child_probs),
        )
        carry = SimCarry(
            tree=tree,
            next_free=jnp.array(1, jnp.int32),
            rng=jax.random.PRNGKey(1),
            sim_depths=jnp.zeros(1, jnp.int32),
            sim_values=jnp.zeros(1, jnp.float32),
        )

        fake_rf = self._make_fake_recurrent_fn(r, v, A_N, N, D)
        result = _run_single_sim(carry, jnp.array(0), None, fake_rf, K, A_N, max_depth, gamma)

        # Root backup value = r + gamma * v
        expected_root_val = r + gamma * v
        assert jnp.allclose(result.sim_values[0], expected_root_val, atol=1e-4), \
            f"Expected {expected_root_val:.4f}, got {result.sim_values[0]:.4f}"

        # Root visit count should be 2 (was 1, got 1 from backup)
        assert result.tree.visit_counts[0] == 2

        # New node (index 1) should have visit count 1
        assert result.tree.visit_counts[1] == 1
