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
