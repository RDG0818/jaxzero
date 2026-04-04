import os
import numpy as np

import ray

from config import CONFIG
from utils.logging_utils import logger


def make_train_step(model, optimizer, value_support, reward_support):
    """
    Returns a JIT-compiled training step function.

    Captures model, optimizer, and supports in a closure so JIT only traces
    once — no static_argnames needed.

    JAX is imported lazily here; this function is only ever called from
    LearnerActor.__init__ after JAX has already been imported in that process.
    """
    import jax
    import jax.numpy as jnp
    import optax
    from utils.utils import scalar_to_support, support_to_scalar

    U = CONFIG.train.unroll_steps
    value_scale = CONFIG.train.value_scale
    consistency_scale = CONFIG.train.consistency_scale

    def train_step(params, opt_state, batch, weights, rng_key):
        # Pre-compute categorical support targets outside loss_fn so they
        # are constants w.r.t. the gradient — zero gradient flows through them.
        value_target_dist = scalar_to_support(
            batch.value_target.mean(axis=2), value_support
        )   # (B, U+1, Sv)
        reward_target_dist = scalar_to_support(
            batch.reward_target.mean(axis=2), reward_support
        )   # (B, U, Sr)

        def loss_fn(p):
            rng_init, _, rng_unroll = jax.random.split(rng_key, 3)
            unroll_keys = jax.random.split(rng_unroll, U)  # (U, 2)

            # ---- Step 0: initial inference ----
            init_out = model.apply(
                {"params": p}, batch.observation, rngs={"dropout": rng_init}
            )
            hidden = init_out.hidden_state  # (B, N, D)

            # Mean over agents N → (B,)
            p0_loss = optax.softmax_cross_entropy(
                init_out.policy_logits, batch.policy_target[:, 0]
            ).mean(axis=-1)

            # Centralized value → (B,)
            v0_loss = optax.softmax_cross_entropy(
                init_out.value_logits, value_target_dist[:, 0]
            )

            # ---- Steps 1..U: unroll via scan ----
            def scan_step(hidden, inputs):
                ai, ri_dist, pi_target, vi_dist, step_key = inputs

                online_proj = model.apply(
                    {"params": p}, hidden, method=model.project_online
                )
                out = model.apply(
                    {"params": p}, hidden, ai,
                    method=model.recurrent_inference,
                    rngs={"dropout": step_key},
                )
                next_hidden = out.hidden_state

                target_proj = jax.lax.stop_gradient(
                    model.apply(
                        {"params": p}, next_hidden, method=model.project_target
                    )
                )

                ri_loss = optax.softmax_cross_entropy(out.reward_logits, ri_dist)
                pi_loss = optax.softmax_cross_entropy(
                    out.policy_logits, pi_target
                ).mean(axis=-1)
                vi_loss = optax.softmax_cross_entropy(out.value_logits, vi_dist)

                B_, N_, D_ = online_proj.shape
                sim = optax.cosine_similarity(
                    online_proj.reshape(B_ * N_, D_),
                    target_proj.reshape(B_ * N_, D_),
                ).reshape(B_, N_).mean(axis=-1)
                cons_loss = -sim

                return next_hidden, (ri_loss, pi_loss, vi_loss, cons_loss)

            # Transpose to step-major for scan: (B, U, ...) → (U, B, ...)
            xs = (
                jnp.moveaxis(batch.actions, 1, 0),
                jnp.moveaxis(reward_target_dist, 1, 0),
                jnp.moveaxis(batch.policy_target[:, 1:], 1, 0),
                jnp.moveaxis(value_target_dist[:, 1:], 1, 0),
                unroll_keys,
            )
            _, (ri_losses, pi_losses, vi_losses, cons_losses) = jax.lax.scan(
                scan_step, hidden, xs
            )
            # Each: (U, B)

            reward_loss      = ri_losses.mean(axis=0)
            policy_loss      = (p0_loss + pi_losses.sum(axis=0)) / (U + 1)
            value_loss       = (v0_loss + vi_losses.sum(axis=0)) / (U + 1)
            consistency_loss = cons_losses.mean(axis=0)

            loss = (
                reward_loss
                + policy_loss
                + value_loss * value_scale
                + consistency_loss * consistency_scale
            )
            total_loss = (loss * weights).mean()

            td_error = jnp.abs(
                support_to_scalar(init_out.value_logits, value_support)
                - batch.value_target[:, 0].mean(axis=1)
            )

            metrics = {
                "total_loss": total_loss,
                "reward_loss": reward_loss.mean(),
                "policy_loss": policy_loss.mean(),
                "value_loss": value_loss.mean(),
                "consistency_loss": consistency_loss.mean(),
            }
            return total_loss, (metrics, td_error)

        (_, (metrics, td_error)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)
        metrics["grad_norm"] = optax.global_norm(grads)

        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        new_priorities = td_error + 1e-6

        return new_params, new_opt_state, metrics, new_priorities

    return jax.jit(train_step)


@ray.remote(num_gpus=1)
class LearnerActor:
    """
    Trains the MuZero model on GPU.

    Pulls batches from the replay buffer, runs a JIT-compiled training step,
    and serves updated parameters to DataActors on request.
    """

    def __init__(self, obs_size: int, action_size: int, replay_buffer_actor):
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.70"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        os.environ["GLOG_minloglevel"] = "2"

        import jax
        import jax.numpy as jnp
        import optax
        from utils.utils import DiscreteSupport
        from model.model import FlaxMAMuZeroNet

        logger.info(f"(Learner pid={os.getpid()}) Initializing on GPU...")

        self.replay_buffer = replay_buffer_actor
        self.train_step_count = 0
        self.rng_key = jax.random.PRNGKey(0)

        value_support = DiscreteSupport(
            min=-CONFIG.model.value_support_size,
            max=CONFIG.model.value_support_size,
        )
        reward_support = DiscreteSupport(
            min=-CONFIG.model.reward_support_size,
            max=CONFIG.model.reward_support_size,
        )

        model = FlaxMAMuZeroNet(CONFIG.model, action_size)
        dummy_obs = jnp.ones((1, CONFIG.train.num_agents, obs_size))
        self.rng_key, init_key = jax.random.split(self.rng_key)
        self.params = model.init(init_key, dummy_obs)["params"]

        lr = CONFIG.train.learning_rate
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr,
            warmup_steps=CONFIG.train.lr_warmup_steps,
            decay_steps=CONFIG.train.num_episodes - CONFIG.train.lr_warmup_steps,
            end_value=lr * CONFIG.train.end_lr_factor,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(CONFIG.train.gradient_clip_norm),
            optax.adamw(learning_rate=lr_schedule),
        )
        self.opt_state = optimizer.init(self.params)
        self.lr_schedule = lr_schedule

        self.train_step = make_train_step(
            model, optimizer, value_support, reward_support
        )
        logger.info(f"(Learner pid={os.getpid()}) Setup complete.")

    def train(self):
        """Samples a batch, runs one training step, updates priorities.
        Returns a metrics dict, or None if the buffer is empty."""
        import jax

        batch, weights, indices = ray.get(
            self.replay_buffer.sample.remote(CONFIG.train.batch_size)
        )
        if batch is None:
            return None

        self.rng_key, train_key = jax.random.split(self.rng_key)
        jax_batch = jax.tree_util.tree_map(jax.device_put, batch)
        jax_weights = jax.device_put(np.array(weights, dtype=np.float32))

        self.params, self.opt_state, metrics, new_priorities = self.train_step(
            self.params, self.opt_state, jax_batch, jax_weights, train_key
        )
        self.train_step_count += 1

        self.replay_buffer.update_priorities.remote(indices, np.array(new_priorities))

        metrics = {k: float(v) for k, v in metrics.items()}
        metrics["learning_rate"] = float(self.lr_schedule(self.train_step_count))
        return metrics

    def get_params(self):
        return self.params

    def get_train_step_count(self) -> int:
        return self.train_step_count
