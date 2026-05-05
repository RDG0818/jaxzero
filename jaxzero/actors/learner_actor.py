import os
import numpy as np
import ray
from jaxzero.config import MAZeroConfig


@ray.remote(num_gpus=1)
class LearnerActor:
    """Owns model params + optimizer on GPU. Trains and serves params to DataActors."""

    def __init__(self, config: MAZeroConfig, replay_buffer_actor):
        # Must set BEFORE any JAX import
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ.setdefault("OMP_NUM_THREADS", "4")

        import jax
        import jax.numpy as jnp
        import optax
        from jaxzero.model.networks import MAMuZeroNet
        from jaxzero.train import make_update_fn
        from jaxzero.reanalyze import ReanalyzeWorker

        self.config = config
        self.replay_buffer = replay_buffer_actor
        self.step = 0
        self._jax = jax
        self._optax = optax

        net = MAMuZeroNet(config=config)
        obs_init = jnp.ones((1, config.num_agents, config.obs_size))
        rng = jax.random.PRNGKey(config.seed)
        self.params = net.init(rng, obs_init)
        self.params = jax.device_put(self.params)

        optimizer = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.learning_rate, eps=config.adam_eps),
        )
        self.opt_state = optimizer.init(self.params)
        self.optimizer = optimizer
        self.update_fn = make_update_fn(net, config)
        self.reanalyze_worker = ReanalyzeWorker(config=config, model=net)
        self.target_params = self.params

    def get_params(self):
        """Return params as numpy — safe to send across Ray process boundaries."""
        return self._jax.tree_util.tree_map(lambda x: np.array(x), self.params)

    def run_training_loop(self, num_steps: int):
        """Run up to num_steps gradient updates. Returns metrics dict or None if buffer empty."""
        cfg = self.config
        losses, r_losses, v_losses, p_losses = [], [], [], []

        for _ in range(num_steps):
            if self.step % cfg.target_model_interval == 0:
                self.target_params = self.params

            beta = min(
                1.0,
                cfg.priority_beta_start
                + (1.0 - cfg.priority_beta_start) * self.step / max(1, cfg.training_steps),
            )
            ctx = ray.get(self.replay_buffer.prepare_batch_context.remote(cfg.batch_size, beta))
            if ctx is None:
                break

            # Targets are computed using target_params to stabilize bootstrap
            batch = self.reanalyze_worker.make_batch(ctx, self.target_params)
            loss, grads, aux, priorities = self.update_fn(self.params, batch)
            updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
            self.params = self._optax.apply_updates(self.params, updates)
            self.replay_buffer.update_priorities.remote(batch.indices, priorities)

            losses.append(float(loss))
            r_losses.append(float(aux["reward_loss"]))
            v_losses.append(float(aux["value_loss"]))
            p_losses.append(float(aux["policy_loss"]))
            self.step += 1

        if not losses:
            return None
        return {
            "step": self.step,
            "total_loss": float(np.mean(losses)),
            "reward_loss": float(np.mean(r_losses)),
            "value_loss": float(np.mean(v_losses)),
            "policy_loss": float(np.mean(p_losses)),
        }
