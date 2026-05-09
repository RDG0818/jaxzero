import os
import time
import numpy as np
import ray
from jaxzero.config import MAZeroConfig


@ray.remote(num_gpus=1)
class LearnerActor:
    """Owns model params + optimizer on GPU. Trains and serves params to DataActors."""

    def __init__(self, config: MAZeroConfig, replay_buffer_actor):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ.setdefault("OMP_NUM_THREADS", "4")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
        os.environ.setdefault("MKL_NUM_THREADS", "4")

        import jax
        import jax.numpy as jnp
        import optax
        from jaxzero.model.networks import MAMuZeroNet
        from jaxzero.train import make_update_fn
        from jaxzero.reanalyze import ReanalyzeWorker, BatchData
        self._BatchData = BatchData

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
        self._compiled = False

    def get_params(self):
        """Return params as numpy — safe to send across Ray process boundaries."""
        return self._jax.tree_util.tree_map(lambda x: np.array(x), self.params)

    def run_training_loop(self, num_steps: int):
        """Run up to num_steps gradient updates. Returns metrics dict or None if buffer empty."""
        cfg = self.config
        losses, r_losses, v_losses, p_losses, c_losses = [], [], [], [], []
        grad_norms, entropies = [], []
        t_buf, t_reanalyze, t_update = 0.0, 0.0, 0.0
        t_wall_start = time.perf_counter()

        for _ in range(num_steps):
            if self.step % cfg.target_model_interval == 0:
                self.target_params = self.params

            beta = min(
                1.0,
                cfg.priority_beta_start
                + (1.0 - cfg.priority_beta_start) * self.step / max(1, cfg.training_steps),
            )

            _t0 = time.perf_counter()
            if cfg.num_reanalyze_actors > 0:
                raw = ray.get(self.replay_buffer.prepare_batch.remote(cfg.batch_size, beta))
                t_buf += time.perf_counter() - _t0
                if raw is None:
                    break
                batch = self._BatchData(**raw)
            else:
                ctx = ray.get(self.replay_buffer.prepare_batch_context.remote(cfg.batch_size, beta))
                t_buf += time.perf_counter() - _t0
                if ctx is None:
                    break
                _t0 = time.perf_counter()
                batch = self.reanalyze_worker.make_batch(ctx, self.target_params)
                t_reanalyze += time.perf_counter() - _t0

            if not self._compiled:
                print("[LearnerActor] compiling update_fn...", flush=True)
            _t0 = time.perf_counter()
            loss, grads, aux, priorities = self.update_fn(self.params, batch)
            grad_norm = self._optax.global_norm(grads)
            self._jax.effects_barrier()
            t_update += time.perf_counter() - _t0
            if not self._compiled:
                print(f"[LearnerActor] compilation done ({t_update:.1f}s)", flush=True)
                self._compiled = True
            grad_norms.append(float(grad_norm))

            updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
            self.params = self._optax.apply_updates(self.params, updates)
            self.replay_buffer.update_priorities.remote(batch.indices, priorities)

            losses.append(float(loss))
            r_losses.append(float(aux["reward_loss"]))
            v_losses.append(float(aux["value_loss"]))
            p_losses.append(float(aux["policy_loss"]))
            c_losses.append(float(aux["consistency_loss"]))
            entropies.append(float(aux["policy_entropy"]))
            self.step += 1

        if not losses:
            return None

        total_t = t_buf + t_reanalyze + t_update
        wall_t = time.perf_counter() - t_wall_start
        return {
            "step": self.step,
            "total_loss": float(np.mean(losses)),
            "reward_loss": float(np.mean(r_losses)),
            "value_loss": float(np.mean(v_losses)),
            "policy_loss": float(np.mean(p_losses)),
            "consistency_loss": float(np.mean(c_losses)),
            "grad_norm": float(np.mean(grad_norms)),
            "policy_entropy": float(np.mean(entropies)),
            # timing (seconds over this call)
            "t_buf": t_buf,
            "t_reanalyze": t_reanalyze,
            "t_update": t_update,
            "gpu_frac": t_update / total_t if total_t > 0 else 0.0,
            "steps_per_sec": len(losses) / wall_t if wall_t > 0 else 0.0,
        }
