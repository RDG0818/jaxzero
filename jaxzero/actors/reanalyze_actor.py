import os
import numpy as np
import ray
from jaxzero.config import MAZeroConfig


@ray.remote(num_cpus=1)
class ReanalyzeActor:
    """Re-runs MCTS on stored games to refresh policy/value targets."""

    def __init__(self, actor_id: int, config: MAZeroConfig, learner_actor, replay_buffer_actor):
        # Match DataActor env setup
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
        os.environ.setdefault("MKL_NUM_THREADS", "2")

        import jax  # noqa: F401
        from jaxzero.model.networks import MAMuZeroNet
        from jaxzero.mcts.sampled_mcts import SampledMCTS

        self.actor_id = actor_id
        self.config = config
        self.learner = learner_actor
        self.replay_buffer = replay_buffer_actor
        self.np_rng = np.random.default_rng(actor_id * 997 + 11)

        net = MAMuZeroNet(config=config)
        self.mcts = SampledMCTS(config=config, model=net)

        # Sync params
        self.params = ray.get(learner_actor.get_params.remote())
        self._param_future = None
        self.steps_since_sync = 0

    def run_reanalyze(self) -> int:
        """Sample batch from buffer, re-run MCTS, and push back results."""
        # 1. Fetch batch info
        ctx = ray.get(self.replay_buffer.sample_for_reanalyze.remote(self.config.reanalyze_batch_size))
        if ctx is None:
            return 0
        
        # 2. Sync params if needed (async)
        if self._param_future is not None:
            ready, _ = ray.wait([self._param_future], timeout=0)
            if ready:
                self.params = ray.get(self._param_future)
                self._param_future = None
                self.steps_since_sync = 0
        
        if self._param_future is None and self.steps_since_sync >= 10:
            self._param_future = self.learner.get_params.remote()

        # 3. Perform MCTS on each sampled position
        # We can batch these into one mcts.search call!
        obs_batch = np.stack([c[2] for c in ctx])
        legal_batch = np.stack([c[3] for c in ctx])
        
        results = self.mcts.search(self.params, obs_batch, legal_batch, self.np_rng)
        
        # 4. Format and push back
        update_data = []
        for i, (game_idx, pos, _, _) in enumerate(ctx):
            # Pad/format stats same as _store_search_stats in train.py
            visits = results.sampled_visit_counts[i]
            actions_pool = results.sampled_actions[i]
            K_full = self.config.sampled_action_times
            K_actual = len(visits)
            pad = K_full - K_actual
            
            if pad > 0:
                N_agents = actions_pool.shape[1]
                actions_padded = np.concatenate(
                    [actions_pool, np.zeros((pad, N_agents), dtype=actions_pool.dtype)], axis=0
                )
                visits_padded = np.concatenate([visits, np.zeros(pad, dtype=visits.dtype)])
                qvals_padded = np.concatenate(
                    [results.sampled_qvalues[i], np.zeros(pad, dtype=np.float32)]
                )
                mask_padded = np.concatenate(
                    [np.ones(K_actual, dtype=bool), np.zeros(pad, dtype=bool)]
                )
            else:
                actions_padded = actions_pool
                visits_padded = visits
                qvals_padded = results.sampled_qvalues[i]
                mask_padded = np.ones(K_full, dtype=bool)

            update_data.append({
                "game_idx": game_idx,
                "pos": pos,
                "policy": visits_padded.astype(np.float32) / visits_padded.sum(),
                "qvalues": qvals_padded.astype(np.float32),
                "actions": actions_padded,
                "mask": mask_padded,
                "root_value": float(results.root_value[i])
            })
        
        self.replay_buffer.update_reanalyzed_stats.remote(update_data)
        self.steps_since_sync += 1
        return len(update_data)
