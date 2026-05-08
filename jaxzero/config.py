from dataclasses import dataclass


@dataclass(frozen=True)
class MAZeroConfig:
    # Environment
    env_name: str = "3m"
    num_agents: int = 3
    obs_size: int = 0
    action_space_size: int = 0
    stacked_observations: int = 4
    max_episode_steps: int = 100

    # Model
    hidden_state_size: int = 128
    fc_representation_layers: tuple[int, ...] = (128, 128)
    fc_dynamic_layers: tuple[int, ...] = (128, 128)
    fc_reward_layers: tuple[int, ...] = (32,)
    fc_value_layers: tuple[int, ...] = (32,)
    fc_policy_layers: tuple[int, ...] = (32,)
    attention_layers: int = 3
    attention_heads: int = 8
    dropout_rate: float = 0.1
    value_support_size: int = 5
    reward_support_size: int = 5

    # MCTS
    num_simulations: int = 100
    sampled_action_times: int = 10
    pb_c_base: float = 19652.0
    pb_c_init: float = 1.25
    root_dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25
    mcts_rho: float = 0.75
    mcts_lambda: float = 0.8
    tree_value_stat_delta_lb: float = 0.01

    # Training
    training_steps: int = 100_000
    batch_size: int = 256
    unroll_steps: int = 5
    td_steps: int = 5
    discount: float = 0.99
    learning_rate: float = 1e-4
    adam_eps: float = 1e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 5.0
    awpo_alpha: float = 1.0
    adv_clip: float = 3.0
    reward_loss_coeff: float = 1.0
    value_loss_coeff: float = 0.25
    policy_loss_coeff: float = 1.0
    consistency_coeff: float = 2.0

    # Replay
    replay_buffer_size: int = 100_000
    min_replay_size: int = 300
    priority_alpha: float = 0.6
    priority_beta_start: float = 0.4
    priority_beta_end: float = 1.0
    target_model_interval: int = 200

    # Reanalyze
    use_reanalyze: bool = True
    revisit_policy_search_rate: float = 0.99

    # Collection
    num_envs_parallel: int = 8
    updates_per_collection: int = 16

    # Async / Ray
    num_actors: int = 3
    num_reanalyze_actors: int = 0  # Set >0 to enable async reanalysis
    reanalyze_batch_size: int = 8
    param_update_interval: int = 1
    learner_steps_per_call: int = 10

    # Logging / eval
    eval_interval: int = 1000
    eval_episodes: int = 32
    log_interval: int = 100
    seed: int = 0

    @property
    def support_size(self) -> int:
        return self.value_support_size * 2 + 1

    @property
    def reward_support_size_total(self) -> int:
        return self.reward_support_size * 2 + 1
