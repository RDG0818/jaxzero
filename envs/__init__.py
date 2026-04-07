from envs.mpe_env_wrapper import MPEEnvWrapper, VecMPEEnvWrapper
from envs.smax_env_wrapper import SMAXEnvWrapper, VecSMAXEnvWrapper


def make_env_wrapper(env_name: str, num_agents: int, max_steps: int):
    """Instantiate the right single-env wrapper based on env_name.

    MPE environments: env_name starts with "MPE_" (e.g. "MPE_simple_spread_v3").
    SMAX environments: scenario string (e.g. "3m", "2s3z", "8m").
    """
    if env_name.startswith("MPE_"):
        return MPEEnvWrapper(env_name, num_agents, max_steps)
    return SMAXEnvWrapper(env_name, num_agents, max_steps)


def make_vec_env_wrapper(env_name: str, num_agents: int, max_steps: int, num_envs: int):
    """Instantiate the right vectorized-env wrapper based on env_name."""
    if env_name.startswith("MPE_"):
        return VecMPEEnvWrapper(env_name, num_agents, max_steps, num_envs)
    return VecSMAXEnvWrapper(env_name, num_agents, max_steps, num_envs)
