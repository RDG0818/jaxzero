# utils/smax_env_wrapper.py
#
# Stub for a JaxMARL SMAX (StarCraft Multi-Agent Challenge) environment wrapper.
#
# To implement, mirror the MPEEnvWrapper interface:
#   - __init__(env_name, num_agents, max_steps)
#   - reset(rng_key) -> (observations: np.ndarray, state)
#   - step(rng_key, state, actions) -> (next_obs, next_state, team_reward, episode_done)
#   - observation_shape: Tuple[int, ...]
#   - action_space_size: int
#
# SMAX reference: https://jaxmarl.foersterlab.com/
