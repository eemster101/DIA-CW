import gymnasium as gym
import minigrid

def make_env(env_key, seed=None, render_mode=None, layout_id=0):
    env = gym.make(env_key, render_mode=render_mode, layout_id=layout_id)
    env.reset(seed=seed)
    return env
