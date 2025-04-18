import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from environment import WarehouseEnv  # Fixed import

# Custom observation wrapper for MiniGrid
class RGBImgObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space["image"]
        
    def observation(self, obs):
        return obs["image"]

# Custom feature extractor
class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

def train_agent():
    # Environment setup with proper wrapper
    env = WarehouseEnv(render_mode="rgb_array")
    env = RGBImgObsWrapper(env)
    
    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )
    
    model = PPO(
        "CnnPolicy",  # Changed to CnnPolicy since we're using image observations
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=1e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )
    
    model.learn(total_timesteps=100000)
    model.save("warehouse_ppo_agent")
    env.close()

def evaluate_agent():
    model = PPO.load("warehouse_ppo_agent")
    env = WarehouseEnv(render_mode="human")
    
    obs, _ = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        if terminated or truncated:
            obs, _ = env.reset()
    
    env.close()

if __name__ == "__main__":
    train_agent()
    # evaluate_agent()  # Uncomment to evaluate after training