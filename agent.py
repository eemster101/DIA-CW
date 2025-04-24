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
    print("Using device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    
    # Train progressively on harder layouts
    for layout_id in [0, 1]:
        env = WarehouseEnv(render_mode="rgb_array", layout_id=layout_id)
        env = RGBImgObsWrapper(env)
        
        if layout_id == 0:  # First layout - create new model
            model = PPO(
                "CnnPolicy",
                env,
                policy_kwargs=dict(
                    features_extractor_class=MinigridFeaturesExtractor,
                    features_extractor_kwargs=dict(features_dim=128),
                ),
                verbose=1,
                learning_rate=1e-4,
                n_steps=1024,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        else:  # Subsequent layouts - continue training existing model
            model.set_env(env)
            
        model.learn(total_timesteps=100000)  # 50k per layout
    
    model.save("warehouse_ppo_agent")
    env.close()

def evaluate_agent():
    model = PPO.load("warehouse_ppo_agent")
    env = WarehouseEnv(render_mode="human", layout_id=1)
    env = RGBImgObsWrapper(env)  # This wraps the original env
    
    obs, _ = env.reset()
    for step in range(200):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Access the original env through unwrapped
        print(f"Step: {step}, Pos: {env.unwrapped.agent_pos}, Dir: {env.unwrapped.agent_dir}, "
              f"Action: {action}, Reward: {reward}")
        
        env.render()
        
        if terminated or truncated:
            obs, _ = env.reset()
    
    env.close()

if __name__ == "__main__":
    train_agent()
    #evaluate_agent()  # Uncomment to evaluate after training