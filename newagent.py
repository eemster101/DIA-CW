import numpy as np
import gymnasium as gym
from gymnasium import spaces
from minigrid.wrappers import FullyObsWrapper, RGBImgPartialObsWrapper
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import random

# Custom Warehouse Environment
class WarehouseEnv(gym.Env):
    def __init__(self, size=10, max_steps=100, num_boxes=5, num_employees=3):
        super().__init__()
        self.size = size
        self.max_steps = max_steps
        self.num_boxes = num_boxes
        self.num_employees = num_employees
        
        # Action space: left, right, forward, pickup, drop, toggle (open/close), done
        self.action_space = spaces.Discrete(7)
        
        # Observation space (fully observable grid)
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=255,
                shape=(size, size, 3),  # RGB image
                dtype='uint8'
            ),
            'direction': spaces.Discrete(4),
            'carrying': spaces.Box(
                low=0,
                high=1,
                shape=(1,),  # 0 or 1 indicating if carrying a box
                dtype='int'
            )
        })
        
        # Initialize state
        self.reset()
    
    def reset(self):
        # Create empty grid
        self.grid = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        
        # Place robot at random position facing random direction
        self.robot_pos = (np.random.randint(1, self.size-1), 
                          np.random.randint(1, self.size-1))
        self.robot_dir = np.random.randint(0, 4)  # 0: right, 1: down, 2: left, 3: up
        self.carrying = 0
        
        # Place boxes (keys in MiniGrid terms)
        self.boxes = []
        for _ in range(self.num_boxes):
            pos = (np.random.randint(1, self.size-1), 
                   np.random.randint(1, self.size-1))
            while pos == self.robot_pos or pos in self.boxes:
                pos = (np.random.randint(1, self.size-1), 
                       np.random.randint(1, self.size-1))
            self.boxes.append(pos)
            self.grid[pos] = [255, 165, 0]  # Orange for boxes
        
        # Place employees (dynamic obstacles)
        self.employees = []
        for _ in range(self.num_employees):
            pos = (np.random.randint(1, self.size-1), 
                   np.random.randint(1, self.size-1))
            while pos == self.robot_pos or pos in self.boxes or pos in self.employees:
                pos = (np.random.randint(1, self.size-1), 
                       np.random.randint(1, self.size-1))
            self.employees.append(pos)
            self.grid[pos] = [0, 0, 255]  # Blue for employees
        
        # Place spills (lava)
        self.spills = []
        for _ in range(2):  # Few spills
            pos = (np.random.randint(1, self.size-1), 
                   np.random.randint(1, self.size-1))
            while pos == self.robot_pos or pos in self.boxes or pos in self.employees:
                pos = (np.random.randint(1, self.size-1), 
                       np.random.randint(1, self.size-1))
            self.spills.append(pos)
            self.grid[pos] = [255, 0, 0]  # Red for spills
        
        # Place delivery points (green)
        self.delivery_points = []
        for _ in range(2):  # Few delivery points
            pos = (np.random.randint(1, self.size-1), 
                   np.random.randint(1, self.size-1))
            while pos == self.robot_pos or pos in self.boxes or pos in self.employees or pos in self.spills:
                pos = (np.random.randint(1, self.size-1), 
                       np.random.randint(1, self.size-1))
            self.delivery_points.append(pos)
            self.grid[pos] = [0, 255, 0]  # Green for delivery points
        
        # Set robot position
        self.grid[self.robot_pos] = [128, 128, 128]  # Gray for robot
        
        self.steps = 0
        self.delivered = 0
        
        return self._get_obs()
    
    def _get_obs(self):
        return {
            'image': self.grid.copy(),
            'direction': self.robot_dir,
            'carrying': np.array([self.carrying], dtype=int)
        }
    
    def step(self, action):
        self.steps += 1
        reward = -0.1  # Small penalty for each step to encourage efficiency
        done = False
        info = {}
        
        # Save previous position
        prev_pos = self.robot_pos
        
        # Execute action
        if action == 0:  # Turn left
            self.robot_dir = (self.robot_dir - 1) % 4
        elif action == 1:  # Turn right
            self.robot_dir = (self.robot_dir + 1) % 4
        elif action == 2:  # Move forward
            x, y = self.robot_pos
            if self.robot_dir == 0:  # Right
                new_pos = (x + 1, y)
            elif self.robot_dir == 1:  # Down
                new_pos = (x, y + 1)
            elif self.robot_dir == 2:  # Left
                new_pos = (x - 1, y)
            else:  # Up
                new_pos = (x, y - 1)
            
            # Check if new position is valid
            if (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size and
                new_pos not in self.spills and new_pos not in self.employees):
                self.robot_pos = new_pos
            else:
                reward = -1  # Penalty for hitting obstacle or boundary
        
        elif action == 3:  # Pickup box
            if not self.carrying and self.robot_pos in self.boxes:
                self.carrying = 1
                self.boxes.remove(self.robot_pos)
                reward = 1  # Reward for picking up a box
        
        elif action == 4:  # Drop box
            if self.carrying:
                if self.robot_pos in self.delivery_points:
                    reward = 5  # Big reward for delivering to correct location
                    self.delivered += 1
                else:
                    reward = -1  # Penalty for dropping in wrong place
                    self.boxes.append(self.robot_pos)
                self.carrying = 0
        
        elif action == 5:  # Toggle (not used in this simple version)
            pass
        
        elif action == 6:  # Done
            done = True
        
        # Update grid visualization
        self.grid[prev_pos] = [0, 0, 0]  # Clear previous position
        
        # Move employees (dynamic obstacles)
        for i, emp in enumerate(self.employees):
            # Random movement
            direction = np.random.randint(0, 4)
            x, y = emp
            if direction == 0:  # Right
                new_pos = (x + 1, y)
            elif direction == 1:  # Down
                new_pos = (x, y + 1)
            elif direction == 2:  # Left
                new_pos = (x - 1, y)
            else:  # Up
                new_pos = (x, y - 1)
            
            # Check if new position is valid
            if (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size and
                new_pos not in self.spills and new_pos != self.robot_pos and
                new_pos not in self.boxes and new_pos not in self.delivery_points):
                self.employees[i] = new_pos
        
        # Update grid with new positions
        for box in self.boxes:
            self.grid[box] = [255, 165, 0]
        for emp in self.employees:
            self.grid[emp] = [0, 0, 255]
        for spill in self.spills:
            self.grid[spill] = [255, 0, 0]
        for dp in self.delivery_points:
            self.grid[dp] = [0, 255, 0]
        self.grid[self.robot_pos] = [128, 128, 128]
        
        # Check termination conditions
        if self.steps >= self.max_steps:
            done = True
        
        return self._get_obs(), reward, done, info

# Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(PolicyNetwork, self).__init__()
        
        # CNN for processing grid observation
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        
        # Calculate CNN output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, *input_shape)
            cnn_output_size = self.cnn(dummy_input).shape[1]
        
        # Combined network
        self.fc = nn.Sequential(
            nn.Linear(cnn_output_size + 2, 64),  # +2 for direction and carrying
            nn.ReLU(),
            nn.Linear(64, num_actions),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, grid_obs, direction, carrying):
        # Process grid observation
        grid_features = self.cnn(grid_obs)
        
        # Combine features
        direction = direction.float().unsqueeze(1)
        carrying = carrying.float().unsqueeze(1)
        combined = torch.cat([grid_features, direction, carrying], dim=1)
        
        # Get action probabilities
        action_probs = self.fc(combined)
        return action_probs

# PPO Agent
class PPOAgent:
    def __init__(self, env):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get observation and action space sizes
        obs_shape = env.observation_space['image'].shape[:2]
        self.num_actions = env.action_space.n
        
        # Initialize policy network
        self.policy = PolicyNetwork(obs_shape, self.num_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.001)
        
        # Hyperparameters
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.K_epochs = 4
        self.buffer = []
        self.buffer_size = 2048
        
    def preprocess_obs(self, obs):
        # Convert observation to tensor
        grid_obs = torch.from_numpy(obs['image']).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
        direction = torch.tensor(obs['direction']).unsqueeze(0).to(self.device)
        carrying = torch.from_numpy(obs['carrying']).unsqueeze(0).to(self.device)
        return grid_obs, direction, carrying
    
    def get_action(self, obs):
        grid_obs, direction, carrying = self.preprocess_obs(obs)
        
        with torch.no_grad():
            action_probs = self.policy(grid_obs, direction, carrying)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item()
    
    def store_transition(self, transition):
        self.buffer.append(transition)
        if len(self.buffer) >= self.buffer_size:
            self.update()
            self.buffer = []
    
    def update(self):
        if not self.buffer:
            return
        
        # Unpack buffer
        old_obs = [t[0] for t in self.buffer]
        old_actions = torch.tensor([t[1] for t in self.buffer], dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor([t[2] for t in self.buffer], dtype=torch.float).to(self.device)
        rewards = torch.tensor([t[3] for t in self.buffer], dtype=torch.float).to(self.device)
        new_obs = [t[4] for t in self.buffer]
        dones = torch.tensor([t[5] for t in self.buffer], dtype=torch.float).to(self.device)
        
        # Calculate discounted rewards
        discounted_rewards = []
        running_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            running_reward = reward + (1 - done) * self.gamma * running_reward
            discounted_rewards.insert(0, running_reward)
        
        # Normalize rewards
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float).to(self.device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-7)
        
        # Convert observations to tensors
        grid_obs = []
        directions = []
        carryings = []
        for obs in old_obs:
            go, d, c = self.preprocess_obs(obs)
            grid_obs.append(go)
            directions.append(d)
            carryings.append(c)
        
        grid_obs = torch.cat(grid_obs, dim=0)
        directions = torch.cat(directions, dim=0)
        carryings = torch.cat(carryings, dim=0)
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluate old actions with current policy
            action_probs = self.policy(grid_obs, directions, carryings)
            dist = Categorical(action_probs)
            log_probs = dist.log_prob(old_actions)
            
            # Calculate ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(log_probs - old_log_probs.detach())
            
            # Calculate surrogate loss
            advantages = discounted_rewards
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Calculate value function loss (simplified)
            value_loss = advantages.pow(2).mean()
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# Training function
def train_agent(env, agent, num_episodes=1000):
    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Get action from agent
            action, log_prob = agent.get_action(obs)
            
            # Take action in environment
            new_obs, reward, done, _ = env.step(action)
            total_reward += reward
            
            # Store transition
            agent.store_transition((obs, action, log_prob, reward, new_obs, done))
            
            obs = new_obs
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}, Total Reward: {total_reward}")

def evaluate_agent(env, agent, num_episodes=100, render=False):
    total_rewards = []
    success_rates = []
    episode_lengths = []
    collision_counts = []
    delivery_counts = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        done = False
        steps = 0
        collisions = 0
        deliveries = 0
        
        while not done:
            if render:
                env.render()
                
            action, _ = agent.get_action(obs)
            obs, reward, done, info = env.step(action)
            
            total_reward += reward
            steps += 1
            
            # Check for collisions (negative rewards)
            if reward < -0.5:  # Threshold for collision penalty
                collisions += 1
            
            # Check for deliveries (positive rewards)
            if reward > 4.5:  # Threshold for delivery reward
                deliveries += 1
        
        total_rewards.append(total_reward)
        episode_lengths.append(steps)
        collision_counts.append(collisions)
        delivery_counts.append(deliveries)
        success_rates.append(1 if deliveries > 0 else 0)  # At least one delivery
        
        if (episode + 1) % 10 == 0:
            print(f"Evaluation Episode {episode + 1}/{num_episodes}")
    
    # Calculate metrics
    metrics = {
        'avg_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'avg_episode_length': np.mean(episode_lengths),
        'success_rate': np.mean(success_rates),
        'avg_collisions': np.mean(collision_counts),
        'avg_deliveries': np.mean(delivery_counts),
        'efficiency': np.mean(delivery_counts) / np.mean(episode_lengths)
    }
    
    return metrics

def comprehensive_evaluation(env, agent):
    # 1. Standard Performance Evaluation
    print("\n=== Standard Performance Evaluation ===")
    standard_metrics = evaluate_agent(env, agent, num_episodes=100)
    for k, v in standard_metrics.items():
        print(f"{k:20}: {v:.2f}")
    
    # 2. Obstacle Avoidance Test
    print("\n=== Obstacle Avoidance Test ===")
    env.num_employees = 10  # High number of dynamic obstacles
    env.num_boxes = 2  # Few boxes to focus on navigation
    obstacle_metrics = evaluate_agent(env, agent, num_episodes=50)
    print(f"Collision rate with high obstacles: {obstacle_metrics['avg_collisions']:.2f}")
    
    # 3. Delivery Efficiency Test
    print("\n=== Delivery Efficiency Test ===")
    env.num_employees = 3  # Reset to normal
    env.num_boxes = 10  # Many boxes to deliver
    efficiency_metrics = evaluate_agent(env, agent, num_episodes=50)
    print(f"Deliveries per episode: {efficiency_metrics['avg_deliveries']:.2f}")
    print(f"Efficiency (deliveries/step): {efficiency_metrics['efficiency']:.4f}")
    
    # 4. Generalization Test (unseen layouts)
    print("\n=== Generalization Test ===")
    original_size = env.size
    env.size = 15  # Larger unseen warehouse
    generalization_metrics = evaluate_agent(env, agent, num_episodes=50)
    print(f"Success rate in larger warehouse: {generalization_metrics['success_rate']:.2f}")
    env.size = original_size  # Reset size
    
    # 5. Stress Test (many dynamic elements)
    print("\n=== Stress Test ===")
    env.num_employees = 8
    env.num_boxes = 8
    stress_metrics = evaluate_agent(env, agent, num_episodes=50)
    print(f"Performance under stress: {stress_metrics['avg_reward']:.2f}")
    
    # Return to original configuration
    env.num_employees = 3
    env.num_boxes = 5


import matplotlib.pyplot as plt

def plot_training_progress(logs):
    """Plot training metrics over time"""
    plt.figure(figsize=(12, 8))
    
    # Reward plot
    plt.subplot(2, 2, 1)
    plt.plot(logs['episodes'], logs['rewards'])
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    # Success rate plot
    plt.subplot(2, 2, 2)
    plt.plot(logs['episodes'], logs['success_rates'])
    plt.title('Success Rates')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    
    # Collisions plot
    plt.subplot(2, 2, 3)
    plt.plot(logs['episodes'], logs['collisions'])
    plt.title('Collision Counts')
    plt.xlabel('Episode')
    plt.ylabel('Collisions per Episode')
    
    # Efficiency plot
    plt.subplot(2, 2, 4)
    plt.plot(logs['episodes'], logs['efficiency'])
    plt.title('Delivery Efficiency')
    plt.xlabel('Episode')
    plt.ylabel('Deliveries per Step')
    
    plt.tight_layout()
    plt.show()

def visualize_agent(env, agent, num_episodes=3):
    """Render agent's performance visually"""
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            env.render()  # Requires proper rendering setup
            action, _ = agent.get_action(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            time.sleep(0.1)  # Slow down for visualization
        
        print(f"Episode {episode + 1} completed with reward: {total_reward}")


if __name__ == "__main__":
    # Create environment
    env = WarehouseEnv(size=10, max_steps=200, num_boxes=5, num_employees=3)
    
    # Create agent and load trained model
    agent = PPOAgent(env)
    agent.policy.load_state_dict(torch.load("warehouse_robot.pth"))
    agent.policy.eval()  # Set to evaluation mode
    
    # Run comprehensive evaluation
    comprehensive_evaluation(env, agent)
    
    # Visualize agent performance
    visualize_agent(env, agent, num_episodes=3)