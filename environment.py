from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Lava  # Added Lava import
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.world_object import WorldObj
import numpy as np
from PIL import Image

class CustomBox(Key):
    def render(self, img):
        # Load your own key image (ensure it's 32x32 or resized)
        key_image = Image.open("assets/box.png").resize((img.shape[1], img.shape[0]))
        key_image = key_image.convert("RGB")
        img[:, :, :] = np.asarray(key_image)

class CustomSpill(Lava):
    def render(self, img):
        # Load your custom lava image
        lava_image = Image.open("assets/stain.png").resize((img.shape[1], img.shape[0]))
        lava_image = lava_image.convert("RGB")
        img[:, :, :] = np.asarray(lava_image)


class WarehouseEnv(MiniGridEnv):
    def __init__(
        self,
        size=10,
        agent_start_pos=(2, 3),
        agent_start_dir=0,
        layout_id=0,  # <-- NEW PARAM
        max_steps: int | None = None,
        **kwargs,
    ):
        self.layout_id = layout_id  # <-- STORE IT
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

        # Define rewards
        self.reward_pickup_key = 1.0
        self.reward_unlock_door = 1.0
        self.reward_reach_goal = 10.0
        self.wall_penalty = -1.0
        self.step_penalty = -0.02
        self.wrong_door_penalty = -0.5
        self.lava_penalty = -5.0

    def _gen_grid(self, width, height):
        if self.layout_id == 0:
            self.layout0(width, height)
        elif self.layout_id == 1:
            self.layout1(width, height)
        elif self.layout_id == 2:
            self.layout2(width, height)
        else:
            raise ValueError(f"Unknown layout_id: {self.layout_id}")


    @staticmethod
    def _gen_mission():
        return "Navigate to the goal"
    
    def layout0(self, width, height):
        width = 6
        height = 6

        self.grid = Grid(width, height)
    
        self.grid.wall_rect(0, 0, width, height)

        self.put_obj(Goal(), width - 2, height - 2)


        self.agent_pos = (1, 1)
        self.agent_dir = 0

        self.has_key = False
        self.door_unlocked = False
        self.prev_dist = np.linalg.norm(np.array(self.agent_pos) - np.array([width - 2, height - 2]))

    def layout1(self, width, height):
        width = 6
        height = 6

        self.grid = Grid(width, height)
    
        self.grid.wall_rect(0, 0, width, height)

        self.put_obj(Goal(), width - 2, height - 2)

        self.grid.set(4, 3, CustomSpill())


        self.agent_pos = (1, 1)
        self.agent_dir = 0

        self.has_key = False
        self.door_unlocked = False
        self.prev_dist = np.linalg.norm(np.array(self.agent_pos) - np.array([width - 2, height - 2]))
    

    def layout2(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        

        # Generate vertical separation wall
        for i in range(0, height):
            self.grid.set(5, i, Wall())
        
        # Place the door and key
        self.grid.set(5, 6, Door(COLOR_NAMES[0], is_locked=True))
        self.grid.set(3, 6, CustomBox(COLOR_NAMES[0]))

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Add lava tiles (customize positions as needed)
        self.grid.set(4, 2, CustomSpill())
        self.grid.set(7, 5, CustomSpill())

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "Navigate to the goal"
        self.has_key = False
        self.door_unlocked = False
        self.prev_dist = np.linalg.norm(np.array(self.agent_pos) - np.array([width - 2, height - 2]))

    def step(self, action):

        reward = self.step_penalty
        terminated = False
        truncated = False
        info = {}
        
        # Get the position in front of the agent before moving
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)
        
        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4
            
        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4
            
        # Move forward
        elif action == self.actions.forward:
            # Check if we're about to step into lava
            if fwd_cell and fwd_cell.type == "lava":
                reward += self.lava_penalty
                info["stepped_in_lava"] = True
                # Still allow movement into lava
                self.agent_pos = fwd_pos
            elif fwd_cell and fwd_cell.type == "wall":
                reward += self.wall_penalty
                info["Hitting_a_wall"] = True
            else:
                # Check for key before moving (only if not stepping into lava)
                if fwd_cell and fwd_cell.type == "key" and fwd_cell.color == COLOR_NAMES[0]:
                    self.grid.set(*fwd_pos, None)
                    self.carrying = fwd_cell
                    reward += self.reward_pickup_key
                    self.has_key = True
                    info["picked_key"] = True
                
                # Check if we can move forward
                can_move = False
                if fwd_cell is None:
                    can_move = True
                elif fwd_cell.can_overlap():
                    can_move = True
                elif fwd_cell.type == "door":
                    # Automatically unlock and open door if we have the key
                    if self.has_key and fwd_cell.color == COLOR_NAMES[0] and fwd_cell.is_locked:
                        fwd_cell.is_locked = False
                        fwd_cell.is_open = True
                        reward += self.reward_unlock_door
                        self.carrying = None  # Key is used up
                        info["unlocked_door"] = True
                    
                    # Pass through if door is unlocked/open
                    if not fwd_cell.is_locked:
                        can_move = True
                        fwd_cell.is_open = True  # Ensure door is marked as open
                        
                if can_move:
                    self.agent_pos = fwd_pos
                
            # Check for goal after moving
            # Check if agent is standing on the goal tile
            current_cell = self.grid.get(*self.agent_pos)
            if current_cell and current_cell.type == "goal":
                reward += self.reward_reach_goal
                terminated = True

        obs = self.gen_obs()
        
        # Check if maximum steps reached
        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True
            
        if fwd_cell and fwd_cell.type == "key":
            reward += 0.2  # Small reward for approaching key
        
             
        # Safe door checking:
        current_cell = self.grid.get(*self.agent_pos)
        if self.has_key and current_cell is not None and current_cell.type == "door":
            reward += 0.3  # Reward for approaching door with key
                    
        # Distance-based reward
        goal_pos = np.array([self.width-2, self.height-2])
        dist_to_goal = np.linalg.norm(np.array(self.agent_pos) - goal_pos)
        reward += 0.01 * (self.prev_dist - dist_to_goal)
        self.prev_dist = dist_to_goal
        

        return obs, reward, terminated, truncated, info

def main():
    env = WarehouseEnv(render_mode="human", layout_id=0)  # Change to 0, 1, or 2
    manual_control = ManualControl(env, seed=42)
    manual_control.start()

if __name__ == "__main__":
    main()