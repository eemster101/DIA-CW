from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Lava  # Added Lava import
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv

class WarehouseEnv(MiniGridEnv):
    def __init__(
        self,
        size=10,
        agent_start_pos=(2, 3),
        agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

        # Define rewards
        self.reward_pickup_key = 1.0
        self.reward_unlock_door = 1.0
        self.reward_reach_goal = 10.0
        self.step_penalty = -0.01
        self.wrong_door_penalty = -0.5
        self.lava_penalty = -0.5  # Penalty for stepping in lava

    @staticmethod
    def _gen_mission():
        return "Navigate to the goal"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate vertical separation wall
        for i in range(0, height):
            self.grid.set(5, i, Wall())
        
        # Place the door and key
        self.grid.set(5, 6, Door(COLOR_NAMES[0], is_locked=True))
        self.grid.set(3, 6, Key(COLOR_NAMES[0]))

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Add lava tiles (customize positions as needed)
        self.grid.set(4, 2, Lava())
        self.grid.set(7, 5, Lava())

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "Navigate to the goal"
        self.has_key = False
        self.door_unlocked = False

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
            new_fwd_cell = self.grid.get(*self.front_pos)
            if new_fwd_cell and new_fwd_cell.type == "goal":
                reward += self.reward_reach_goal
                terminated = True

        obs = self.gen_obs()
        
        # Check if maximum steps reached
        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True
            
        return obs, reward, terminated, truncated, info

def main():
    env = WarehouseEnv(render_mode="human")

    # enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()

if __name__ == "__main__":
    main()