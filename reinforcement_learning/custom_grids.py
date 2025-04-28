from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Lava
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.world_object import WorldObj
import numpy as np
from PIL import Image



class CustomBox(Key):
    """
    A custom box object that can be picked up (rendered with a custom image)
    """
    def render(self, img):
        key_image = Image.open("assets/box.png").resize((img.shape[1], img.shape[0]))
        key_image = key_image.convert("RGB")
        img[:, :, :] = np.asarray(key_image)


class CustomSpill(Lava):
    """
    A custom spill object that acts as an obstacle (rendered with a custom image)
    """
    def render(self, img):
        lava_image = Image.open("assets/stain.png").resize((img.shape[1], img.shape[0]))
        lava_image = lava_image.convert("RGB")
        img[:, :, :] = np.asarray(lava_image)


class WarehouseEnv(MiniGridEnv):
    """
    ## Description
    
    Warehouse environment with multiple layouts containing boxes to collect and spills to avoid.
    The agent must navigate through the warehouse, collect boxes (keys), and reach the goal.
    
    ## Mission Space
    
    "Navigate to the goal"
    
    ## Action Space
    
    | Num | Name         | Action                    |
    |-----|--------------|---------------------------|
    | 0   | left         | Turn left                 |
    | 1   | right        | Turn right                |
    | 2   | forward      | Move forward              |
    | 3   | pickup       | Pick up an object         |
    | 4   | drop         | Unused                    |
    | 5   | toggle       | Toggle/activate an object |
    | 6   | done         | Unused                    |
    
    ## Observation Encoding
    
    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked
    
    ## Rewards
    
    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.
    
    ## Termination
    
    The episode ends if any one of the following conditions is met:
    1. The agent reaches the goal.
    2. Timeout (see `max_steps`).
    """

    def __init__(
        self,
        size=10,
        agent_start_pos=(2, 3),
        agent_start_dir=0,
        layout_id=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.layout_id = layout_id
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

    @staticmethod
    def _gen_mission():
        return "Navigate to the goal"

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)
        
        # Generate walls around the perimeter
        self.grid.wall_rect(0, 0, width, height)
        
        # Generate the selected layout
        layout_method = getattr(self, f"layout{self.layout_id}", None)
        if layout_method:
            layout_method(width, height)
        else:
            raise ValueError(f"Unknown layout_id: {self.layout_id}")

        # Place the agent if start position wasn't set by the layout
        if not hasattr(self, 'agent_pos'):
            self.place_agent()

    def layout0(self, width, height):
        """Simple empty layout with goal in bottom-right"""
        self.put_obj(Goal(), width - 2, height - 2)
        self.agent_pos = (1, 1)
        self.agent_dir = 0

    def layout1(self, width, height):
        """Empty layout with one spill obstacle"""
        self.put_obj(Goal(), width - 2, height - 2)
        self.grid.set(4, 3, CustomSpill())
        self.agent_pos = (1, 1)
        self.agent_dir = 0

    def layout2(self, width, height):
        """Layout with locked door, key, and spills"""
        # Vertical separation wall
        for i in range(0, height):
            self.grid.set(5, i, Wall())
        
        # Place the door and key
        self.grid.set(5, 6, Door(COLOR_NAMES[0], is_locked=True))
        self.grid.set(3, 6, CustomBox(COLOR_NAMES[0]))

        # Place goal
        self.put_obj(Goal(), width - 2, height - 2)

        # Add spills
        self.grid.set(4, 2, CustomSpill())
        self.grid.set(7, 5, CustomSpill())

        # Place agent
        self.agent_pos = self.agent_start_pos if self.agent_start_pos else (1, 1)
        self.agent_dir = self.agent_start_dir

    def layout3(self, width, height):

        width = 10
        height = 10
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.grid.wall_rect(2, 1, 1, 3)
        self.grid.wall_rect(4, 1, 1, 3)
        self.grid.wall_rect(6, 1, 1, 3)

        self.grid.wall_rect(2, 6, 1, 3)
        self.grid.wall_rect(4, 6, 1, 3)
        self.grid.wall_rect(6, 6, 1, 3)


        self.put_obj(Goal(), 5, height - 2)
        self.agent_pos = (1, 4)
        self.agent_dir = 0
        self.has_key = False
        self.door_unlocked = False
        self.prev_dist = np.linalg.norm(np.array(self.agent_pos) - np.array([5, height - 2]))

    def layout4(self, width, height):
        width = 10
        height = 10
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.grid.wall_rect(2, 1, 1, 3)
        self.grid.wall_rect(4, 1, 1, 3)
        self.grid.wall_rect(6, 1, 1, 3)

        self.grid.set(4, 5, CustomSpill())

        self.grid.wall_rect(2, 6, 1, 3)
        self.grid.wall_rect(4, 6, 1, 3)
        self.grid.wall_rect(6, 6, 1, 3)


        self.put_obj(Goal(), 5, height - 2)
        self.agent_pos = (1, 4)
        self.agent_dir = 0
        self.has_key = False
        self.door_unlocked = False
        self.prev_dist = np.linalg.norm(np.array(self.agent_pos) - np.array([5, height - 2]))
    
    
    def layout5(self, width, height):
        width = 9
        height = 9
        self.grid = Grid(width, height)

        self.grid.wall_rect(1, 0, 1, 1)
        self.grid.wall_rect(3, 0, 1, 1)
        self.grid.wall_rect(5, 0, 1, 1)
        self.grid.wall_rect(7, 0, 1, 1)

        self.grid.set(5, 1, CustomBox(COLOR_NAMES[0]))

        self.grid.wall_rect(0, 3, 1, 1)
        self.grid.wall_rect(2, 3, 1, 1)
        self.grid.wall_rect(4, 3, 1, 1)
        self.grid.wall_rect(6, 3, 1, 1)

        self.grid.wall_rect(8, 4, 1, 1)


        self.grid.wall_rect(2, 5, 1, 1)
        self.grid.wall_rect(4, 5, 1, 1)
        self.grid.wall_rect(6, 5, 1, 1)

        self.grid.wall_rect(2, 7, 1, 1)
        self.grid.wall_rect(4, 7, 1, 1)

        self.grid.set(6, 7, CustomBox(COLOR_NAMES[0]))
        self.grid.wall_rect(6, 8, 3, 1)

        self.put_obj(Goal(), 8, 5)
        self.agent_pos = (0, 5)
        self.agent_dir = 0
        self.has_key = False
        self.door_unlocked = False
        self.prev_dist = np.linalg.norm(np.array(self.agent_pos) - np.array([8, 5]))

    def layout6(self, width, height):
        width = 10
        height = 10
        self.grid = Grid(width, height)

        self.grid.wall_rect(0, 0, width, height)

        self.grid.wall_rect(2, 0, 1, 3)
        self.grid.wall_rect(4, 0, 1, 3)

        self.grid.wall_rect(7, 2, 1, 1)

        self.grid.wall_rect(4, 5, 3, 1)

        self.grid.wall_rect(1, 7, 6, 1)

        self.put_obj(Goal(), 1, 8)
        self.agent_pos = (1, 5)
        self.agent_dir = 0
        self.has_key = False
        self.door_unlocked = False
        self.prev_dist = np.linalg.norm(np.array(self.agent_pos) - np.array([1, 8]))

    def layout7(self, width, height):
        width = 10
        height = 10
        self.grid = Grid(width, height)

        self.grid.wall_rect(0, 0, width, height)

        self.grid.wall_rect(2, 0, 1, 3)
        self.grid.wall_rect(4, 0, 1, 3)

        self.grid.wall_rect(7, 2, 1, 1)

        self.grid.set(8, 2, CustomBox(COLOR_NAMES[0]))
        self.grid.set(4, 3, CustomSpill())

        self.grid.wall_rect(4, 5, 3, 1)

        self.grid.wall_rect(1, 7, 6, 1)

        self.put_obj(Goal(), 1, 8)
        self.agent_pos = (1, 5)
        self.agent_dir = 0
        self.has_key = False
        self.door_unlocked = False
        self.prev_dist = np.linalg.norm(np.array(self.agent_pos) - np.array([1, 8]))

    def layout8(self, width, height):
        width = 10
        height = 10
        self.grid = Grid(width, height)

        self.grid.wall_rect(0, 0, width, height)

        self.grid.wall_rect(2, 0, 1, 8)

        self.grid.wall_rect(4, 2, 1, 6)

        self.grid.wall_rect(6, 0, 1, 5)
        self.grid.wall_rect(6, 6, 1, 3)

        self.put_obj(Goal(), 8, 5)
        self.agent_pos = (1, 1)
        self.agent_dir = 0
        self.has_key = False
        self.door_unlocked = False
        self.prev_dist = np.linalg.norm(np.array(self.agent_pos) - np.array([8, 5]))

    def layout9(self, width, height):
        width = 10
        height = 10
        self.grid = Grid(width, height)

        self.grid.wall_rect(0, 0, width, height)

        self.grid.wall_rect(2, 0, 1, 8)

        self.grid.wall_rect(4, 2, 1, 6)

        self.grid.set(5, 7, CustomSpill())
        self.grid.set(7, 3, CustomBox(COLOR_NAMES[0]))

        self.grid.wall_rect(6, 0, 1, 5)
        self.grid.wall_rect(6, 6, 1, 3)

        self.put_obj(Goal(), 8, 5)
        self.agent_pos = (1, 1)
        self.agent_dir = 0
        self.has_key = False
        self.door_unlocked = False
        self.prev_dist = np.linalg.norm(np.array(self.agent_pos) - np.array([8, 5]))

    def layout10(self, width, height):
        width = 10
        height = 10
        self.grid = Grid(width, height)

        self.grid.wall_rect(0, 0, width, height)

        self.grid.wall_rect(2, 2, 3, 1)
        self.grid.wall_rect(6, 2, 2, 1)

        self.grid.wall_rect(2, 4, 3, 1)

        self.grid.wall_rect(7, 5, 1, 1)

        self.grid.wall_rect(2, 7, 3, 1)

        self.grid.wall_rect(6, 7, 1, 2)

        self.put_obj(Goal(), 8, 8)
        self.agent_pos = (1, 1)
        self.agent_dir = 0
        self.has_key = False
        self.door_unlocked = False
        self.prev_dist = np.linalg.norm(np.array(self.agent_pos) - np.array([8, 8]))
    
    def layout11(self, width, height):
        width = 10
        height = 10
        self.grid = Grid(width, height)

        self.grid.wall_rect(0, 0, width, height)

        self.grid.wall_rect(2, 2, 3, 1)
        self.grid.wall_rect(6, 2, 2, 1)

        self.grid.wall_rect(2, 4, 3, 1)

        self.grid.wall_rect(7, 5, 1, 1)

        self.grid.set(5, 7, CustomSpill())
        self.grid.set(6, 3, CustomBox(COLOR_NAMES[0]))
        self.grid.set(2, 8, CustomBox(COLOR_NAMES[0]))

        self.grid.wall_rect(2, 7, 3, 1)

        self.grid.wall_rect(6, 7, 1, 2)

        self.put_obj(Goal(), 8, 8)
        self.agent_pos = (1, 1)
        self.agent_dir = 0
        self.has_key = False
        self.door_unlocked = False
        self.prev_dist = np.linalg.norm(np.array(self.agent_pos) - np.array([8, 8]))

    def step(self, action):
        # Store previous position
        prev_pos = self.agent_pos
        
        # Perform the original step
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Get forward position
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos) if 0 <= fwd_pos[0] < self.grid.width and 0 <= fwd_pos[1] < self.grid.height else None

        
        # Handle interactions with objects
        if fwd_cell:
            if isinstance(fwd_cell, Key) and fwd_cell.color == COLOR_NAMES[0]:
                # Pick up key
                self.grid.set(*fwd_pos, None)
                self.carrying = fwd_cell
                info["picked_key"] = True
                
            if isinstance(fwd_cell, Door) and self.carrying and self.carrying.type == "key":
                # Unlock door with key
                fwd_cell.is_locked = False
                fwd_cell.is_open = True
                info["unlocked_door"] = True
            
            if isinstance(fwd_cell, Lava):
                # Penalize stepping in lava
                info["stepped_in_lava"] = True
                reward = -0.5

        # Check if reached goal
        current_cell = self.grid.get(*self.agent_pos)
        if current_cell and current_cell.type == "goal":
            terminated = True
            reward = self._reward()
        
        return obs, reward, terminated, truncated, info