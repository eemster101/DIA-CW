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

    def find_path(self):
        start = self.agent_pos
        goal = (self.grid.width - 2, self.grid.height - 2)  # The goal is always at (width-2, height-2)
        path = a_star_search(start, goal, self.grid)
        return path