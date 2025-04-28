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
import heapq

class CustomBox(Key):
    def render(self, img):
        key_image = Image.open("assets/box.png").resize((img.shape[1], img.shape[0]))
        key_image = key_image.convert("RGB")
        img[:, :, :] = np.asarray(key_image)

class CustomSpill(Lava):
    def render(self, img):
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
        self.path = []  # Store the found path

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
        elif self.layout_id ==3:
            self.layout3(width, height)
        elif self.layout_id ==4:
            self.layout4(width, height)
        elif self.layout_id ==5:
            self.layout5(width, height)
        elif self.layout_id ==6:
            self.layout6(width, height)
        elif self.layout_id ==7:
            self.layout7(width, height)
        elif self.layout_id ==8:
            self.layout8(width, height)
        elif self.layout_id ==9:
            self.layout9(width, height)
        elif self.layout_id ==10:
            self.layout10(width, height)
        elif self.layout_id ==11:
            self.layout11(width, height)
        else:
            raise ValueError(f"Unknown layout_id: {self.layout_id}")
        
        # Find path after grid is generated
        self.path = self.find_path()
        print(f"Path to follow: {self.path}")

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
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        
        # Generate vertical separation wall
        for i in range(0, height):
            self.grid.set(5, i, Wall())
        
        # Place the door and key
        self.grid.set(5, 6, Door(COLOR_NAMES[0], is_locked=True))
        self.grid.set(3, 6, CustomBox(COLOR_NAMES[0]))

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Add lava tiles
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



    def a_star_search(self, start, goal):
        """A* pathfinding implementation"""
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Up, Right, Down, Left
        
        open_list = []
        heapq.heappush(open_list, (0 + heuristic(start, goal), 0, start))
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        while open_list:
            _, current_g_score, current = heapq.heappop(open_list)
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            
            for direction in directions:
                neighbor = (current[0] + direction[0], current[1] + direction[1])
                
                # Check bounds and obstacles
                if not (0 <= neighbor[0] < self.grid.width and 0 <= neighbor[1] < self.grid.height):
                    continue
                cell = self.grid.get(*neighbor)
                if cell and (cell.type == "wall" or isinstance(cell, (Wall, Lava))):  # Skip walls and lava
                    continue
                
                tentative_g_score = current_g_score + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_score[neighbor], tentative_g_score, neighbor))
        
        return []  # No path found

    def find_path(self):
        """Find path from agent to all boxes, pick them up, and then go to the goal."""
            
        start = self.agent_pos
        boxes = []
        goal_pos = None

        # Find all boxes and the goal
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                cell = self.grid.get(x, y)
                if cell:
                    if cell.type == "key":
                        boxes.append((x, y))
                    elif cell.type == "goal":
                        goal_pos = (x, y)

        if goal_pos is None:
            raise ValueError("Goal not found in the grid.")

        if not boxes:
            # No boxes, go directly to the goal
            path_to_goal = self.a_star_search(start, goal_pos)
            if not path_to_goal:
                raise ValueError("No path to goal found.")
            return path_to_goal

        # If there are boxes, go to each one and then to the goal
        path_to_goal = []  # To store the full path
        current_pos = start

        # Pick up all the boxes
        for box_pos in boxes:
            # Path from current position to the box
            path_to_box = self.a_star_search(current_pos, box_pos)
            if not path_to_box:
                raise ValueError(f"No path to box at {box_pos} found.")

            # Add path to the full path
            path_to_goal += path_to_box

            # "Pick up" the box (not implemented here, but you can add logic for this)
            current_pos = box_pos  # Now the agent is at the box's location

        # After picking up all boxes, head to the goal
        path_from_last_box_to_goal = self.a_star_search(current_pos, goal_pos)
        if not path_from_last_box_to_goal:
            raise ValueError("No path from last box to goal found.")

        # Add the final path to the goal
        path_to_goal += path_from_last_box_to_goal[1:]  # Skip the first position as it repeats

        return path_to_goal



    def step(self, action):
        # Store previous position
        prev_pos = self.agent_pos
        
        # Perform the original step
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Get forward position after movement and check bounds
        fwd_pos = self.front_pos
        fwd_cell = None
        if 0 <= fwd_pos[0] < self.grid.width and 0 <= fwd_pos[1] < self.grid.height:
            fwd_cell = self.grid.get(*fwd_pos)
        
        # Only process fwd_cell if it's within bounds
        if fwd_cell:
            if isinstance(fwd_cell, Key) and fwd_cell.color == COLOR_NAMES[0]:
                self.grid.set(*fwd_pos, None)
                self.carrying = fwd_cell
                self.has_key = True
                info["picked_key"] = True
                
            if isinstance(fwd_cell, Door) and self.has_key:
                fwd_cell.is_locked = False
                fwd_cell.is_open = True
                self.door_unlocked = True
                info["unlocked_door"] = True
            
            if isinstance(fwd_cell, Lava):
                info["stepped_in_lava"] = True
                # Still allow movement into lava
                self.agent_pos = fwd_pos

        # Check current cell for goal
        current_cell = self.grid.get(*self.agent_pos)
        if current_cell and current_cell.type == "goal":
            terminated = True
        
        # Update path if position changed
        if action > 0 and not np.array_equal(self.agent_pos, prev_pos):
            self.path = self.find_path()
        
        return obs, reward, terminated, truncated, info