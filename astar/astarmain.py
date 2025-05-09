from environment import WarehouseEnv
from minigrid.wrappers import RGBImgPartialObsWrapper
from minigrid.core.actions import Actions
import time
from PIL import Image
import tkinter as tk
from tkinter import simpledialog


def visualize_path(env, path):
    """Create visualization from grid data"""
    # Create blank image
    img = Image.new('RGB', (env.grid.width*32, env.grid.height*32), (255, 255, 255))
    pixels = img.load()
    
    # Draw grid (walls)
    for i in range(env.grid.width):
        for j in range(env.grid.height):
            cell = env.grid.get(i, j)
            if cell and cell.type == "wall":
                # Draw wall
                for x in range(i*32, (i+1)*32):
                    for y in range(j*32, (j+1)*32):
                        pixels[x,y] = (100, 100, 100)  # Grey color for wall
    
    # Highlight path (red)
    for pos in path[:-1]:  # All positions except the last
        x = pos[0] * 32 + 16
        y = pos[1] * 32 + 16
        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                if 0 <= x+dx < img.width and 0 <= y+dy < img.height:
                    pixels[x+dx, y+dy] = (255, 0, 0)  # Red

    # Highlight final agent position (green)
    final_pos = path[-1]
    x = final_pos[0] * 32 + 16
    y = final_pos[1] * 32 + 16
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            if 0 <= x+dx < img.width and 0 <= y+dy < img.height:
                pixels[x+dx, y+dy] = (0, 255, 0)  # Green
    
    return img

def ask_layout_id():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    layout_id = simpledialog.askinteger("Select Layout", "Enter the layout ID you want to use (1-11):")
    
    root.destroy()
    return layout_id

def get_direction_vector(dir):
    # Map dir to (dx, dy): 0=right, 1=down, 2=left, 3=up
    return [(1, 0), (0, 1), (-1, 0), (0, -1)][dir]

def get_required_action(agent_pos, agent_dir, next_pos):
    dx = next_pos[0] - agent_pos[0]
    dy = next_pos[1] - agent_pos[1]

    # Convert that to desired direction index
    if dx == 1 and dy == 0:
        desired_dir = 0  # right
    elif dx == 0 and dy == 1:
        desired_dir = 1  # down
    elif dx == -1 and dy == 0:
        desired_dir = 2  # left
    elif dx == 0 and dy == -1:
        desired_dir = 3  # up
    else:
        return None  # Shouldn't happen

    # Decide on the action needed to face the desired direction
    if agent_dir == desired_dir:
        return Actions.forward
    elif (agent_dir - desired_dir) % 4 == 1:
        return Actions.left
    else:
        return Actions.right

def test_warehouse():
    layout_id = ask_layout_id() 
    env = WarehouseEnv(size=10, layout_id=layout_id, render_mode="human")
    wrapped_env = RGBImgPartialObsWrapper(env)
    obs, _ = wrapped_env.reset()

    # Access the unwrapped environment for agent state
    base_env = wrapped_env.unwrapped
    path = base_env.path

    for target_pos in path[1:]:  # Start from second because agent is at path[0]
        while True:
            agent_pos = base_env.agent_pos
            agent_dir = base_env.agent_dir

            if tuple(agent_pos) == tuple(target_pos):
                break  # Move to next target in path

            action = get_required_action(agent_pos, agent_dir, target_pos)

            if action is not None:
                obs, reward, done, truncated, info = wrapped_env.step(action)
                wrapped_env.render()  # Render each step
                time.sleep(0.2)  # Slow down for visual feedback
                
                if done:
                    print("Reached the goal!")
                    break
            else:
                print(f"Unexpected move from {agent_pos} to {target_pos}")
                break

    print("Path followed successfully.")


    path_img = visualize_path(base_env, path)
    path_img.show()



if __name__ == "__main__":
    test_warehouse()