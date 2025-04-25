from environment1 import WarehouseEnv
import numpy as np
from PIL import Image
import heapq

def visualize_path(env, path):
    """Render the environment with the path highlighted"""
    # Create a frame with agent's current view
    frame = env.render()
    
    # Convert to PIL Image for manipulation
    img = Image.fromarray(frame)
    pixels = img.load()
    
    # Highlight path (red dots)
    for pos in path:
        # Convert grid coordinates to pixel coordinates
        # Note: This depends on your rendering scale - adjust as needed
        x = pos[0] * 32 + 16  # 32 is tile size, 16 is center
        y = pos[1] * 32 + 16
        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                if 0 <= x+dx < img.width and 0 <= y+dy < img.height:
                    pixels[x+dx, y+dy] = (255, 0, 0)  # Red
    
    return img

def a_star_search(start, goal, grid):
    # Heuristic function: Manhattan distance
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    # Directions (Up, Right, Down, Left)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
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
            
            if not (0 <= neighbor[0] < grid.width and 0 <= neighbor[1] < grid.height):  # Out of bounds
                continue
            if isinstance(grid.get(*neighbor), Wall):  # Skip walls
                continue
            
            tentative_g_score = current_g_score + 1  # Cost to move is always 1
            
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f_score[neighbor], tentative_g_score, neighbor))
    
    return []  # No path found

def main():
    # Initialize environment
    env = WarehouseEnv(
        size=10,
        layout_id=2,  # Use layout with obstacles
        render_mode="human"
    )
    env.reset()
    
    # Find path using A*
    path = env.find_path()
    
    if path:
        print("Path found with length:", len(path))
        print("Path coordinates:", path)
        
        # Visualize the path
        img = visualize_path(env, path)
        img.show()
        
        # Step through the path manually
        for pos in path[1:]:  # Skip starting position
            # Calculate required action to reach next position
            current_pos = env.agent_pos
            dx = pos[0] - current_pos[0]
            dy = pos[1] - current_pos[1]
            
            # Determine action (simplified)
            if dx == 1:
                action = env.actions.right
                env.step(env.actions.forward)
                env.step(action)
            elif dx == -1:
                action = env.actions.left
                env.step(env.actions.forward)
                env.step(action)
            elif dy == 1:
                env.step(env.actions.forward)
            elif dy == -1:
                env.step(env.actions.left)
                env.step(env.actions.left)
                env.step(env.actions.forward)
            
            env.render()  # Update visualization
    else:
        print("No path found to goal!")

if __name__ == "__main__":
    main()