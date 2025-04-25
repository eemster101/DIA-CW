import heapq

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
