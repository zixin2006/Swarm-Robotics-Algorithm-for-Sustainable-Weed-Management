import numpy as np

def random_cell_selection(M, n):
    # Initialize the grid to track visits
    grid = np.zeros((M, M), dtype=int)
    
    # Initialize bots' positions and distances traveled
    bots = [{'pos': np.random.randint(0, M, size=2), 'distance': 0} for _ in range(n)]
    
    # Directions: Up, Down, Left, Right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Continue moving until the entire grid is covered
    while not np.all(grid > 0):
        for bot in bots:
            # Mark the current cell as visited
            x, y = bot['pos']
            grid[x, y] += 1
            
            # Choose a random direction and move the bot
            dx, dy = directions[np.random.randint(0, 4)]
            new_x, new_y = x + dx, y + dy
            
            # Ensure the new position is within the grid boundaries
            new_x = max(0, min(M - 1, new_x))
            new_y = max(0, min(M - 1, new_y))
            
            # Update bot's position and increment distance traveled
            bot['pos'] = np.array([new_x, new_y])
            bot['distance'] += 1

    # Find the maximum distance traveled by the bots
    max_distance = max(bot['distance'] for bot in bots)

    return max_distance

# Parameters
M = 48  # Grid size
n = 4   # Number of bots

rlist = []

# Run the simulation
for i in range(20):
    value = random_cell_selection(M, n)
    rlist.append(value)
    print(value)

result = sum(rlist) / len(rlist)
print(result)
rlist = []
