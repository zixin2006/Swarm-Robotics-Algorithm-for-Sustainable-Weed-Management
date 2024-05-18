import numpy as np

# Define parameters
M = 16  # Grid size
N = 6 # Number of patches
n = 3  # Number of agents
k_att = 0.1  # Attractive force scaling factor
k_rep = 0.02  # Repulsive force scaling factor
d_0 = 4  # Distance threshold for repulsive force

def strategic_cell_selection(M, N, n, k_att, k_rep, d_0):

    # Initialize a MxM grid with 2d Gaussian distribution for weed density
    weed_density = np.random.rand(M, M)

    def check_overlap(existing_patches, new_patch):
        """Check if the new_patch overlaps with existing patches."""
        for patch in existing_patches:
            if not (new_patch[1] < patch[0] or new_patch[0] > patch[1] or
                    new_patch[3] < patch[2] or new_patch[2] > patch[3]):
                return True  # Overlap found
        return False

    def generate_non_overlapping_patches(M, N):
        """Generate N non-overlapping patches within an MxM grid."""
        weed_density = np.ones((M, M))
        patches_info = []  # List to store patches' start_x, end_x, start_y, end_y

        for _ in range(N):
            placed = False
            attempts = 0
            while not placed and attempts < 100:  # Limit attempts to prevent infinite loops
                patch_width = np.random.randint(1, M // 2)  # Adjust size limits as needed
                patch_height = np.random.randint(1, M // 2)
                start_x = np.random.randint(0, M - patch_width)
                start_y = np.random.randint(0, M - patch_height)
                new_patch = (start_x, start_x + patch_width, start_y, start_y + patch_height)
                
                # Check for overlap
                if not check_overlap(patches_info, new_patch):
                    # If no overlap, place the patch
                    weed_density[start_x:new_patch[1], start_y:new_patch[3]] = np.random.uniform(1, 2, size=[patch_width, patch_height])
                    patches_info.append(new_patch)
                    placed = True
                attempts += 1

            return weed_density

    # Generate weed density map
    weed_density = generate_non_overlapping_patches(M, N)

    # Initialize agents at random cells
    agents = [{'pos': np.random.rand(2) * M, 'covered': set()} for _ in range(n)]

    def calculate_attractive_force(agent_pos, cell_pos, k_att):
        d_ij = cell_pos - agent_pos
        dist_ij = np.linalg.norm(d_ij)
        if dist_ij == 0:  # Avoid division by zero
            return np.array([0.0, 0.0])
        return -k_att / (1 + dist_ij**2)**2 * d_ij

    def calculate_repulsive_force(agent_pos, covered_cells, k_rep, d_0):
        force_rep = np.zeros(2)
        for covered_cell in covered_cells:
            d_ij = np.array(covered_cell) - agent_pos
            dist_ij = np.linalg.norm(d_ij)
            if dist_ij < d_0 and dist_ij != 0:
                force_rep += k_rep * ((1 / dist_ij - 1 / d_0)**3 / dist_ij**2) * d_ij
        return force_rep

    def get_utility(force, p):
        theta_ij = np.arctan2(force[1], force[0])
        return (1 / np.pi) * (1 / (p + theta_ij**2))

    def move_agent(agent, grid, covered_cells, M):
        best_utility = -np.inf
        best_move = None
        hierarchy = 1
        
        while best_move is None and hierarchy <= M:
            for dx in range(-hierarchy, hierarchy + 1):
                for dy in range(-hierarchy, hierarchy + 1):
                    if abs(dx) + abs(dy) == hierarchy:
                        neighbor = (agent['pos'][0] + dx, agent['pos'][1] + dy)
                        if 0 <= neighbor[0] < M and 0 <= neighbor[1] < M and neighbor not in covered_cells:
                            attractive_force = calculate_attractive_force(agent['pos'], neighbor, k_att)
                            repulsive_force = calculate_repulsive_force(agent['pos'], covered_cells, k_rep, d_0)
                            total_force = attractive_force + repulsive_force
                            utility = get_utility(total_force, grid[int(neighbor[0]), int(neighbor[1])])
                            if utility > best_utility:
                                best_utility = utility
                                best_move = neighbor
            hierarchy += 1
        return best_move, hierarchy

    # Initialize runtime for each agent
    agents_runtime = [0] * n

    # Run simulation until all cells are covered
    covered_cells = set()
    while len(covered_cells) < M * M:
        for i, agent in enumerate(agents):
            # Move agent to the best cell based on utility
            best_cell, hierarchy = move_agent(agent, weed_density, covered_cells, M)
            if best_cell:
                # Update agent's position and mark the cell as covered
                agent['pos'] = np.array(best_cell)
                covered_cells.add(best_cell)
                # Update individual agent runtime including the hierarchy time penalty
                agents_runtime[i] += 2 + (hierarchy - 1) * 3

    # Return the total runtime of the agent that finished last
    total_runtime = max(agents_runtime)

    return total_runtime

rlist = []
final = []

result = strategic_cell_selection(
M = 32,  # Grid size
N = 6, # Number of patches
n = 10,  # Number of agents
k_att = 0.1,  # Attractive force scaling factor
k_rep = 0.02,  # Repulsive force scaling factor
d_0 = 5, # Distance threshold for repulsive force
)
print(result)
