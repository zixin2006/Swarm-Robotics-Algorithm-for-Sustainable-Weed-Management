from cell_selection_base import strategic_cell_selection
from random_walk_base import random_cell_selection
from uniform_cell_base import uniform_cell_selection

strategic1 = []
random1 = []
sequential = []

nan = []
parameterTest = [0.02, 0.05, 0.08, 0.11, 0.14]
for p in parameterTest:
    for i in range(5):
        runtime = strategic_cell_selection(
    M = 32,  # Grid size
    N = 6, # Number of patches
    n = 6,  # Number of agents
    k_att = 0.1,  # Attractive force scaling factor
    k_rep = p,  # Repulsive force scaling factor
    d_0 = 8, # Distance threshold for repulsive force
    )
        nan.append(runtime)
    value = sum(nan) / len(nan)
    random1.append(value)
    print(value)
    nan = []
print(random1)