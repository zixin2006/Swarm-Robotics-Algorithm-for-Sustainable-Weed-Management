
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmap = plt.get_cmap('viridis')
new_cmap = truncate_colormap(cmap, 0.15, 0.85)

# Function to calculate Manhattan distance from the center of a 5x5 matrix
def manhattan_distance_from_center(x, y):
    center_x, center_y = 2, 2  # Center of a 5x5 matrix
    return abs(x - center_x) + abs(y - center_y)

matrix = np.zeros((5, 5))

# Update the matrix to use Manhattan distance
for i in range(5):
    for j in range(5):
        matrix[i, j] = manhattan_distance_from_center(i, j)

# Set the center value to NaN again for visual purposes
matrix[2, 2] = np.nan

# Visualizing the updated matrix using Manhattan distance
plt.figure(figsize=(8, 8))
cax = plt.matshow(matrix, cmap=new_cmap, fignum=1)
plt.colorbar(cax, label='Manhattan Distance from Center')
plt.title('Distance Hierarchy in a Sample Field')

# Annotate each cell with the corresponding Manhattan distance
for (i, j), value in np.ndenumerate(matrix):
    if np.isnan(value):
        plt.text(j, i, 'Agent m', va='center', ha='center')
    else:
        plt.text(j, i, int(value), va='center', ha='center')

plt.show()
