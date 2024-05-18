import numpy as np
import matplotlib.pyplot as plt

def gaussian_2d(x, y, x0, y0, sigma_x, sigma_y, amplitude):
    """2D Gaussian function centered at (x0, y0)"""
    return amplitude * np.exp(-(((x-x0)**2 / (2*sigma_x**2)) + ((y-y0)**2 / (2*sigma_y**2))))

# Grid settings
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, y)

# Potential field calculation
V = np.zeros_like(X)
# Parameters for Gaussian components: (x0, y0, sigma_x, sigma_y, amplitude)
gaussians = [
    (-0.5, 0.4, 0.1, 0.1, 0.11),  # Convex region
    (0.38, 0.2, 0.15, 0.15, -0.12),  # Concave region
    (-0.75, 0.75, 0.08, 0.08, -0.15), # Concave region
    (0.5, -0.6, 0.1, 0.1, 0.2),
    (-0.48, -0.3, 0.15, 0.15, -0.1),
    (-0.3, -0.4, 0.12, 0.12, -0.1) # Another convex region
]

for x0, y0, sigma_x, sigma_y, amplitude in gaussians:
    V += gaussian_2d(X, Y, x0, y0, sigma_x, sigma_y, amplitude)

# Visualize the potential field
plt.figure(figsize=(10, 8))
contour = plt.contourf(X, Y, V, levels=350, cmap='viridis')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar(contour)
plt.grid(True)
#plt.title('')
plt.show()
