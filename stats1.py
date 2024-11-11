import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib

# Data points
x_strategic = np.array([8, 16, 24, 32, 40, 48])
y_strategic = np.array([82, 327, 734, 1295, 1635, 2918])

x_random = np.array([8, 16, 24, 32, 40, 48])
y_random = np.array([200, 953, 3592, 6122, 10504, 17387])

# Linear regression for Strategic, n=4
coefficients_strategic = np.polyfit(x_strategic, y_strategic, 1)
linear_regression = np.poly1d(coefficients_strategic)

# Exponential regression for Random, n=4
# Transform for exponential fit: y = c * exp(dx) => log(y) = log(c) + dx
def exponential_model(x, d, log_c):
    return log_c + d * x

log_y_random = np.log(y_random)
params, _ = curve_fit(exponential_model, x_random, log_y_random)
d, log_c = params
exponential_regression = lambda x: np.exp(log_c) * np.exp(d * x)



# Generating smoother curves for the fits
x_smooth = np.linspace(min(x_strategic), max(x_strategic), 300)
y_strategic_smooth = linear_regression(x_smooth)
y_random_smooth = exponential_regression(x_smooth)


# Attempt to set Calibri as the default font if available
matplotlib.rcParams['font.sans-serif'] = "Calibri"


# Plotting the results with smoother curves
plt.figure(figsize=(10, 6))

# Strategic data and linear fit
plt.scatter(x_strategic, y_strategic, color='blue', label='Strategic, $n=4$ Data')
plt.plot(x_smooth, y_strategic_smooth, 'b--', label=f'Linear Fit: $y = {coefficients_strategic[0]:.2f}x {coefficients_strategic[1]:+.2f}$')

# Random data and exponential fit
plt.scatter(x_random, y_random, color='red', label='Random, $n=4$ Data')
plt.plot(x_smooth, y_random_smooth, 'r--', label=f'Exponential Fit: $y = {np.exp(log_c):.2f}e^{{0.107x}}$')

plt.title('Comparison of Weeding Mechanism with Fits')
plt.xlabel('$M$')
plt.ylabel('Runtime of Full Coverage (s)')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

plt.show()