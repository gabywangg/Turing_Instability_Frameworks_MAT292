import numpy as np
import matplotlib.pyplot as plt
# make more indepth comments
# Parameters
L = 50.0           # domain length
N = 200            # number of grid points per dimension
dx = L / N         # spatial resolution
dt = 1e-3          # time step
steps = 1000000     # number of iterations

# Schnakenberg parameters
a = 0.2
b = 1.3
Du = 0.01
Dv = 0.05

# Initialize fields
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)

u = a + b + 0.01 * np.random.randn(N, N)   # initial u field
v = b / (a + b)**2 + 0.01 * np.random.randn(N, N)  # initial v field

# Laplacian operator (finite difference)
def laplacian(Z):
    return (
        -4*Z
        + np.roll(Z, 1, axis=0)
        + np.roll(Z, -1, axis=0)
        + np.roll(Z, 1, axis=1)
        + np.roll(Z, -1, axis=1)
    ) / dx**2

# Time stepping
for i in range(steps):
    Lu = laplacian(u)
    Lv = laplacian(v)
    
    du = Du * Lu + a - u + u**2 * v
    dv = Dv * Lv + b - u**2 * v

    u += dt * du
    v += dt * dv

    # Display progress
    if i % 100000 == 0:
        plt.imshow(u, cmap='viridis', extent=[0, L, 0, L])
        plt.title(f"u at step {i}")
        plt.colorbar()
        plt.savefig(f"u_step_{i}.png")
        plt.close()


# Final plot
plt.figure(figsize=(6, 5))
plt.imshow(u, cmap='viridis', extent=[0, L, 0, L])
plt.title("Final u pattern (Schnakenberg)")
plt.colorbar()
plt.show()