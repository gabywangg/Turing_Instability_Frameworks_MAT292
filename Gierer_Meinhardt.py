import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 100.0           # domain length
N = 200            # grid points
dx = L / N
dt = 1e-3
steps = 200000

# model parameters
a = 0.1  #a sets the response speed of the inhibitor. larger a means faster inhibitor reaction relative to the activator 
b = 0.8 # b sets how much inhibitor is generated per unit of activator.
D1 = 0.1 # activator diffusion
D2 = 0.1  # inhibitor diffusion

# initialize fields
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)

u = 1 + 0.01 * np.random.randn(N, N)
v = a * b + 0.01 * np.random.randn(N, N)

# laplacian (periodic BC) 
def laplacian(Z):
    return (
        -4*Z
        + np.roll(Z, 1, axis=0)
        + np.roll(Z, -1, axis=0)
        + np.roll(Z, 1, axis=1)
        + np.roll(Z, -1, axis=1)
    ) / dx**2

# Time stepping
plt.ion() #matplotlib functions
# turn OFF interactive plotting
# plt.ion()

for i in range(steps):
    Lu = laplacian(u)
    Lv = laplacian(v)

    du = 1 + (u**2 / v) - u + D1 * Lu
    dv = a * b * (u**2) - a * v + D2 * Lv

    u += dt * du
    v += dt * dv

# plot once at the end
plt.imshow(u, cmap="viridis", extent=[0, L, 0, L])
plt.title(f"u at t = {steps*dt:.2f} s")

plt.ioff()
plt.show()
