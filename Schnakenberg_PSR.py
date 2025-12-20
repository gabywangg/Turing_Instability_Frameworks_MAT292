import numpy as np
import matplotlib.pyplot as plt
# Parameters
L = 50.0           # domain length
N = 200            # number of grid points per dimension
dx = L / N         # grid spacing used in the finite difference formulas
dt = 1e-3          # time step for the forward Euler method
steps = 200000     # number of iterations

# Schnakenberg parameters
a = 0.1 #a sets the response speed of the inhibitor. larger a means faster inhibitor reaction relative to the activator 
b = 0.9 # b sets how much inhibitor is generated per unit of activator
Du = 0.01 # activator diffusion
Dv = 0.2 # inhibitor diffusion

save_every = 1000
exclude_radius = 5
psr_series = []
times = []

def psr_2d(field, r=5):
    peak_idx = np.unravel_index(np.argmax(field), field.shape)
    pi, pj = peak_idx
    peak_val = field[pi, pj]
    mask = np.ones(field.shape, dtype=bool)
    i0 = max(pi - r, 0); i1 = min(pi + r + 1, field.shape[0])
    j0 = max(pj - r, 0); j1 = min(pj + r + 1, field.shape[1])
    mask[i0:i1, j0:j1] = False
    sidelobe = field[mask]
    mu = float(np.mean(sidelobe))
    sigma = float(np.std(sidelobe))
    if sigma == 0.0:
        return float("inf") if peak_val > mu else 0.0
    return (float(peak_val) - mu) / sigma

# creates the spatial grid
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)

# initialize solutions fields
u = a + b + 0.01 * np.random.randn(N, N)   # initial u field, adding random perturbations to numerical progress
v = b / (a + b)**2 + 0.01 * np.random.randn(N, N)  # initial v field, adding random perturbations to numerical progress

# defines the discrete Laplacian
def laplacian(Z):
    return (
        -4*Z
        + np.roll(Z, 1, axis=0)
        + np.roll(Z, -1, axis=0)
        + np.roll(Z, 1, axis=1)
        + np.roll(Z, -1, axis=1)
    ) / dx**2

# time stepping loop
for i in range(steps):
    # spatial diffusion at the current time step
    Lu = laplacian(u) # discrete Laplacian operator to the u field
    Lv = laplacian(v) # discrete Laplacian operator to the v field

    # evaluates the reaction–diffusion equations, forms the right side of the PDE
    du = Du * Lu + a - u + u**2 * v
    dv = Dv * Lv + b - u**2 * v

    # time integration, forward euler method
    u += dt * du
    v += dt * dv

    if i % save_every == 0:
        psr_series.append(psr_2d(u, r=exclude_radius))
        times.append(i * dt)

    # display progress, periodic visualization to evaluate the pattern
    if i % 100000 == 0:
        plt.imshow(u, cmap='viridis', extent=[0, L, 0, L])
        plt.title(f"u at step {i}")
        plt.colorbar()
        plt.savefig(f"u_step_{i}.png")
        plt.close()

    # loop repeats for the number of steps inputted above


# Final plot
plt.figure(figsize=(6, 5))
plt.imshow(u, cmap='viridis', extent=[0, L, 0, L])
plt.title(f"u at t = {steps*dt:.2f} s")
plt.colorbar()

plt.figure()
plt.plot(times, psr_series)
plt.xlabel("time (s)")
plt.ylabel("PSR")
plt.title(f"PSR vs time (every {save_every} steps)")
if psr_series:
    k = int(np.argmax(psr_series))
    print(f"[Schnakenberg] Max PSR = {psr_series[k]:.4f} at t = {times[k]:.4f}s (step {k*save_every})")

plt.show()
