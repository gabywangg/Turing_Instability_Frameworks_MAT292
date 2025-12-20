# import libraries
import numpy as np
import matplotlib.pyplot as plt

# Spatial doman and discretization settings
L = 100         # domain length
N = 100         # number of grid points per dimension
dx = 0.1        # grid spacing used in the finite difference formulas

# time discretization
dt = 0.01           # time step for the forward Euler method
steps = 20000       # number of iterations

# !!!FOR TA: SHOULD ONLY CHANGE THESE FOUR FOR DIFFERENT PATTERNS!!!
# Brusselator parameters settings, labeled as project proposal
a = 1.0
b = 1.7
# Setting diffusion coefficients
Du = 0.01
Dv = 0.2


# PSR Logging, saves every 1000 euler steps
save_every = 1000
exclude_radius = 5
psr_series = []
times = []

def psr_2d(field, r=5):
    """
    Computing peak sharpness ratio for the 2D scalar field.
    i.e. u, activator field.

    This function operates as follows:
    Finds global maximum peak.
    Creates a radius around the peak.
    Computes mean standard deviation of sidelobes
    Computers PSR following PSR = (peak - mu) / sigma
    """
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

# initialize solution fields (small random perturbations around the homogeneous steady state)
u0 = a
v0 = b / a if a != 0 else 0.0
u = u0 + 0.01 * np.random.randn(N, N)
v = v0 + 0.01 * np.random.randn(N, N)

# defines the discrete Laplacian
def laplacian(Z):
    """
    Discrete 2D laplacian implemented with a 5-point stecil
    with periodic boundary conditions.
    """
    return (
        -4*Z
        + np.roll(Z, 1, axis=0)
        + np.roll(Z, -1, axis=0)
        + np.roll(Z, 1, axis=1)
        + np.roll(Z, -1, axis=1)
    ) / dx**2

# time stepping loop (Eulers Method)
for i in range(steps):
    # spatial diffusion at the current time step
    Lu = laplacian(u) # discrete Laplacian operator to the u field
    Lv = laplacian(v) # discrete Laplacian operator to the v field

    # evaluates the reaction–diffusion equations (reaction terms)
    fu = a - (b + 1.0) * u + (u**2) * v
    fv = b * u - (u**2) * v

    # Combines into RHS of PDE
    du = Du * Lu + fu
    dv = Dv * Lv + fv

    # time integration, forward Euler method
    u += dt * du
    v += dt * dv

    # Record PSR metric periodically for graphing later
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



# Final concentration activator field
plt.figure(figsize=(6, 5))
plt.imshow(u, cmap='viridis', extent=[0, L, 0, L])
plt.title(f"u at t = {steps*dt:.2f} s")
plt.colorbar()

# PSR vs time graph
plt.figure()
plt.plot(times, psr_series)
plt.xlabel("time (s)")
plt.ylabel("PSR")
plt.title(f"PSR vs time (every {save_every} steps)")
if psr_series:
    k = int(np.argmax(psr_series))
    print(f"[Brusselator] Max PSR = {psr_series[k]:.4f} at t = {times[k]:.4f}s (step {k*save_every})")

plt.show()
