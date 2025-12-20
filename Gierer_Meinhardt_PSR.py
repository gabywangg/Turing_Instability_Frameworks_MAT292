import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 100.0           # domain length
N = 200            # grid points
dx = 0.1
dt = 0.001
steps = 100000

# model parameters
a = 0.8  #a sets the response speed of the inhibitor. larger a means faster inhibitor reaction relative to the activator 
b = 0.2 # b sets how much inhibitor is generated per unit of activator.
D1 = 0.01 # activator diffusion
D2 = 0.20 # inhibitor diffusion

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

    if i % save_every == 0:
        psr_series.append(psr_2d(u, r=exclude_radius))
        times.append(i * dt)

# plot once at the end
plt.imshow(u, cmap="viridis", extent=[0, L, 0, L])
plt.title(f"u at t = {steps*dt:.2f} s")

plt.figure()
plt.plot(times, psr_series)
plt.xlabel("time (s)")
plt.ylabel("PSR")
plt.title(f"PSR vs time (every {save_every} steps)")
if psr_series:
    k = int(np.argmax(psr_series))
    print(f"[GM] Max PSR = {psr_series[k]:.4f} at t = {times[k]:.4f}s (step {k*save_every})")

plt.ioff()
plt.show()
