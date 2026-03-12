"""
Example: Visualizing the QPPG energy landscape.

Shows how wells activate/deactivate as mu changes (bifurcation).
Demonstrates the core physics: mu-controlled phase transitions
create or destroy attractor basins.

No API key required — runs entirely locally.
"""

import numpy as np
import torch
from qppg import QPPGSubstrate

# Create a 2D substrate for visualization
substrate = QPPGSubstrate(d=2, n_wells=4, seed=42, confinement=0.01, sigma_base=1.0)

# Place wells at known locations
with torch.no_grad():
    substrate.well_centers.data = torch.tensor([
        [2.0, 0.0],
        [-2.0, 0.0],
        [0.0, 2.0],
        [0.0, -2.0],
    ])
    substrate.base_depths.data = torch.tensor([3.0, 2.0, 1.0, 0.5])
    substrate.mu_sensitivity.data = torch.tensor([-1.0, 0.0, 1.0, 1.5])

# Sweep mu and count active wells
print("mu-sweep: counting attractor basins\n")
print(f"{'mu':>6} | {'Active Wells':>12} | {'Basins':>6}")
print("-" * 32)

test_points = torch.tensor([
    [2.0, 0.0], [-2.0, 0.0], [0.0, 2.0], [0.0, -2.0],
    [1.0, 1.0], [-1.0, -1.0], [0.5, -0.5], [-0.5, 0.5],
], dtype=torch.float32)

for mu in np.linspace(-3, 3, 13):
    substrate.mu = mu
    labels, n_basins, centers = substrate.find_attractors(test_points)
    # Count active wells (positive depth)
    depths = substrate.base_depths + substrate.mu_sensitivity * mu
    active = int((depths > 0).sum())
    print(f"{mu:6.1f} | {active:12d} | {n_basins:6d}")

print("\nAs mu increases, wells with positive mu_sensitivity activate")
print("and wells with negative mu_sensitivity deactivate.")
print("This is the bifurcation mechanism.")
