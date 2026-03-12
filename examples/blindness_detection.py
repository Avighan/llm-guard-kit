"""
Example: Blindness detection on synthetic data.

Demonstrates BlindnessDetector classifying points as FAMILIAR (near wells),
NOVEL (new attractor), or BLIND (outside all basins). Also shows
EnergyAccountant tracking budget pressure.

No API key required — runs entirely locally.
"""

import numpy as np
import torch
from qppg import QPPGSubstrate, BlindnessDetector, EnergyAccountant, InputClassification

# --- Part 1: Setup substrate with 3 known wells ---
substrate = QPPGSubstrate(d=8, n_wells=6, seed=42, confinement=0.001, sigma_base=1.5)

well_positions = np.array([
    [3, 0, 0, 0, 0, 0, 0, 0],
    [-3, 0, 0, 0, 0, 0, 0, 0],
    [0, 3, 0, 0, 0, 0, 0, 0],
], dtype=np.float32)

with torch.no_grad():
    substrate.well_centers.data[:3] = torch.tensor(well_positions)
    substrate.well_centers.data[3:] = 100.0  # push unused wells far away
    substrate.base_depths.data[:3] = 3.0
    substrate.base_depths.data[3:] = 0.0

# --- Part 2: Generate test points ---
np.random.seed(42)

# In-distribution: near the 3 wells
in_dist = np.vstack([
    well_positions[0] + np.random.randn(30, 8) * 0.3,
    well_positions[1] + np.random.randn(30, 8) * 0.3,
    well_positions[2] + np.random.randn(30, 8) * 0.3,
]).astype(np.float32)

# Out-of-distribution: far from all wells
ood = (np.random.randn(30, 8) * 2.0 + np.array([0, 0, 0, 0, 10, 0, 0, 0])).astype(np.float32)

all_points = np.vstack([in_dist, ood])
ground_truth = np.array([0]*90 + [1]*30)  # 0=familiar, 1=blind

# --- Part 3: Run blindness detection ---
detector = BlindnessDetector(
    substrate=substrate,
    blindness_threshold=0.5,
    velocity_threshold=1e-3,
)

result = detector.detect(all_points, known_centers=well_positions)

print("=== Blindness Detection Results ===")
print(f"Total points:    {len(all_points)}")
print(f"Familiar:        {len(result['familiar_indices'])}")
print(f"Blind:           {len(result['blind_indices'])}")
print(f"Blind fraction:  {result['blind_fraction']:.1%}")
print(f"Mean blindness:  {result['mean_blindness']:.3f}")
print(f"Attractors found: {result['n_attractors']}")

# Check accuracy
true_blind = set(range(90, 120))
detected_blind = set(result['blind_indices'])
tp = len(true_blind & detected_blind)
fp = len(detected_blind - true_blind)
fn = len(true_blind - detected_blind)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
print(f"\nPrecision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")

# --- Part 4: Energy accounting ---
print("\n=== Energy Accounting Demo ===")
accountant = EnergyAccountant(
    total_budget=30.0,
    maintenance_cost_per_attractor=2.0,
    exploration_cost=8.0,
)
accountant.register_attractors(well_positions)

for cycle in range(6):
    state = accountant.tick()
    explore_ok = "yes" if state.can_explore else "NO"
    consolidate = " [CONSOLIDATE]" if state.consolidation_triggered else ""
    print(f"  Cycle {cycle}: {state.n_attractors} attractors, "
          f"remaining={state.remaining:.1f}, pressure={state.pressure:.2f}, "
          f"can_explore={explore_ok}{consolidate}")

    if state.consolidation_triggered:
        new_centers, _, shed = accountant.consolidate(
            well_positions, depths=np.array([3.0, 1.0, 2.0]),
            merge_threshold=10.0
        )
        print(f"    -> Consolidated: {len(well_positions)} -> {len(new_centers)} attractors, shed {len(shed)}")
        break
