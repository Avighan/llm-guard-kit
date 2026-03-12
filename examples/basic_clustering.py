"""
Example: Basic QPPG clustering on synthetic data.

Demonstrates the core QPPGOnlineClusterer on a simple 2D dataset
with 3 clusters that appear one at a time (streaming novelty detection).

No API key required — runs entirely locally.
"""

import numpy as np
from qppg import QPPGOnlineClusterer

# Create synthetic data: 3 clusters appearing in 3 phases
np.random.seed(42)

phase1 = np.random.randn(50, 8) * 0.5 + np.array([2, 0, 0, 0, 0, 0, 0, 0])
phase2 = np.random.randn(50, 8) * 0.5 + np.array([-2, 0, 0, 0, 0, 0, 0, 0])
phase3 = np.random.randn(50, 8) * 0.5 + np.array([0, 3, 0, 0, 0, 0, 0, 0])

# Initialize clusterer
clusterer = QPPGOnlineClusterer(d=8, n_wells=10, seed=42)

# Stream data in phases
for i, (data, name) in enumerate([(phase1, "Phase 1"), (phase2, "Phase 2"), (phase3, "Phase 3")]):
    labels = clusterer.process_batch(data, phase_idx=i)
    print(f"{name}: {clusterer.get_n_clusters()} clusters detected")

# Check for novelty events (bifurcations)
events = clusterer.get_novelty_events()
print(f"\nNovelty events (bifurcations): {len(events)}")
for phase, old_n, new_n in events:
    print(f"  Phase {phase}: {old_n} -> {new_n} clusters")
