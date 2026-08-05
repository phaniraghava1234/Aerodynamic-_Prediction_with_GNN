"""Smoke tests on synthetic data mimicking the AirfRANS (N, 12) layout.

Run:  python tests/test_pipeline.py
Verifies: normalization fitting, Delaunay graph construction, feature shapes,
DataLoader batching, and a model forward + backward pass.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_processing import build_graph, fit_norm_stats
from models import IN_FEATURES, OUT_FEATURES, BaselineGNN


def synthetic_sim(n=500, seed=0):
    rng = np.random.default_rng(seed)
    sim = np.zeros((n, 12))
    sim[:, 0:2] = rng.uniform([-2, -1.5], [4, 1.5], size=(n, 2))   # position
    sim[:, 2:4] = [45.0, 3.0]                                       # inlet velocity
    sim[:, 4] = np.abs(sim[:, 1])                                   # fake wall distance
    surf = rng.random(n) < 0.05
    sim[surf, 5:7] = rng.normal(size=(surf.sum(), 2))               # normals on surface
    sim[:, 7:11] = rng.normal(size=(n, 4)) * [40, 5, 500, 0.001]    # targets
    sim[:, 11] = surf
    return sim


sims = [synthetic_sim(500, s) for s in range(3)]
stats = fit_norm_stats(sims)
assert stats["target_std"].shape == (4,)

graphs = [build_graph(s, stats, f"sim_{i}") for i, s in enumerate(sims)]
g = graphs[0]
assert g.x.shape == (500, IN_FEATURES), g.x.shape
assert g.y.shape == (500, OUT_FEATURES), g.y.shape
assert g.edge_index.shape[0] == 2 and g.edge_index.max() < 500
assert g.edge_attr.shape == (g.edge_index.shape[1], 3)
# undirected: every edge present in both directions
e = set(map(tuple, g.edge_index.T.tolist()))
assert all((b, a) in e for a, b in e)
# scaled positions in [0, 1], normalized targets ~N(0, 1)
assert g.pos.min() >= 0 and g.pos.max() <= 1
assert abs(torch.cat([gr.y for gr in graphs]).mean().item()) < 0.01

loader = DataLoader(graphs, batch_size=2, shuffle=True)
model = BaselineGNN()
batch = next(iter(loader))
out = model(batch)
assert out.shape == (batch.num_nodes, OUT_FEATURES), out.shape
torch.nn.functional.mse_loss(out, batch.y).backward()
assert all(p.grad is not None for p in model.parameters())

n_params = sum(p.numel() for p in model.parameters())
print(f"All checks passed. Batch of {batch.num_graphs} graphs, "
      f"{batch.num_nodes} nodes, {batch.num_edges} edges. "
      f"Model: {n_params:,} parameters.")
