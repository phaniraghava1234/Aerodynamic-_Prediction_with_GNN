"""Convert AirfRANS point clouds into PyTorch Geometric graph datasets.

Run:  python src/data_processing.py
Writes train.pt / val.pt / test.pt / norm_stats.pt to data/processed/.

Graph construction:
  - Nodes  = mesh points.
  - Edges  = Delaunay triangulation of the 2D node positions (connects
    physically adjacent cells). Known limitation: Delaunay also creates a few
    spurious edges through the airfoil interior; acceptable for a baseline.
  - Node features (8): min-max scaled position (2), z-scored inlet velocity (2),
    z-scored wall distance (1), surface normals (2), on-surface flag (1).
  - Edge features (3): displacement (dx, dy) and distance in scaled coordinates.
  - Targets (4): z-scored velocity (2), p/rho (1), nu_t (1).

Scalers are fitted on the training split only. norm_stats.pt stores everything
needed to de-normalize predictions back to physical units.
"""
from pathlib import Path

import airfrans as af
import numpy as np
import torch
from scipy.spatial import Delaunay
from torch_geometric.data import Data

PROJECT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT / "data" / "Dataset"
OUT = PROJECT / "data" / "processed"

N_VAL = 20  # simulations held out from the 200-sample 'scarce' train split

# Column indices in the raw (N, 12) arrays
POS, INLET, DIST, NORMALS, TARGETS, SURF = (
    slice(0, 2), slice(2, 4), slice(4, 5), slice(5, 7), slice(7, 11), slice(11, 12),
)


def fit_norm_stats(dataset):
    """Fit normalization statistics on the training simulations."""
    pos = np.concatenate([s[:, POS] for s in dataset])
    feats = np.concatenate([np.hstack([s[:, INLET], s[:, DIST]]) for s in dataset])
    targets = np.concatenate([s[:, TARGETS] for s in dataset])
    return {
        "pos_min": torch.tensor(pos.min(axis=0), dtype=torch.float),
        "pos_max": torch.tensor(pos.max(axis=0), dtype=torch.float),
        "feat_mean": torch.tensor(feats.mean(axis=0), dtype=torch.float),
        "feat_std": torch.tensor(feats.std(axis=0), dtype=torch.float),
        "target_mean": torch.tensor(targets.mean(axis=0), dtype=torch.float),
        "target_std": torch.tensor(targets.std(axis=0), dtype=torch.float),
    }


def delaunay_edges(pos):
    """Undirected edge_index (2, E) from Delaunay triangulation of 2D points."""
    tri = Delaunay(pos)
    s = tri.simplices  # (T, 3)
    pairs = np.concatenate([s[:, [0, 1]], s[:, [1, 2]], s[:, [0, 2]]], axis=0)
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    edge_index = np.concatenate([pairs, pairs[:, ::-1]], axis=0).T  # both directions
    return torch.tensor(edge_index.copy(), dtype=torch.long)


def build_graph(sim, stats, name):
    """Convert one (N, 12) simulation array into a PyG Data object."""
    assert sim.shape[1] == 12, f"expected 12 columns, got {sim.shape[1]} in {name}"
    pos_raw = sim[:, POS]

    # Normalize
    pos = (torch.tensor(pos_raw, dtype=torch.float) - stats["pos_min"]) / (
        stats["pos_max"] - stats["pos_min"]
    )
    feats = torch.tensor(np.hstack([sim[:, INLET], sim[:, DIST]]), dtype=torch.float)
    feats = (feats - stats["feat_mean"]) / stats["feat_std"]
    normals = torch.tensor(sim[:, NORMALS], dtype=torch.float)  # unit-scale, not scaled
    surf = torch.tensor(sim[:, SURF], dtype=torch.float)
    y = (torch.tensor(sim[:, TARGETS], dtype=torch.float) - stats["target_mean"]) / stats[
        "target_std"
    ]

    x = torch.cat([pos, feats, normals, surf], dim=1)  # (N, 8)

    edge_index = delaunay_edges(pos_raw)
    disp = pos[edge_index[1]] - pos[edge_index[0]]  # (E, 2) in scaled coords
    edge_attr = torch.cat([disp, disp.norm(dim=1, keepdim=True)], dim=1)  # (E, 3)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                pos=pos, surf=surf.squeeze(1).bool(), name=name)


def process_split(dataset, names, stats):
    return [build_graph(sim, stats, name) for sim, name in zip(dataset, names)]


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)

    train_raw, train_names = af.dataset.load(root=str(DATA_ROOT), task="scarce", train=True)
    test_raw, test_names = af.dataset.load(root=str(DATA_ROOT), task="scarce", train=False)

    # Hold out the last N_VAL simulations for validation
    val_raw, val_names = train_raw[-N_VAL:], train_names[-N_VAL:]
    train_raw, train_names = train_raw[:-N_VAL], train_names[:-N_VAL]

    stats = fit_norm_stats(train_raw)  # training split only — no leakage
    torch.save(stats, OUT / "norm_stats.pt")

    for split, (raw, names) in {
        "train": (train_raw, train_names),
        "val": (val_raw, val_names),
        "test": (test_raw, test_names),
    }.items():
        graphs = process_split(raw, names, stats)
        torch.save(graphs, OUT / f"{split}.pt")
        n_nodes = sum(g.num_nodes for g in graphs)
        print(f"{split}: {len(graphs)} graphs, {n_nodes} total nodes -> {OUT / f'{split}.pt'}")
