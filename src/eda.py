"""Exploratory data analysis of the AirfRANS 'scarce' training split.

Run:  python src/eda.py
Outputs plots and summary statistics to results/eda/.

Column layout of each (N, 12) simulation array (airfrans.dataset.load):
  0-1  position x, y            [m]
  2-3  inlet velocity U_x, U_y  [m/s]
  4    distance to airfoil      [m]
  5-6  surface normals n_x, n_y (0 off-surface)
  7-8  velocity u, v            [m/s]        <- target
  9    pressure / density       [m^2/s^2]    <- target
  10   turbulent kin. viscosity [m^2/s]      <- target
  11   on-airfoil boolean
"""
import json
from pathlib import Path

import airfrans as af
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay

PROJECT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT / "data" / "Dataset"
OUT = PROJECT / "results" / "eda"

TARGET_NAMES = ["velocity_x [m/s]", "velocity_y [m/s]", "p/rho [m2/s2]", "nu_t [m2/s]"]
TARGET_COLS = [7, 8, 9, 10]


def plot_sample_meshes(dataset, names, n_samples=4):
    """Scatter plots of a few point clouds, colored by wall distance."""
    idx = np.linspace(0, len(dataset) - 1, n_samples, dtype=int)
    fig, axes = plt.subplots(n_samples, 2, figsize=(14, 4 * n_samples))
    for row, i in enumerate(idx):
        sim = dataset[i]
        # Full domain
        ax = axes[row, 0]
        s = ax.scatter(sim[:, 0], sim[:, 1], c=sim[:, 4], s=0.5, cmap="viridis")
        ax.set_title(f"{names[i]} — full domain ({sim.shape[0]} nodes)")
        ax.set_aspect("equal")
        fig.colorbar(s, ax=ax, label="wall distance [m]")
        # Zoom on airfoil
        ax = axes[row, 1]
        s = ax.scatter(sim[:, 0], sim[:, 1], c=sim[:, 4], s=0.5, cmap="viridis")
        surf = sim[sim[:, 11] == 1]
        ax.scatter(surf[:, 0], surf[:, 1], c="red", s=1, label="surface nodes")
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.4, 0.4)
        ax.set_aspect("equal")
        ax.set_title("airfoil zoom")
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "sample_meshes.png", dpi=150)
    plt.close(fig)


def plot_target_distributions(dataset):
    """Histograms and box plots of the 4 target fields over all training nodes."""
    all_targets = np.concatenate([sim[:, TARGET_COLS] for sim in dataset], axis=0)
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for j, name in enumerate(TARGET_NAMES):
        axes[0, j].hist(all_targets[:, j], bins=100)
        axes[0, j].set_title(name)
        axes[0, j].set_yscale("log")
        axes[1, j].boxplot(all_targets[:, j], showfliers=False)
        axes[1, j].set_title(f"{name} (no outliers)")
    fig.suptitle("Target field distributions — 'scarce' training split")
    fig.tight_layout()
    fig.savefig(OUT / "target_distributions.png", dpi=150)
    plt.close(fig)
    return all_targets


def graph_stats(sim):
    """Delaunay-graph statistics for one simulation (preview of Phase 3)."""
    tri = Delaunay(sim[:, :2])
    edges = set()
    for simplex in tri.simplices:
        for a in range(3):
            for b in range(a + 1, 3):
                edges.add((min(simplex[a], simplex[b]), max(simplex[a], simplex[b])))
    n_nodes = sim.shape[0]
    n_edges = len(edges)
    degrees = np.zeros(n_nodes, dtype=int)
    for a, b in edges:
        degrees[a] += 1
        degrees[b] += 1
    return {
        "n_nodes": int(n_nodes),
        "n_undirected_edges": int(n_edges),
        "avg_degree": float(degrees.mean()),
        "max_degree": int(degrees.max()),
    }


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    dataset, names = af.dataset.load(root=str(DATA_ROOT), task="scarce", train=True)
    print(f"Loaded {len(dataset)} training simulations.")
    print(f"First simulation shape: {dataset[0].shape}")

    plot_sample_meshes(dataset, names)
    all_targets = plot_target_distributions(dataset)

    stats = {
        "n_simulations": len(dataset),
        "nodes_per_sim_min": int(min(s.shape[0] for s in dataset)),
        "nodes_per_sim_max": int(max(s.shape[0] for s in dataset)),
        "nodes_per_sim_mean": float(np.mean([s.shape[0] for s in dataset])),
        "targets": {
            name: {
                "mean": float(all_targets[:, j].mean()),
                "std": float(all_targets[:, j].std()),
                "min": float(all_targets[:, j].min()),
                "max": float(all_targets[:, j].max()),
            }
            for j, name in enumerate(TARGET_NAMES)
        },
        "sample_graph": graph_stats(dataset[0]),
    }
    with open(OUT / "eda_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    print(f"\nPlots and stats written to {OUT}")
