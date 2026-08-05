"""Extended EDA for the FULL AirfRANS dataset (1000 Simulations).
Creates a comprehensive CSV table of aerodynamic statistics per airfoil,
distributions across all 1000 sims, and generates a massive shape grid.
"""
import os
import csv
from pathlib import Path

import airfrans as af
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT / "data" / "Dataset"
OUT = PROJECT / "results" / "eda"

TARGET_NAMES = ["velocity_x [m/s]", "velocity_y [m/s]", "p/rho [m2/s2]", "nu_t [m2/s]"]
TARGET_COLS = [7, 8, 9, 10]

def gen_extended_stats_1000(dataset, names):
    """Generate a detailed CSV for every simulation."""
    fields = [
        "Simulation_Name", "Total_Nodes", "Surface_Nodes",
        "U_target_mean", "U_target_max", "U_target_min",
        "V_target_mean", "V_target_max", "V_target_min",
        "Pressure_mean", "Pressure_max", "Pressure_min",
        "NuT_mean", "NuT_max", "NuT_min"
    ]
    
    stats_list = []
    for sim, name in zip(dataset, names):
        n_nodes = sim.shape[0]
        surf_nodes = int(np.sum(sim[:, 11] == 1))
        u, v, p, nu = sim[:, 7], sim[:, 8], sim[:, 9], sim[:, 10]
        
        stats_list.append({
            "Simulation_Name": name,
            "Total_Nodes": n_nodes,
            "Surface_Nodes": surf_nodes,
            "U_target_mean": float(np.mean(u)),
            "U_target_max": float(np.max(u)),
            "U_target_min": float(np.min(u)),
            "V_target_mean": float(np.mean(v)),
            "V_target_max": float(np.max(v)),
            "V_target_min": float(np.min(v)),
            "Pressure_mean": float(np.mean(p)),
            "Pressure_max": float(np.max(p)),
            "Pressure_min": float(np.min(p)),
            "NuT_mean": float(np.mean(nu)),
            "NuT_max": float(np.max(nu)),
            "NuT_min": float(np.min(nu)),
        })
        
    csv_file = OUT / "all_1000_detailed_stats.csv"
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(stats_list)
    print(f"[OK] Generated comprehensive table for all {len(dataset)} airfoils at: {csv_file}")


def plot_target_distributions_1000(dataset):
    """Histograms and box plots saved as 4 separate images for each target field."""
    all_targets = np.concatenate([sim[:, TARGET_COLS] for sim in dataset], axis=0)
    
    file_names = ["dist_velocity_x", "dist_velocity_y", "dist_pressure", "dist_nut"]
    
    for j, name in enumerate(TARGET_NAMES):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram
        axes[0].hist(all_targets[:, j], bins=100)
        axes[0].set_title(f"{name} - Histogram")
        axes[0].set_yscale("log")
        
        # Box plot
        axes[1].boxplot(all_targets[:, j], showfliers=False)
        axes[1].set_title(f"{name} (no outliers) - Box Plot")
        
        fig.suptitle(f"{name} Distribution — Full 1000 Simulations")
        fig.tight_layout()
        
        output_path = OUT / f"all_1000_{file_names[j]}.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        
    print(f"[OK] Generated 4 separate physical target distribution images at: {OUT}")


def plot_all_airfoils_1000(dataset, names):
    """Massive scatter plots showing the shape of all 1000 airfoils in a huge grid."""
    n = len(dataset)
    cols = 20  # Make it wide since there are 1000
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2 * rows))
    axes = axes.flatten()
    
    for i in range(n):
        sim = dataset[i]
        # Filter to surface points only
        surf = sim[sim[:, 11] == 1]
        
        ax = axes[i]
        ax.scatter(surf[:, 0], surf[:, 1], c="black", s=0.2)
        ax.set_title(names[i][:15], fontsize=6) # tiny font for massive grid
        ax.axis("off")
        ax.set_aspect("equal")
        # Standardize limits so relative thicknesses are evident
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.25, 0.25)
        
    for i in range(n, len(axes)):
        axes[i].axis("off")
        
    fig.tight_layout()
    output_path = OUT / "all_1000_airfoils_shapes.png"
    fig.savefig(output_path, dpi=200) # Slightly higher DPI because grid is massive
    plt.close(fig)
    print(f"[OK] Generated massive 1000 airfoil shapes grid at: {output_path}")


def plot_sample_meshes_separate(dataset, names, n_samples=4):
    """Plot sample meshes into distinct image files for higher clarity."""
    idx = np.linspace(0, len(dataset) - 1, n_samples, dtype=int)
    for sample_id, i in enumerate(idx):
        sim = dataset[i]
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Full domain
        ax = axes[0]
        s = ax.scatter(sim[:, 0], sim[:, 1], c=sim[:, 4], s=0.5, cmap="viridis")
        ax.set_title(f"{names[i]} - Full Domain ({sim.shape[0]} nodes)")
        ax.set_aspect("equal")
        fig.colorbar(s, ax=ax, label="Wall distance [m]", fraction=0.046, pad=0.04)
        
        # Zoom on airfoil
        ax = axes[1]
        ax.scatter(sim[:, 0], sim[:, 1], c=sim[:, 4], s=0.5, cmap="viridis")
        surf = sim[sim[:, 11] == 1]
        ax.scatter(surf[:, 0], surf[:, 1], c="red", s=1, label="Surface nodes")
        ax.set_title("Near-Airfoil Zoom")
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.4, 0.4)
        ax.set_aspect("equal")
        ax.legend(loc="upper right")
        
        fig.suptitle(f"Sample Mesh {sample_id + 1}: {names[i]}")
        fig.tight_layout()
        
        output_path = OUT / f"all_1000_sample_mesh_{sample_id + 1}.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        
    print(f"[OK] Generated {n_samples} separate sample mesh images at: {OUT}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    
    print("[INFO] Loading 'full' task dataset from AirfRANS (Train 800 + Test 200 = 1000 total)...")
    try:
        # The 'full' task utilizes the entire dataset.
        train_data, train_names = af.dataset.load(root=str(DATA_ROOT), task="full", train=True)
        test_data, test_names = af.dataset.load(root=str(DATA_ROOT), task="full", train=False)
        
        dataset_1000 = train_data + test_data
        names_1000 = train_names + test_names
        
        print(f"[INFO] Successfully loaded {len(dataset_1000)} simulations!")
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        exit(1)
        
    print("\nGenerating extended statistics table for 1000 items (this will take a moment)...")
    gen_extended_stats_1000(dataset_1000, names_1000)
    
    print("\nGenerating physical target distributions plot for 1000 items...")
    plot_target_distributions_1000(dataset_1000)
    
    print("\nGenerating massive grid of all 1000 airfoil shapes... (this will take a few minutes...)")
    plot_all_airfoils_1000(dataset_1000, names_1000)
    print("\nGenerating separate sample mesh plots...")
    plot_sample_meshes_separate(dataset_1000, names_1000, n_samples=10)
    
    print("\n[SUCCESS] EDA for the complete 1000 dataset finished!")
