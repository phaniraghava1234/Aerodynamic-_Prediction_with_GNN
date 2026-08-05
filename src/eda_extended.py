"""Extended EDA for AirfRANS dataset.
Creates a comprehensive CSV table of aerodynamic statistics per airfoil,
and generates a massive 20x10 shape grid.
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

def gen_extended_stats(dataset, names):
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
        
    csv_file = OUT / "airfoil_detailed_stats.csv"
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(stats_list)
    print(f"Generated comprehensive table for {len(dataset)} airfoils at: {csv_file}")


def plot_all_airfoils(dataset, names):
    """Scatter plots showing the shape of every single airfoil in the dataset."""
    n = len(dataset)
    cols = 10
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2 * rows))
    axes = axes.flatten()
    
    for i in range(n):
        sim = dataset[i]
        # Filter to surface points only
        surf = sim[sim[:, 11] == 1]
        
        ax = axes[i]
        ax.scatter(surf[:, 0], surf[:, 1], c="black", s=1)
        ax.set_title(names[i][:20], fontsize=8) 
        ax.axis("off")
        ax.set_aspect("equal")
        # Standardize limits so relative thicknesses are evident
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.25, 0.25)
        
    for i in range(n, len(axes)):
        axes[i].axis("off")
        
    fig.tight_layout()
    output_path = OUT / "all_airfoils_shapes.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Generated all airfoil shapes grid at: {output_path}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    
    if DATA_ROOT.exists():
        total_items = len([x for x in DATA_ROOT.iterdir() if x.is_dir()])
        print(f"\n[INFO] Total items in raw 14GB 'Dataset' folder: {total_items}")
    else:
        print("\n[INFO] Dataset root not found.")
        
    print("[INFO] Loading 'scarce' dataset...")
    dataset, names = af.dataset.load(root=str(DATA_ROOT), task="scarce", train=True)
    print(f"[INFO] Analyzed {len(dataset)} simulations for the 'scarce' Machine Learning task.\n")
    
    gen_extended_stats(dataset, names)
    print("\nGenerating all 200 airfoil shapes plot... (this might take a few seconds)")
    plot_all_airfoils(dataset, names)
