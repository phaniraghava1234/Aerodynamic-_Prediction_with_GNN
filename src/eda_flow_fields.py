"""Aerodynamic Flow Field EDA for AirfRANS dataset.
Visualizes the actual physical target variables (Pressure, Velocity magnitude,
and Turbulence Viscosity) mapped directly onto the mesh to intuitively understand
the fluid physics.
"""
import os
from pathlib import Path

import airfrans as af
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT / "data" / "Dataset"
OUT = PROJECT / "results" / "flow_fields"


def plot_flow_fields(dataset, names, n_samples=20):
    idx = np.linspace(0, len(dataset) - 1, n_samples, dtype=int)
    for sample_id, i in enumerate(idx):
        sim = dataset[i]
        
        # Calculate velocity magnitude from U_target(7) and V_target(8)
        u, v = sim[:, 7], sim[:, 8]
        vel_mag = np.sqrt(u**2 + v**2)
        
        p = sim[:, 9] # Kinematic Pressure
        nut = sim[:, 10] # Turbulent viscosity
        
        sim_name = names[i]
        tag = sample_id + 1
        
        # 1. Velocity Magnitude
        fig, ax = plt.subplots(figsize=(10, 6))
        sc1 = ax.scatter(sim[:, 0], sim[:, 1], c=vel_mag, s=1.5, cmap="jet")
        ax.set_title(f"Velocity Magnitude: {sim_name}\n(Wake visible behind airfoil)")
        ax.set_aspect("equal")
        ax.set_xlim(-0.5, 2.0)
        ax.set_ylim(-0.8, 0.8)
        fig.colorbar(sc1, ax=ax, label="[m/s]")
        fig.tight_layout()
        fig.savefig(OUT / f"sample_{tag}_01_velocity.png", dpi=150)
        plt.close(fig)
        
        # 2. Kinematic Pressure
        fig, ax = plt.subplots(figsize=(10, 6))
        sc2 = ax.scatter(sim[:, 0], sim[:, 1], c=p, s=1.5, cmap="jet")
        ax.set_title(f"Kinematic Pressure: {sim_name}\n(Stagnation point at leading edge)")
        ax.set_aspect("equal")
        ax.set_xlim(-0.5, 2.0)
        ax.set_ylim(-0.8, 0.8)
        fig.colorbar(sc2, ax=ax, label="[m²/s²]")
        fig.tight_layout()
        fig.savefig(OUT / f"sample_{tag}_02_pressure.png", dpi=150)
        plt.close(fig)
        
        # 3. Turbulent Viscosity
        fig, ax = plt.subplots(figsize=(10, 6))
        sc3 = ax.scatter(sim[:, 0], sim[:, 1], c=nut, s=1.5, cmap="jet")
        ax.set_title(f"Turbulent Viscosity (nu_t): {sim_name}")
        ax.set_aspect("equal")
        ax.set_xlim(-0.5, 2.0)
        ax.set_ylim(-0.8, 0.8)
        fig.colorbar(sc3, ax=ax, label="[m²/s]")
        fig.tight_layout()
        fig.savefig(OUT / f"sample_{tag}_03_turbulence.png", dpi=150)
        plt.close(fig)
        
        print(f"[OK] Saved 3 separate physics contours for {sim_name}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading training dataset for physics flow visualization...")
    
    # We can use scarce training dataset as a highly representative set
    dataset, names = af.dataset.load(root=str(DATA_ROOT), task="scarce", train=True)
    
    print("Generating colored flow fields (Velocity, Pressure, Turbulence) for 20 sample airfoils...")
    plot_flow_fields(dataset, names, n_samples=20)
    
    print("\n[SUCCESS] Flow field visualizations complete!")
