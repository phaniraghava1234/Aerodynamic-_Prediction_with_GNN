"""Portfolio-grade metric EDA for AirfRANS.

Covers everything a reviewer would ask about:
  1.  Data cleanliness audit (NaNs, Infs, negative wall distances, duplicate positions)
  2.  Input feature distributions (what the model consumes, not just targets)
  3.  Flight parameter extraction from simulation names (inlet velocity, AoA, NACA series)
  4.  CL / CD / L/D computed from surface pressure via postprocess.force_coefficients
  5.  Train vs Test distribution overlay on the scarce split
  6.  Delaunay graph stats aggregated across ALL sims (not just sim 0)
  7.  Edge length distribution per sim (min, p5, p50, p95, max, std)
  8.  Correlation heatmap: flight parameters vs target stats vs CL/CD
  9.  Per-simulation comprehensive CSV with percentiles (p5, p50, p95, std)
  10. Cp suction peak and stagnation point per airfoil
  11. Boundary layer node count and near-wall vs far-field target variance
  12. Mesh refinement ratio (max_edge / min_edge) and node count vs mesh density
  13. Wake turbulence intensity (mean nu_t downstream of trailing edge)
"""
import csv
import json
import sys
from pathlib import Path

import airfrans as af
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay

# Import the project's own postprocess for CL/CD
sys.path.insert(0, str(Path(__file__).resolve().parent))
from postprocess import force_coefficients, order_surface

PROJECT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT / "data" / "Dataset"
OUT = PROJECT / "results" / "eda_metrics"


# ---------------------------------------------------------------------------
# 1. Parse flight parameters from simulation folder names
# ---------------------------------------------------------------------------
def parse_sim_name(name):
    """Extract inlet velocity, AoA, and NACA parameters from the folder name.

    Naming convention (from airfrans.readthedocs.io):
      airFoil2D_SST_{inlet_vel}_{AoA}_{naca_params...}
    3 trailing numbers after AoA = NACA 4-digit; 4 = NACA 5-digit.
    """
    parts = name.split("_")
    nums = parts[2:]  # everything after 'airFoil2D_SST_'
    nums = [float(x) for x in nums]

    inlet_vel = nums[0]
    aoa_deg = nums[1]
    naca_params = nums[2:]

    naca_series = 5 if len(naca_params) == 4 else 4

    # Reynolds number: Re = V * L / nu, L=1m chord, nu=1.56e-5 (dataset convention)
    reynolds = inlet_vel * 1.0 / 1.56e-5

    # Thickness in percent chord is the last param for both NACA-4 and NACA-5
    # (AirfRANS naming convention; arXiv:2212.07564, airfrans.readthedocs.io).
    thickness_pct = naca_params[-1]

    return {
        "inlet_velocity_m_s": inlet_vel,
        "aoa_deg": aoa_deg,
        "naca_series": naca_series,
        "naca_params": naca_params,
        "reynolds": reynolds,
        "thickness_pct": thickness_pct,
    }


# ---------------------------------------------------------------------------
# 2. Data cleanliness audit
# ---------------------------------------------------------------------------
def audit_cleanliness(dataset, names):
    """Check every simulation for NaNs, Infs, negative wall distances, duplicate positions."""
    report = []
    for sim, name in zip(dataset, names):
        n_nan = int(np.isnan(sim).sum())
        n_inf = int(np.isinf(sim).sum())
        wall_dist = sim[:, 4]
        n_neg_dist = int((wall_dist < 0).sum())
        pos = sim[:, :2]
        _, counts = np.unique(pos, axis=0, return_counts=True)
        n_dup = int((counts > 1).sum())

        if n_nan or n_inf or n_neg_dist or n_dup:
            report.append({
                "name": name, "n_nan": n_nan, "n_inf": n_inf,
                "n_neg_wall_dist": n_neg_dist, "n_duplicate_positions": n_dup,
            })

    return report


# ---------------------------------------------------------------------------
# 3. CL / CD from surface pressure
# ---------------------------------------------------------------------------
def compute_cl_cd(sim):
    """Compute pressure-based CL and CD for one simulation using postprocess.py."""
    surf_mask = sim[:, 11] == 1
    surf_pos = sim[surf_mask][:, :2]
    p_rho_surf = sim[surf_mask][:, 9]
    u_inf = sim[0, 2:4]
    cl, cd = force_coefficients(surf_pos, p_rho_surf, u_inf)
    return float(cl), float(cd)


# ---------------------------------------------------------------------------
# 4. Delaunay + edge-length stats for one sim
# ---------------------------------------------------------------------------
def delaunay_stats(sim):
    """Delaunay graph and edge length statistics for one simulation."""
    pos = sim[:, :2]
    tri = Delaunay(pos)
    edges = set()
    for simplex in tri.simplices:
        for a in range(3):
            for b in range(a + 1, 3):
                edges.add((min(simplex[a], simplex[b]), max(simplex[a], simplex[b])))

    n_nodes = pos.shape[0]
    n_edges = len(edges)
    degrees = np.zeros(n_nodes, dtype=int)
    lengths = []
    for a, b in edges:
        degrees[a] += 1
        degrees[b] += 1
        lengths.append(np.linalg.norm(pos[a] - pos[b]))

    lengths = np.array(lengths)
    return {
        "n_nodes": int(n_nodes),
        "n_edges": int(n_edges),
        "avg_degree": float(degrees.mean()),
        "max_degree": int(degrees.max()),
        "degree_std": float(degrees.std()),
        "edge_len_min": float(lengths.min()),
        "edge_len_p5": float(np.percentile(lengths, 5)),
        "edge_len_p50": float(np.percentile(lengths, 50)),
        "edge_len_p95": float(np.percentile(lengths, 95)),
        "edge_len_max": float(lengths.max()),
        "edge_len_std": float(lengths.std()),
        "mesh_refinement_ratio": float(lengths.max() / lengths.min()) if lengths.min() > 0 else float("inf"),
    }


# ---------------------------------------------------------------------------
# 5. Cp analysis + stagnation point
# ---------------------------------------------------------------------------
def cp_and_stagnation(sim):
    """Compute Cp distribution metrics and stagnation point location."""
    surf_mask = sim[:, 11] == 1
    surf_pos = sim[surf_mask][:, :2]
    p_rho_surf = sim[surf_mask][:, 9]
    u_inf = sim[0, 2:4]
    q_inf = 0.5 * np.linalg.norm(u_inf) ** 2  # dynamic pressure / rho

    cp = p_rho_surf / q_inf  # Cp = (p - p_inf) / q, with gauge p_inf=0

    # Stagnation point: where Cp is maximum (highest pressure)
    stag_idx = np.argmax(cp)
    stag_x = float(surf_pos[stag_idx, 0])
    stag_y = float(surf_pos[stag_idx, 1])

    return {
        "Cp_min": float(cp.min()),  # suction peak (most negative = strongest suction)
        "Cp_max": float(cp.max()),  # stagnation Cp (should be ~1.0)
        "Cp_mean": float(cp.mean()),
        "stagnation_x": stag_x,
        "stagnation_y": stag_y,
    }


# ---------------------------------------------------------------------------
# 6. Boundary layer and near-wall analysis
# ---------------------------------------------------------------------------
def boundary_layer_analysis(sim, bl_threshold=0.05):
    """Analyse near-wall vs far-field regions.

    bl_threshold: wall distance [m] below which a node is considered 'near-wall'.
    """
    wall_dist = sim[:, 4]
    near_wall = wall_dist < bl_threshold
    far_field = ~near_wall

    n_bl_nodes = int(near_wall.sum())
    bl_fraction = float(n_bl_nodes / sim.shape[0])

    # Target variance split: near-wall vs far-field
    targets = sim[:, 7:11]
    tgt_names = ["U", "V", "P", "NuT"]
    result = {
        "n_boundary_layer_nodes": n_bl_nodes,
        "boundary_layer_fraction": bl_fraction,
    }
    for j, tn in enumerate(tgt_names):
        if near_wall.sum() > 0:
            result[f"{tn}_nearwall_std"] = float(targets[near_wall, j].std())
        else:
            result[f"{tn}_nearwall_std"] = 0.0
        if far_field.sum() > 0:
            result[f"{tn}_farfield_std"] = float(targets[far_field, j].std())
        else:
            result[f"{tn}_farfield_std"] = 0.0

    return result


# ---------------------------------------------------------------------------
# 7. Wake turbulence
# ---------------------------------------------------------------------------
def wake_turbulence(sim, x_wake_start=1.0):
    """Mean nu_t in the wake region downstream of the trailing edge."""
    wake_mask = sim[:, 0] > x_wake_start
    nut_wake = sim[wake_mask, 10]
    return {
        "wake_nut_mean": float(nut_wake.mean()) if len(nut_wake) > 0 else 0.0,
        "wake_nut_max": float(nut_wake.max()) if len(nut_wake) > 0 else 0.0,
        "n_wake_nodes": int(wake_mask.sum()),
    }


# ---------------------------------------------------------------------------
# 8. Mesh density (nodes per unit area)
# ---------------------------------------------------------------------------
def mesh_density(sim):
    """Compute mesh density as nodes per unit area of the bounding box."""
    pos = sim[:, :2]
    x_range = pos[:, 0].max() - pos[:, 0].min()
    y_range = pos[:, 1].max() - pos[:, 1].min()
    area = x_range * y_range
    return {
        "domain_area": float(area),
        "mesh_density_nodes_per_m2": float(sim.shape[0] / area) if area > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# 9. Build the master table
# ---------------------------------------------------------------------------
def field_percentiles(arr, prefix):
    """Return p5, p50, p95, mean, std, min, max for a 1D array."""
    return {
        f"{prefix}_min": float(arr.min()),
        f"{prefix}_p5": float(np.percentile(arr, 5)),
        f"{prefix}_p50": float(np.percentile(arr, 50)),
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_p95": float(np.percentile(arr, 95)),
        f"{prefix}_max": float(arr.max()),
        f"{prefix}_std": float(arr.std()),
    }


def build_master_table(dataset, names):
    """Build one row per simulation with all metrics."""
    rows = []
    for i, (sim, name) in enumerate(zip(dataset, names)):
        print(f"  [{i+1}/{len(dataset)}] Processing {name}...")
        fp = parse_sim_name(name)
        cl, cd = compute_cl_cd(sim)
        ld = cl / cd if abs(cd) > 1e-12 else float("nan")

        # Delaunay stats
        gs = delaunay_stats(sim)

        # Cp + stagnation
        cp_stats = cp_and_stagnation(sim)

        # Boundary layer
        bl = boundary_layer_analysis(sim)

        # Wake turbulence
        wt = wake_turbulence(sim)

        # Mesh density
        md = mesh_density(sim)

        # Target percentiles (p5, p50, p95, mean, std, min, max)
        u_pct = field_percentiles(sim[:, 7], "U")
        v_pct = field_percentiles(sim[:, 8], "V")
        p_pct = field_percentiles(sim[:, 9], "P")
        nut_pct = field_percentiles(sim[:, 10], "NuT")

        # Input feature percentiles
        wd_pct = field_percentiles(sim[:, 4], "wall_dist")

        row = {
            "name": name,
            # Flight parameters
            "inlet_velocity_m_s": fp["inlet_velocity_m_s"],
            "aoa_deg": fp["aoa_deg"],
            "reynolds": fp["reynolds"],
            "naca_series": fp["naca_series"],
            "thickness_pct": fp["thickness_pct"],
            "inlet_u": float(sim[0, 2]),
            "inlet_v": float(sim[0, 3]),
            # Graph topology
            "n_nodes": gs["n_nodes"],
            "n_surface_nodes": int(np.sum(sim[:, 11] == 1)),
            "surface_node_fraction": float(np.sum(sim[:, 11] == 1) / sim.shape[0]),
            "n_edges": gs["n_edges"],
            "avg_degree": gs["avg_degree"],
            "max_degree": gs["max_degree"],
            "degree_std": gs["degree_std"],
            # Edge lengths
            "edge_len_min": gs["edge_len_min"],
            "edge_len_p5": gs["edge_len_p5"],
            "edge_len_p50": gs["edge_len_p50"],
            "edge_len_p95": gs["edge_len_p95"],
            "edge_len_max": gs["edge_len_max"],
            "edge_len_std": gs["edge_len_std"],
            "mesh_refinement_ratio": gs["mesh_refinement_ratio"],
            # Mesh density
            **md,
            # Aerodynamic performance
            "CL": cl,
            "CD": cd,
            "L_over_D": ld,
            # Cp analysis
            **cp_stats,
            # Boundary layer
            **bl,
            # Wake turbulence
            **wt,
            # Target percentiles
            **u_pct, **v_pct, **p_pct, **nut_pct,
            # Wall distance percentiles
            **wd_pct,
        }
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# 10. CD diagnostics — surface any sim whose pressure-integrated CD is negative
#     or inside the noise floor. Negative CD is a strong indicator of a
#     surface-ordering artefact in postprocess.order_surface (angle-around-
#     centroid ordering fails for non-star-shaped airfoils). Small positive
#     CD (< 0.005) is in the pressure-integration noise band because viscous
#     shear is excluded here by design (see README).
# ---------------------------------------------------------------------------
def report_cd_diagnostics(rows, out_dir):
    """Report sims with negative or near-zero CD; save to cd_diagnostics.json."""
    neg = [r for r in rows if r["CD"] < 0]
    tiny = [r for r in rows if 0 <= r["CD"] < 0.005]
    report = {
        "total_sims": len(rows),
        "negative_cd_count": len(neg),
        "small_positive_cd_count_lt_0p005": len(tiny),
        "negative_cd_sims": [
            {"name": r["name"], "CD": r["CD"], "CL": r["CL"], "aoa_deg": r["aoa_deg"]}
            for r in neg
        ],
        "note": (
            "Negative CD points to a surface-ordering issue in "
            "postprocess.order_surface (angle-around-centroid fails for non-"
            "star-shaped airfoils). Small positive CD (< 0.005) is inside the "
            "pressure-only integration noise floor (viscous shear excluded)."
        ),
    }
    with open(out_dir / "cd_diagnostics.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  CD diagnostics: {len(neg)}/{len(rows)} negative, "
          f"{len(tiny)}/{len(rows)} tiny (< 0.005). See cd_diagnostics.json.")


# ---------------------------------------------------------------------------
# 11. Plotting helpers — each metric gets its own standalone image
# ---------------------------------------------------------------------------

def _save(fig, name):
    """Save a single figure and close it."""
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=150)
    plt.close(fig)


def plot_input_distributions(dataset, label, suffix):
    """Distribution of input features — one image per feature."""
    all_inlet = np.concatenate([sim[:, 2:4] for sim in dataset])
    all_dist = np.concatenate([sim[:, 4:5] for sim in dataset])
    all_normals = np.concatenate([sim[:, 5:7] for sim in dataset])
    all_surf = np.concatenate([sim[:, 11:12] for sim in dataset])

    items = [
        (all_inlet[:, 0], f"Inlet Velocity X [m/s] ({label})", f"input_inlet_vel_x_{suffix}"),
        (all_inlet[:, 1], f"Inlet Velocity Y [m/s] ({label})", f"input_inlet_vel_y_{suffix}"),
        (all_dist.ravel(), f"Wall Distance [m] ({label})", f"input_wall_distance_{suffix}"),
        (all_normals[:, 0], f"Surface Normal X ({label})", f"input_normal_x_{suffix}"),
        (all_normals[:, 1], f"Surface Normal Y ({label})", f"input_normal_y_{suffix}"),
    ]
    for data, title, fname in items:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(data, bins=100)
        ax.set_title(title)
        ax.set_yscale("log")
        _save(fig, fname)

    # Surface flag bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(["Off-surface", "On-surface"],
           [int((all_surf == 0).sum()), int((all_surf == 1).sum())])
    ax.set_title(f"Surface Flag Distribution ({label})")
    ax.set_yscale("log")
    _save(fig, f"input_surface_flag_{suffix}")


def plot_train_vs_test_overlay(train_data, test_data):
    """Overlay histograms of train vs test — one image per field."""
    train_targets = np.concatenate([s[:, 7:11] for s in train_data])
    test_targets = np.concatenate([s[:, 7:11] for s in test_data])
    target_names = ["Velocity_X", "Velocity_Y", "Pressure", "NuT"]
    target_labels = ["Velocity X", "Velocity Y", "Pressure (p/rho)", "Nu_t"]

    for j, (tname, tlabel) in enumerate(zip(target_names, target_labels)):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(train_targets[:, j], bins=100, alpha=0.6, label="Train (180)", density=True)
        ax.hist(test_targets[:, j], bins=100, alpha=0.6, label="Test (200)", density=True)
        ax.set_title(f"Train vs Test: {tlabel}")
        ax.set_yscale("log")
        ax.legend()
        _save(fig, f"train_vs_test_target_{tname}")

    # Input features overlay
    train_inputs = np.concatenate([s[:, 2:5] for s in train_data])
    test_inputs = np.concatenate([s[:, 2:5] for s in test_data])
    input_names = ["Inlet_Vel_X", "Inlet_Vel_Y", "Wall_Distance"]
    input_labels = ["Inlet Vel X", "Inlet Vel Y", "Wall Distance"]

    for j, (iname, ilabel) in enumerate(zip(input_names, input_labels)):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(train_inputs[:, j], bins=100, alpha=0.6, label="Train (180)", density=True)
        ax.hist(test_inputs[:, j], bins=100, alpha=0.6, label="Test (200)", density=True)
        ax.set_title(f"Train vs Test: {ilabel}")
        ax.set_yscale("log")
        ax.legend()
        _save(fig, f"train_vs_test_input_{iname}")


def plot_flight_params(rows):
    """Flight condition distributions — one image each."""
    items = [
        ([r["inlet_velocity_m_s"] for r in rows], "Inlet Velocity [m/s]", "flight_inlet_velocity"),
        ([r["aoa_deg"] for r in rows], "Angle of Attack [deg]", "flight_aoa"),
        ([r["reynolds"] for r in rows], "Reynolds Number", "flight_reynolds"),
    ]
    for data, title, fname in items:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(data, bins=30, edgecolor="black")
        ax.set_title(title)
        _save(fig, fname)


def plot_cl_cd(rows):
    """CL, CD, L/D — one image each."""
    cl = [r["CL"] for r in rows]
    cd = [r["CD"] for r in rows]
    ld = [r["L_over_D"] for r in rows]
    aoa = [r["aoa_deg"] for r in rows]
    vel = [r["inlet_velocity_m_s"] for r in rows]

    # Drag Polar
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(cd, cl, c=vel, cmap="viridis", s=15)
    ax.set_xlabel("CD (pressure only)")
    ax.set_ylabel("CL")
    ax.set_title("Drag Polar (colored by inlet vel)")
    fig.colorbar(sc, ax=ax, label="Inlet Vel [m/s]")
    _save(fig, "aero_drag_polar")

    # CL vs AoA
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(aoa, cl, s=15, c=vel, cmap="viridis")
    ax.set_xlabel("Angle of Attack [deg]")
    ax.set_ylabel("CL")
    ax.set_title("CL vs AoA")
    fig.colorbar(sc, ax=ax, label="Inlet Vel [m/s]")
    _save(fig, "aero_cl_vs_aoa")

    # CD vs AoA
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(aoa, cd, s=15, c=vel, cmap="viridis")
    ax.set_xlabel("Angle of Attack [deg]")
    ax.set_ylabel("CD (pressure)")
    ax.set_title("CD vs AoA")
    fig.colorbar(sc, ax=ax, label="Inlet Vel [m/s]")
    _save(fig, "aero_cd_vs_aoa")

    # L/D vs AoA
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(aoa, ld, s=15, c=vel, cmap="viridis")
    ax.set_xlabel("Angle of Attack [deg]")
    ax.set_ylabel("L/D")
    ax.set_title("Lift-to-Drag Ratio vs AoA")
    fig.colorbar(sc, ax=ax, label="Inlet Vel [m/s]")
    _save(fig, "aero_ld_vs_aoa")


def plot_delaunay_aggregated(rows):
    """Aggregated Delaunay stats — one image each."""
    items = [
        ([r["n_nodes"] for r in rows], "Nodes per Sim", "graph_nodes_per_sim"),
        ([r["n_edges"] for r in rows], "Undirected Edges per Sim", "graph_edges_per_sim"),
        ([r["avg_degree"] for r in rows], "Average Degree per Sim", "graph_avg_degree"),
        ([r["degree_std"] for r in rows], "Degree Std per Sim", "graph_degree_std"),
        ([r["edge_len_min"] for r in rows], "Min Edge Length per Sim", "graph_edge_len_min"),
        ([r["edge_len_p50"] for r in rows], "Median (p50) Edge Length per Sim", "graph_edge_len_p50"),
        ([r["edge_len_p95"] for r in rows], "p95 Edge Length per Sim", "graph_edge_len_p95"),
        ([r["mesh_refinement_ratio"] for r in rows], "Mesh Refinement Ratio (max/min)", "graph_mesh_refinement_ratio"),
    ]
    for data, title, fname in items:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(data, bins=30, edgecolor="black")
        ax.set_title(title)
        _save(fig, fname)


def plot_mesh_density_vs_nodes(rows):
    """Mesh density analysis — one image each."""
    n_nodes = [r["n_nodes"] for r in rows]
    density = [r["mesh_density_nodes_per_m2"] for r in rows]
    refine = [r["mesh_refinement_ratio"] for r in rows]
    vel = [r["inlet_velocity_m_s"] for r in rows]

    # Node count vs mesh density
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(n_nodes, density, c=vel, cmap="viridis", s=15)
    ax.set_xlabel("Node Count")
    ax.set_ylabel("Mesh Density [nodes/m²]")
    ax.set_title("Node Count vs Mesh Density")
    fig.colorbar(sc, ax=ax, label="Inlet Vel [m/s]")
    _save(fig, "mesh_nodecount_vs_density")

    # Node count vs refinement ratio
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(n_nodes, refine, s=15, c=vel, cmap="viridis")
    ax.set_xlabel("Node Count")
    ax.set_ylabel("Refinement Ratio (max/min edge)")
    ax.set_title("Node Count vs Mesh Refinement Ratio")
    _save(fig, "mesh_nodecount_vs_refinement")

    # Mesh density distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(density, bins=30, edgecolor="black")
    ax.set_title("Mesh Density Distribution")
    ax.set_xlabel("Nodes / m²")
    _save(fig, "mesh_density_distribution")


def plot_boundary_layer_analysis(rows):
    """BL analysis — one image each."""
    # BL node fraction
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist([r["boundary_layer_fraction"] for r in rows], bins=30, edgecolor="black")
    ax.set_title("Boundary Layer Node Fraction (d < 0.05m)")
    _save(fig, "bl_node_fraction")

    # Near-wall vs far-field variance for each target
    for tn in ["U", "V", "P", "NuT"]:
        nw = [r[f"{tn}_nearwall_std"] for r in rows]
        ff = [r[f"{tn}_farfield_std"] for r in rows]
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(ff, nw, s=15, alpha=0.7)
        ax.set_xlabel(f"{tn} Far-field Std")
        ax.set_ylabel(f"{tn} Near-wall Std")
        ax.set_title(f"{tn}: Near-wall vs Far-field Variance")
        lim = max(max(nw), max(ff))
        ax.plot([0, lim], [0, lim], "r--", alpha=0.4, label="Equal")
        ax.legend()
        _save(fig, f"bl_nearwall_vs_farfield_{tn}")

    # Wake turbulence
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist([r["wake_nut_mean"] for r in rows], bins=30, edgecolor="black")
    ax.set_title("Wake Turbulence (mean nu_t, x > 1m)")
    _save(fig, "bl_wake_turbulence")


def plot_cp_analysis(rows):
    """Cp analysis — one image each."""
    aoa = [r["aoa_deg"] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(aoa, [r["Cp_min"] for r in rows], s=15, alpha=0.7)
    ax.set_xlabel("AoA [deg]")
    ax.set_ylabel("Cp_min (suction peak)")
    ax.set_title("Suction Peak vs AoA")
    _save(fig, "cp_suction_peak_vs_aoa")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(aoa, [r["Cp_max"] for r in rows], s=15, alpha=0.7)
    ax.set_xlabel("AoA [deg]")
    ax.set_ylabel("Cp_max (stagnation)")
    ax.set_title("Stagnation Cp vs AoA")
    _save(fig, "cp_stagnation_vs_aoa")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(aoa, [r["stagnation_x"] for r in rows], s=15, alpha=0.7)
    ax.set_xlabel("AoA [deg]")
    ax.set_ylabel("Stagnation Point X [m]")
    ax.set_title("Stagnation Point Location vs AoA")
    _save(fig, "cp_stagnation_point_location")


def plot_correlation_heatmap(rows):
    """Correlation heatmap — this one stays as a single large image (it needs to be).

    Reynolds and mesh_density_nodes_per_m2 are omitted: Reynolds is a constant
    multiple of inlet_velocity_m_s, and mesh density is a constant multiple of
    n_nodes (bounding box is fixed across AirfRANS sims). Keeping both members of
    each collinear pair would double-print every off-diagonal correlation.
    """
    keys = [
        "inlet_velocity_m_s", "aoa_deg", "thickness_pct",
        "CL", "CD", "L_over_D",
        "Cp_min", "Cp_max",
        "U_mean", "U_std", "V_mean", "V_std",
        "P_mean", "P_std", "NuT_mean", "NuT_std",
        "n_nodes", "avg_degree", "edge_len_p50", "mesh_refinement_ratio",
        "boundary_layer_fraction", "wake_nut_mean",
    ]
    labels = [
        "Inlet Vel", "AoA", "Thickness %",
        "CL", "CD", "L/D",
        "Cp min", "Cp max",
        "U mean", "U std", "V mean", "V std",
        "P mean", "P std", "NuT mean", "NuT std",
        "Nodes", "Avg Deg", "Edge p50", "Refine Ratio",
        "BL Fraction", "Wake NuT",
    ]

    data = np.array([[r[k] for k in keys] for r in rows])
    # Drop any sim with non-finite values in the selected columns rather than
    # substituting zero (which np.nan_to_num would do). Zero is a valid value in
    # every column here, so substitution would poison the correlation silently.
    row_finite = np.all(np.isfinite(data), axis=1)
    n_dropped = int((~row_finite).sum())
    if n_dropped:
        print(f"  [WARN] correlation heatmap: dropped {n_dropped}/{len(rows)} "
              f"sims with non-finite values (usually CD ~ 0 -> L/D inf).")
    data = data[row_finite]
    corr = np.corrcoef(data.T)

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(corr[i, j]) > 0.6 else "black")

    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Correlation Heatmap: Flight Params × Targets × Aero × Mesh × BL", fontsize=13)
    _save(fig, "correlation_heatmap")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)

    # Load scarce split (train + test)
    print("[1/10] Loading scarce dataset (train + test)...")
    train_raw, train_names = af.dataset.load(root=str(DATA_ROOT), task="scarce", train=True)
    test_raw, test_names = af.dataset.load(root=str(DATA_ROOT), task="scarce", train=False)

    # Val holdout (same as data_processing.py)
    val_raw, val_names = train_raw[-20:], train_names[-20:]
    train_only_raw, train_only_names = train_raw[:-20], train_names[:-20]

    # ---- Data cleanliness audit ----
    print("[2/10] Running data cleanliness audit...")
    issues = audit_cleanliness(train_raw + test_raw, train_names + test_names)
    audit_path = OUT / "data_cleanliness_audit.json"
    with open(audit_path, "w") as f:
        json.dump({
            "total_simulations_checked": len(train_raw) + len(test_raw),
            "simulations_with_issues": len(issues),
            "issues": issues if issues else "NONE - all simulations clean",
        }, f, indent=2)
    print(f"  Cleanliness audit saved to {audit_path}")
    if issues:
        print(f"  WARNING: {len(issues)} simulations have data issues!")
    else:
        print("  All simulations clean (no NaNs, Infs, negative distances, or duplicates)")

    # ---- Input feature distributions ----
    print("[3/10] Plotting input feature distributions...")
    plot_input_distributions(train_only_raw, "Train (180 sims)", "train")
    plot_input_distributions(test_raw, "Test (200 sims)", "test")

    # ---- Train vs Test overlay ----
    print("[4/10] Generating train vs test distribution overlays...")
    plot_train_vs_test_overlay(train_only_raw, test_raw)

    # ---- Build master table ----
    print("[5/10] Building master table (ALL metrics per simulation)...")
    print("  This runs Delaunay + force integration + Cp + BL analysis for every sim.")
    print("  Uses train_only_raw (180 sims): the 20 val sims are held out so master-")
    print("  table statistics don't leak into normalisation/hyperparameter choices.")
    print("  Expect ~10-20 minutes...")
    rows = build_master_table(train_only_raw, train_only_names)

    # Save master CSV
    csv_path = OUT / "master_metrics_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Master table saved to {csv_path}")

    # ---- CD diagnostics (surface any negative/noise-floor CD before we plot them) ----
    print("  Running CD diagnostics (flags pressure-only integration failures)...")
    report_cd_diagnostics(rows, OUT)

    # ---- Flight parameter plots ----
    print("[6/10] Plotting flight condition distributions...")
    plot_flight_params(rows)

    # ---- CL / CD / L/D analysis ----
    print("[7/10] Plotting CL/CD/L/D analysis...")
    plot_cl_cd(rows)

    # ---- Cp analysis ----
    print("[8/10] Plotting Cp and stagnation point analysis...")
    plot_cp_analysis(rows)

    # ---- Delaunay aggregated stats ----
    print("[9/10] Plotting aggregated Delaunay & mesh density stats...")
    plot_delaunay_aggregated(rows)
    plot_mesh_density_vs_nodes(rows)

    # ---- Boundary layer + wake ----
    print("  Plotting boundary layer & wake analysis...")
    plot_boundary_layer_analysis(rows)

    # ---- Correlation heatmap ----
    print("[10/10] Generating correlation heatmap...")
    plot_correlation_heatmap(rows)

    print(f"\n[SUCCESS] Portfolio-grade metric EDA complete! All outputs in: {OUT}")
    print(f"  Master CSV with {len(rows[0])} columns per simulation: {csv_path}")
