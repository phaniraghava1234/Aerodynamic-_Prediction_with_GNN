"""Shared post-processing: de-normalization and aerodynamic coefficients.

All predictions and stored targets are z-scored; every metric and plot must be
computed on de-normalized (physical) values.
"""
import numpy as np
import torch


def denorm_targets(y, stats):
    """(N, 4) normalized targets/predictions -> physical units."""
    return y * stats["target_std"] + stats["target_mean"]


def raw_positions(pos, stats):
    """Min-max scaled positions -> meters."""
    return pos * (stats["pos_max"] - stats["pos_min"]) + stats["pos_min"]


def inlet_velocity(data, stats):
    """Free-stream velocity vector (2,) in m/s, recovered from node features."""
    return data.x[0, 2:4] * stats["feat_std"][:2] + stats["feat_mean"][:2]


def pressure_coefficient(p_rho, u_inf):
    """Cp = (p - p_inf)/(0.5 rho U^2) with gauge p_inf = 0 and p given as p/rho."""
    return p_rho / (0.5 * float(torch.linalg.norm(u_inf)) ** 2)


def order_surface(points):
    """Order airfoil surface points counter-clockwise by angle around the centroid.

    NACA 4/5-digit airfoils are star-shaped w.r.t. their centroid, so angle
    ordering recovers the contour.
    """
    c = points.mean(axis=0)
    return np.argsort(np.arctan2(points[:, 1] - c[1], points[:, 0] - c[0]))


def force_coefficients(surf_pos, p_rho_surf, u_inf):
    """Pressure-based lift and drag coefficients from surface pressures.

    Integrates F/rho = -sum p_mid * n_hat * dl over the closed contour
    (viscous shear neglected). Chord = 1 m for AirfRANS airfoils.
    Returns (CL, CD).
    """
    order = order_surface(surf_pos)
    pts = surf_pos[order]
    p = p_rho_surf[order]
    nxt = np.roll(np.arange(len(pts)), -1)
    seg = pts[nxt] - pts                       # (M, 2) tangents, CCW
    p_mid = 0.5 * (p + p[nxt])
    # Outward normal of a CCW contour, scaled by segment length: (dy, -dx)
    n_dl = np.stack([seg[:, 1], -seg[:, 0]], axis=1)
    force = -(p_mid[:, None] * n_dl).sum(axis=0)  # F/rho, (2,)

    u = np.asarray(u_inf, dtype=float)
    alpha = np.arctan2(u[1], u[0])
    q = 0.5 * (u @ u)                          # dynamic pressure / rho, chord = 1
    drag = force[0] * np.cos(alpha) + force[1] * np.sin(alpha)
    lift = -force[0] * np.sin(alpha) + force[1] * np.cos(alpha)
    return lift / q, drag / q


def r_squared(true, pred):
    true, pred = np.asarray(true), np.asarray(pred)
    ss_res = ((true - pred) ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum()
    return 1.0 - ss_res / ss_tot
