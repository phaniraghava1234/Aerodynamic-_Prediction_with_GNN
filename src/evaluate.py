"""Evaluate the trained baseline on the test set.

Run:  python src/evaluate.py [--checkpoint models/baseline_best.pt]
Writes to results/evaluation/:
  metrics.json                 field-wise MSE/MAE, lift/drag errors and R^2
  lift_drag_scatter.png        predicted vs true CL and CD with R^2
  case_<name>_fields.png       true/pred/error maps (p/rho and |U|), 3 cases
  case_<name>_cp.png           surface pressure coefficient vs x/c
  case_<name>_streamlines.png  true vs predicted streamlines

All metrics are computed on de-normalized (physical) values.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import griddata

from models import BaselineGNN
from postprocess import (denorm_targets, force_coefficients, inlet_velocity,
                         pressure_coefficient, r_squared, raw_positions)

PROJECT = Path(__file__).resolve().parents[1]
FIELDS = ["velocity_x", "velocity_y", "p_rho", "nu_t"]


def load_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = BaselineGNN(hidden=ckpt["hidden"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def predict_all(model, dataset, stats, device):
    """Return list of per-graph dicts with de-normalized predictions and truth."""
    results = []
    with torch.no_grad():
        for data in dataset:
            pred_n = model(data.to(device)).cpu()
            data = data.cpu()
            results.append({
                "data": data,
                "pred": denorm_targets(pred_n, stats),
                "true": denorm_targets(data.y, stats),
                "pos": raw_positions(data.pos, stats),
                "u_inf": inlet_velocity(data, stats),
            })
    return results


def field_maps(res, out_path):
    pos = res["pos"].numpy()
    rows = {
        "p/rho [m2/s2]": (res["true"][:, 2].numpy(), res["pred"][:, 2].numpy()),
        "|U| [m/s]": (res["true"][:, :2].norm(dim=1).numpy(),
                      res["pred"][:, :2].norm(dim=1).numpy()),
    }
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    for i, (label, (t, p)) in enumerate(rows.items()):
        vmin, vmax = t.min(), t.max()
        for j, (field, title) in enumerate([(t, "true"), (p, "predicted")]):
            s = axes[i, j].scatter(pos[:, 0], pos[:, 1], c=field, s=0.5,
                                   cmap="RdBu_r", vmin=vmin, vmax=vmax)
            axes[i, j].set_title(f"{label} — {title}")
            fig.colorbar(s, ax=axes[i, j])
        s = axes[i, 2].scatter(pos[:, 0], pos[:, 1], c=np.abs(p - t), s=0.5,
                               cmap="magma")
        axes[i, 2].set_title(f"{label} — |error|")
        fig.colorbar(s, ax=axes[i, 2])
        for ax in axes[i]:
            ax.set_xlim(-0.5, 2.0)
            ax.set_ylim(-0.75, 0.75)
            ax.set_aspect("equal")
    fig.suptitle(res["data"].name)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def cp_plot(res, out_path):
    surf = res["data"].surf
    x = res["pos"][surf, 0].numpy()
    cp_true = pressure_coefficient(res["true"][surf, 2], res["u_inf"]).numpy()
    cp_pred = pressure_coefficient(res["pred"][surf, 2], res["u_inf"]).numpy()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, cp_true, s=6, label="CFD (true)")
    ax.scatter(x, cp_pred, s=6, marker="x", label="GNN (predicted)")
    ax.invert_yaxis()
    ax.set_xlabel("x/c")
    ax.set_ylabel("$C_p$")
    ax.set_title(f"Surface pressure coefficient — {res['data'].name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def streamline_plot(res, out_path):
    pos = res["pos"].numpy()
    window = (pos[:, 0] > -0.6) & (pos[:, 0] < 2.1) & (np.abs(pos[:, 1]) < 0.85)
    pts = pos[window]
    gx, gy = np.meshgrid(np.linspace(-0.5, 2.0, 250), np.linspace(-0.75, 0.75, 150))
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, key, title in [(axes[0], "true", "CFD (true)"),
                           (axes[1], "pred", "GNN (predicted)")]:
        vel = res[key][:, :2].numpy()[window]
        u = griddata(pts, vel[:, 0], (gx, gy), method="linear")
        v = griddata(pts, vel[:, 1], (gx, gy), method="linear")
        speed = np.sqrt(u**2 + v**2)
        ax.streamplot(gx, gy, u, v, color=speed, cmap="viridis", density=1.5)
        surf = res["data"].surf
        ax.fill(res["pos"][surf, 0].numpy(), res["pos"][surf, 1].numpy(), "k")
        ax.set_title(title)
        ax.set_aspect("equal")
    fig.suptitle(f"Streamlines — {res['data'].name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=PROJECT / "data" / "processed")
    ap.add_argument("--checkpoint", type=Path, default=PROJECT / "models" / "baseline_best.pt")
    ap.add_argument("--out", type=Path, default=PROJECT / "results" / "evaluation")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = torch.load(args.data_dir / "norm_stats.pt", weights_only=False)
    test_set = torch.load(args.data_dir / "test.pt", weights_only=False)
    model = load_model(args.checkpoint, device)
    results = predict_all(model, test_set, stats, device)

    # Field-wise metrics over all test nodes
    err = torch.cat([r["pred"] - r["true"] for r in results])
    metrics = {
        "field_mse": {f: float((err[:, j] ** 2).mean()) for j, f in enumerate(FIELDS)},
        "field_mae": {f: float(err[:, j].abs().mean()) for j, f in enumerate(FIELDS)},
    }

    # Lift and drag per test case
    cl_t, cl_p, cd_t, cd_p = [], [], [], []
    for r in results:
        surf = r["data"].surf
        sp = r["pos"][surf].numpy()
        u = r["u_inf"].numpy()
        cl, cd = force_coefficients(sp, r["true"][surf, 2].numpy(), u)
        cl_t.append(cl); cd_t.append(cd)
        cl, cd = force_coefficients(sp, r["pred"][surf, 2].numpy(), u)
        cl_p.append(cl); cd_p.append(cd)
    cl_t, cl_p, cd_t, cd_p = map(np.array, (cl_t, cl_p, cd_t, cd_p))
    metrics["lift"] = {
        "r2": float(r_squared(cl_t, cl_p)),
        "mean_abs_error": float(np.abs(cl_p - cl_t).mean()),
        "mean_rel_error": float(np.mean(np.abs(cl_p - cl_t) / (np.abs(cl_t) + 1e-8))),
    }
    metrics["drag_pressure_only"] = {
        "r2": float(r_squared(cd_t, cd_p)),
        "mean_abs_error": float(np.abs(cd_p - cd_t).mean()),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, t, p, name, r2 in [(axes[0], cl_t, cl_p, "$C_L$", metrics["lift"]["r2"]),
                               (axes[1], cd_t, cd_p, "$C_D$ (pressure)",
                                metrics["drag_pressure_only"]["r2"])]:
        ax.scatter(t, p, s=15)
        lims = [min(t.min(), p.min()), max(t.max(), p.max())]
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_xlabel(f"true {name}")
        ax.set_ylabel(f"predicted {name}")
        ax.set_title(f"{name}   $R^2$ = {r2:.3f}")
    fig.tight_layout()
    fig.savefig(args.out / "lift_drag_scatter.png", dpi=150)
    plt.close(fig)

    # Best / median / worst cases by pressure MSE
    p_mse = np.array([float(((r["pred"][:, 2] - r["true"][:, 2]) ** 2).mean())
                      for r in results])
    order = np.argsort(p_mse)
    for label, idx in [("best", order[0]), ("median", order[len(order) // 2]),
                       ("worst", order[-1])]:
        r = results[idx]
        name = f"{label}_{r['data'].name}"
        field_maps(r, args.out / f"case_{name}_fields.png")
        cp_plot(r, args.out / f"case_{name}_cp.png")
        streamline_plot(r, args.out / f"case_{name}_streamlines.png")
        metrics.setdefault("cases", {})[label] = {
            "name": str(r["data"].name), "pressure_mse": float(p_mse[idx]),
        }

    with open(args.out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"\nOutputs written to {args.out}")


if __name__ == "__main__":
    main()
