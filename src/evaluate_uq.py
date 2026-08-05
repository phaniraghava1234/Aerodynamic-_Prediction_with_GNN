"""Monte Carlo Dropout uncertainty quantification on the test set.

Run:  python src/evaluate_uq.py [--passes 50]
Writes to results/uq/:
  uq_metrics.json                  mean predictive std per field,
                                   uncertainty-error correlation
  uncertainty_vs_error.png         does high uncertainty flag high error?
  case_<name>_uncertainty_map.png  spatial map of predictive std (p/rho)
  case_<name>_cp_band.png          Cp with a +/- 2 sigma uncertainty band

Method: the trained model is set to eval(), then only its Dropout modules are
switched back to train mode. T stochastic forward passes yield an ensemble;
its mean is the prediction, its std the model uncertainty. Normalization is
linear, so de-normalized mean = mean * sigma + mu and std = std * sigma.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluate import FIELDS, load_model
from postprocess import denorm_targets, inlet_velocity, pressure_coefficient, raw_positions

PROJECT = Path(__file__).resolve().parents[1]


def enable_mc_dropout(model):
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def mc_predict(model, data, stats, device, passes):
    """T stochastic passes -> de-normalized predictive mean and std, (N, 4) each."""
    preds = []
    with torch.no_grad():
        data = data.to(device)
        for _ in range(passes):
            preds.append(model(data).cpu())
    preds = torch.stack(preds)                       # (T, N, 4) normalized
    mean = denorm_targets(preds.mean(dim=0), stats)
    std = preds.std(dim=0) * stats["target_std"]     # linear scaling
    return mean, std


def uncertainty_map(pos, std_p, surf, name, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    s = ax.scatter(pos[:, 0], pos[:, 1], c=std_p, s=0.5, cmap="magma")
    ax.fill(pos[surf, 0], pos[surf, 1], "w")
    fig.colorbar(s, ax=ax, label="predictive std of p/rho [m2/s2]")
    ax.set_xlim(-0.5, 2.0)
    ax.set_ylim(-0.75, 0.75)
    ax.set_aspect("equal")
    ax.set_title(f"Model uncertainty — {name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def cp_band_plot(pos, mean_p, std_p, true_p, surf, u_inf, name, out_path):
    x = pos[surf, 0]
    order = np.argsort(x)
    q = 0.5 * float(np.linalg.norm(u_inf)) ** 2
    cp_mean = (mean_p[surf] / q)[order]
    cp_band = (2 * std_p[surf] / q)[order]
    cp_true = (true_p[surf] / q)[order]
    xs = x[order]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.fill_between(xs, cp_mean - cp_band, cp_mean + cp_band,
                    alpha=0.3, label=r"$\pm 2\sigma$")
    ax.scatter(xs, cp_mean, s=6, label="GNN predictive mean")
    ax.scatter(xs, cp_true, s=6, marker="x", label="CFD (true)")
    ax.invert_yaxis()
    ax.set_xlabel("x/c")
    ax.set_ylabel("$C_p$")
    ax.set_title(f"$C_p$ with MC Dropout uncertainty — {name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=PROJECT / "data" / "processed")
    ap.add_argument("--checkpoint", type=Path, default=PROJECT / "models" / "baseline_best.pt")
    ap.add_argument("--passes", type=int, default=50)
    ap.add_argument("--out", type=Path, default=PROJECT / "results" / "uq")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = torch.load(args.data_dir / "norm_stats.pt", weights_only=False)
    test_set = torch.load(args.data_dir / "test.pt", weights_only=False)
    model = load_model(args.checkpoint, device)
    enable_mc_dropout(model)

    all_std, all_abs_err, per_case = [], [], []
    for data in test_set:
        mean, std = mc_predict(model, data, stats, device, args.passes)
        true = denorm_targets(data.y.cpu(), stats)
        all_std.append(std)
        all_abs_err.append((mean - true).abs())
        per_case.append({"data": data.cpu(), "mean": mean, "std": std, "true": true})

    all_std = torch.cat(all_std)
    all_abs_err = torch.cat(all_abs_err)

    # Does uncertainty flag error? Pearson correlation per field over all nodes.
    corr = {}
    for j, f in enumerate(FIELDS):
        corr[f] = float(np.corrcoef(all_std[:, j].numpy(), all_abs_err[:, j].numpy())[0, 1])
    metrics = {
        "passes": args.passes,
        "mean_predictive_std": {f: float(all_std[:, j].mean()) for j, f in enumerate(FIELDS)},
        "uncertainty_error_pearson_r": corr,
    }

    # Uncertainty vs error scatter for pressure (subsampled for plot size)
    idx = np.random.default_rng(0).choice(len(all_std), min(50000, len(all_std)), replace=False)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(all_std[idx, 2], all_abs_err[idx, 2], s=1, alpha=0.2)
    ax.set_xlabel("predictive std of p/rho [m2/s2]")
    ax.set_ylabel("|error| of p/rho [m2/s2]")
    ax.set_title(f"Uncertainty vs. error (pressure), r = {corr['p_rho']:.3f}")
    fig.tight_layout()
    fig.savefig(args.out / "uncertainty_vs_error.png", dpi=150)
    plt.close(fig)

    # Most and least uncertain cases: uncertainty map + Cp band
    case_unc = np.array([float(c["std"][:, 2].mean()) for c in per_case])
    order = np.argsort(case_unc)
    for label, i in [("least_uncertain", order[0]), ("most_uncertain", order[-1])]:
        c = per_case[i]
        pos = raw_positions(c["data"].pos, stats).numpy()
        surf = c["data"].surf.numpy()
        u_inf = inlet_velocity(c["data"], stats).numpy()
        name = f"{label}_{c['data'].name}"
        uncertainty_map(pos, c["std"][:, 2].numpy(), surf, c["data"].name,
                        args.out / f"case_{name}_uncertainty_map.png")
        cp_band_plot(pos, c["mean"][:, 2].numpy(), c["std"][:, 2].numpy(),
                     c["true"][:, 2].numpy(), surf, u_inf, c["data"].name,
                     args.out / f"case_{name}_cp_band.png")
        metrics.setdefault("cases", {})[label] = {
            "name": str(c["data"].name), "mean_pressure_std": float(case_unc[i]),
        }

    with open(args.out / "uq_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"\nOutputs written to {args.out}")


if __name__ == "__main__":
    main()
