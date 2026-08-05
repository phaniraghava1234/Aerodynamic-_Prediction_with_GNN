"""End-to-end smoke test: synthetic dataset -> train -> evaluate -> evaluate_uq.

Run:  python tests/test_full_pipeline.py
Uses a circular 'airfoil' so surface ordering and lift/drag integration are
exercised. Verifies that all three scripts run and produce their outputs.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from data_processing import build_graph, fit_norm_stats

TMP = PROJECT / "tests" / "tmp"
DATA = TMP / "processed"


def synthetic_sim(seed, n_field=1200, n_surf=80):
    """Point cloud around a circular 'airfoil' of radius 0.15 at (0.5, 0)."""
    rng = np.random.default_rng(seed)
    center, radius = np.array([0.5, 0.0]), 0.15
    pts = rng.uniform([-1.0, -1.0], [2.5, 1.0], size=(n_field * 2, 2))
    pts = pts[np.linalg.norm(pts - center, axis=1) > radius][:n_field]
    theta = np.linspace(0, 2 * np.pi, n_surf, endpoint=False)
    surf_pts = center + radius * np.stack([np.cos(theta), np.sin(theta)], axis=1)
    normals = np.stack([np.cos(theta), np.sin(theta)], axis=1)

    n = len(pts) + n_surf
    sim = np.zeros((n, 12))
    sim[:, 0:2] = np.concatenate([pts, surf_pts])
    aoa = np.deg2rad(rng.uniform(-5, 10))
    uinf = 40 + 20 * rng.random()
    sim[:, 2:4] = [uinf * np.cos(aoa), uinf * np.sin(aoa)]
    sim[:, 4] = np.linalg.norm(sim[:, 0:2] - center, axis=1) - radius
    sim[len(pts):, 5:7] = normals
    sim[len(pts):, 11] = 1
    # Smooth learnable targets
    x, y = sim[:, 0], sim[:, 1]
    sim[:, 7] = uinf * (1 - np.exp(-2 * sim[:, 4]))
    sim[:, 8] = 5 * np.sin(2 * x) * np.exp(-sim[:, 4])
    sim[:, 9] = 0.5 * uinf**2 * np.exp(-3 * sim[:, 4]) * np.cos(np.arctan2(y, x - 0.5))
    sim[:, 10] = 1e-3 * np.exp(-sim[:, 4])
    return sim


if TMP.exists():
    shutil.rmtree(TMP)
DATA.mkdir(parents=True)

sims = [synthetic_sim(s) for s in range(12)]
stats = fit_norm_stats(sims[:6])
torch.save(stats, DATA / "norm_stats.pt")
for split, sl in {"train": slice(0, 6), "val": slice(6, 8), "test": slice(8, 12)}.items():
    graphs = [build_graph(s, stats, f"synth_{split}_{i}") for i, s in enumerate(sims[sl])]
    torch.save(graphs, DATA / f"{split}.pt")

env_args = ["--data-dir", str(DATA)]
ckpt = TMP / "model.pt"


def run(script, *extra):
    cmd = [sys.executable, str(PROJECT / "src" / script), *env_args,
           "--checkpoint", str(ckpt), *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        raise SystemExit(f"{script} failed")
    return r.stdout


run("train.py", "--epochs", "2", "--hidden", "32", "--log-dir", str(TMP / "runs"))
assert ckpt.exists()

out_eval = TMP / "evaluation"
run("evaluate.py", "--out", str(out_eval))
expected = ["metrics.json", "lift_drag_scatter.png"]
missing = [f for f in expected if not (out_eval / f).exists()]
assert not missing, missing
assert len(list(out_eval.glob("case_*_fields.png"))) == 3
assert len(list(out_eval.glob("case_*_cp.png"))) == 3
assert len(list(out_eval.glob("case_*_streamlines.png"))) == 3

out_uq = TMP / "uq"
run("evaluate_uq.py", "--out", str(out_uq), "--passes", "5")
assert (out_uq / "uq_metrics.json").exists()
assert (out_uq / "uncertainty_vs_error.png").exists()
assert len(list(out_uq.glob("case_*_uncertainty_map.png"))) == 2
assert len(list(out_uq.glob("case_*_cp_band.png"))) == 2

# MC dropout must actually be stochastic: std should not be all zeros
import json
uq = json.loads((out_uq / "uq_metrics.json").read_text())
assert any(v > 0 for v in uq["mean_predictive_std"].values()), "no stochasticity"

print("Full pipeline smoke test passed: train -> evaluate -> evaluate_uq.")
