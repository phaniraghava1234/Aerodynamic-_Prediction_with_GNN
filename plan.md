# plan.md — end-to-end run plan for the AirfRANS GNN surrogate

Author: Claude Code, 2026-07-25. Written from a fresh scan of this repo. All numbers labelled "measured" were verified this session; everything else is labelled as assumption, external claim, or unknown.

---

## 1. Goal (restated)

Take this repo from its current state — code complete, dataset extracted, no run yet — to a full trained model, evaluation report, and MC-Dropout uncertainty report, then a short portfolio write-up. Do it on the local RTX 2060 6 GB if we can; use a free/cheap cloud GPU only if the local run is impractical, staying under a **$15 total** cloud budget.

---

## 2. Current state (measured 2026-07-25)

| Item | State | Evidence |
|---|---|---|
| `src/*.py` (9 files) | Complete, self-consistent | Read each file this session |
| `tests/test_pipeline.py`, `tests/test_full_pipeline.py` | Present | `PROGRESS.md` says both pass on synthetic; not re-run this session |
| `data/Dataset/` | Extracted, 1000 sim folders + `manifest.json` | `iterdir()` count = 1001; sample names printed |
| `data/Dataset.zip` (~10 GB) | Still present after extraction | `ls -la` |
| `data/processed/` | Does not exist | `ls` returned "No such file or directory" |
| `models/` | Empty | `ls` |
| `results/` | Empty | `ls` |
| GPU | RTX 2060, 6144 MiB, ~5.1 GiB free, CUDA UMD 13.3, driver 610.62 | `nvidia-smi` |
| Python | Anaconda 3.11.3, no `torch` / `torch_geometric` / `airfrans` / `tensorboard` | `python -c "import torch"` → ModuleNotFoundError |
| `.venv` | Does not exist in repo | `ls -a` on repo root |

**Consequence:** step zero is environment setup. Steps in `PROGRESS.md`'s "Next up" section are still the right order.

---

## 3. Constraints

- **Hardware:** RTX 2060 6 GB VRAM (~5.1 GiB effectively free); i7-9750H; 32 GB RAM.
- **Time:** the repo's own `CLAUDE.md` claims "expect a few hours for 200 epochs" — this is an external claim, **not verified**. Treat as an order-of-magnitude hint until the pilot at §4.5 gives real seconds-per-epoch.
- **VRAM head-room:** AirfRANS graphs are ~180k nodes each; `CLAUDE.md` says default batch size 1 fits 6 GB. Do not change without a memory test.
- **Cloud budget:** ≤ $15 total across all cloud spend for this project.
- **Scope discipline** (per this repo's `CLAUDE.md`): no physics-informed loss, no hierarchical GNN, no multi-fidelity, no extra models. Delaunay-edges-through-interior, one-bit surface flag, pressure-only drag, angle-ordered surface, fixed 20-of-200 val slice — all deliberate scope cuts; leave them alone.

---

## 4. Step-by-step plan

Each step is a checkbox. Run in order. After a step, look at its "gate" before starting the next.

### 4.1 Environment setup (~2 minutes)

Use the existing `gnn_surrogate` conda env — verified 2026-07-25 to contain Python 3.12.11, torch 2.8.0+cu128 (CUDA working on RTX 2060), torch-geometric 2.6.1 with the full C++ extension stack (pyg-lib, torch-cluster, torch-scatter, torch-sparse, torch-spline-conv), airfrans 0.1.5.1, numpy 2.1.2, scipy 1.16.2, scikit-learn 1.7.2, matplotlib 3.10.6, pyvista 0.46.3.

```bash
conda activate gnn_surrogate
```

Only one package from `requirements.txt` is missing: `tensorboard`. Install just that:

```bash
pip install tensorboard
```

**Do not** re-run `pip install -r requirements.txt` blindly — it would pull in duplicates and could disturb the working torch+cu128 stack.

**Gate (already passing as of the check on 2026-07-25 — re-run only if the env has changed since):**

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import torch_geometric, airfrans, tensorboard; print('all critical imports OK')"
```

Expected: `True NVIDIA GeForce RTX 2060` on the first line, `all critical imports OK` on the second.

### 4.2 Smoke tests (2–3 minutes, no data needed)

```bash
python tests/test_pipeline.py
python tests/test_full_pipeline.py
```

**Gate:** both print "passed" and exit 0. These use synthetic data — they prove the code path works in *your* env, not that the real dataset works. Do not skip.

### 4.3 EDA on the real dataset (~5 minutes)

```bash
python src/eda.py
```

Writes to `results/eda/`: `sample_meshes.png`, `target_distributions.png`, `eda_stats.json`.

**Gate:** open `results/eda/eda_stats.json`. Sanity check:
- `n_simulations` = 180 (train split is 200 total; `data_processing.py` will hold out 20 for val).
  Wait — re-check: `eda.py` loads the full 200-sim scarce split; the 20-sim val holdout happens inside `data_processing.py`. So `eda.py` should report **200**, and `data_processing.py` will report 180 train + 20 val.
- `nodes_per_sim_min` and `_max`: both should be in the ~150k–200k range (AirfRANS typical).
- Sample graph stats: `avg_degree` ≈ 6 (Delaunay in 2D).

If any of those look wrong, stop and diagnose before Phase 4.4.

### 4.4 Convert to PyG graphs (~10–20 minutes, one-time)

```bash
python src/data_processing.py
```

Writes `data/processed/{train,val,test}.pt` (~180 + 20 + 200 graphs) and `data/processed/norm_stats.pt`.

**Gate:**
- Script prints "train: 180 graphs, ~N nodes → …" etc.
- On disk `data/processed/` totals a few GB (each `.pt` will be ~1 GB depending on edge count).
- If Delaunay chokes on any simulation (extreme aspect ratios), the script will crash there — record which sim name in an issue and drop it manually as a follow-up; do not silently skip.

### 4.5 Pilot training run (**start here — do not go straight to 200 epochs**)

The whole point of a pilot is to answer three questions **before** committing to a multi-hour run:
1. Does the loop actually decrease loss on real data?
2. How long is one epoch on this GPU?
3. Do we hit OOM at the default hidden dim?

```bash
python src/train.py --epochs 20 --hidden 64 --checkpoint models/pilot.pt --log-dir results/runs/pilot
```

`--hidden 64` first, not the default 128 — cheaper on VRAM, still enough signal.

**In another terminal:**
```bash
tensorboard --logdir results/runs
```
Open http://localhost:6006 and watch `loss/train` and `loss/val`.

**Gates (all three must hold before Phase 4.6):**
- Train loss is monotonically decreasing over the 20 epochs (not necessarily every epoch, but the trend).
- Val loss is either decreasing or plateauing above train loss (as expected with dropout).
- Per-epoch wall time is < 5 minutes. If it's much larger, we have to move to cloud (§6) before the full run.

**If OOM:** the script will raise `torch.OutOfMemoryError`. In that case, drop `--hidden` to 32 and try again. Do **not** touch batch size — it's fixed at 1 for a reason.

### 4.6 Full training run

Two choices, decide from what the pilot showed:

**Option A — pilot looked good → full 200-epoch run at hidden 128:**
```bash
python src/train.py --epochs 200 --hidden 128
```
Writes `models/baseline_best.pt` (best val), logs to `results/runs/<timestamp>/`.

**Option B — pilot was borderline (near OOM, or slow) → run at hidden 64:**
```bash
python src/train.py --epochs 200 --hidden 64
```
The model has fewer parameters (~42k vs ~126k), portfolio numbers will be a touch worse, but it *will* finish.

**Recommendation:** default to Option A unless the pilot forces B. Only escalate to cloud (§6) if even Option B doesn't fit in your daily uptime window.

**Training-strategy note (answers "batch training vs one shot?" from the original message):**
- Batch size **must stay at 1**. AirfRANS graphs are ~180k nodes each; `torch_geometric.loader.DataLoader` with batch_size=2 would stack them into a 360k-node super-graph and blow the 6 GB budget. This is not a preference, it's a hard constraint set by the data — `CLAUDE.md` calls it out explicitly.
- What you *do* get for free is **one full pass through all 180 training graphs per epoch**, one graph at a time — which is already what the code does. There is no separate "train 1 batch then check result then train another batch" mode to enable; the training loop already checkpoints the best-validation model each epoch, so if the run dies you resume from `baseline_best.pt` by re-running with a fresh `--epochs` count (the current `train.py` does not have `--resume`; if you want that, add it deliberately — a 4-line change loading `state_dict` at start).
- The learning-rate schedule is `ReduceLROnPlateau(factor=0.5, patience=10)` — it will halve the LR after 10 val-loss stall epochs. There is **no early stopping**; the run always finishes the epoch count. Optional cheap add if you want it: kill the process when `lr < 1e-6`, and use the best-checkpoint anyway.

### 4.7 Evaluation

```bash
python src/evaluate.py
```

Writes to `results/evaluation/`: `metrics.json`, `lift_drag_scatter.png`, and per-case field/Cp/streamline plots for best/median/worst-by-pressure-MSE cases.

**Gate:** open `metrics.json`. Compare to what you'd expect for a small GNN on scarce data:
- `field_mse` on velocity should be < variance of the target (i.e. beating "predict the mean"). If not, training didn't converge.
- `lift.r2` should be > 0.7 to be worth writing up; > 0.9 is a genuinely good result on this data-size regime.
- `drag_pressure_only.r2` will typically be worse than lift's, because pressure-integrated drag misses viscous shear (a known limitation, documented in `README.md`).

### 4.8 MC Dropout uncertainty quantification

```bash
python src/evaluate_uq.py --passes 50
```

Writes `results/uq/`: `uq_metrics.json`, `uncertainty_vs_error.png`, uncertainty-map and Cp-with-±2σ-band plots for the most and least uncertain test cases.

**Gate:** `uq_metrics.json["uncertainty_error_pearson_r"]` — the four Pearson correlations between predictive std and absolute error. If any of them is > 0.3, the story "MC Dropout σ flags high-error regions" holds up; if all are near zero, the UQ chapter is weaker (report it honestly, don't tune the passes count until it looks better — that's data-snooping).

### 4.9 Portfolio write-up

Fill in the checklist in `README.md` "Portfolio write-up checklist (after a real training run)":
- Test metrics from `results/evaluation/metrics.json` and `results/uq/uq_metrics.json`.
- Strongest figures: Cp-with-uncertainty-band, lift/drag scatter, one field-error map, one uncertainty map.
- One paragraph on where uncertainty concentrates and whether σ correlates with error.

Add one entry to `PROGRESS.md` before ending the session. Template already there.

---

## 5. Training-strategy details (batching, VRAM, OOM ladder)

Answers to specific questions from the original message.

**"Can we batch-train and check between batches?"**
Not in the way you meant. Two things are getting conflated:
- **Mini-batching** across graphs: as above, this is capped at batch size 1 by graph size. Not a knob.
- **Iterative training** (train a bit, look, train more): you already have this via TensorBoard + the best-val checkpoint. `tail -f`-style monitoring in `results/runs/`. If a run is going nowhere, kill it with Ctrl-C; the best epoch is already saved.

**OOM ladder (top of the ladder first, only step down when you have to):**
1. `--hidden 128`, batch 1 → the default the repo was designed for.
2. `--hidden 64`, batch 1.
3. `--hidden 32`, batch 1 (this is what the smoke test uses).
4. Neighbor-sampling / subgraph training via `torch_geometric.loader.NeighborLoader` — this is a **real code change**, not a flag. Do not do this without asking; it changes the training semantics enough that the evaluation code may need updates too.

**"Should I train one batch, look, retrain?"** No — that adds bias (you're peeking at val implicitly). Just let the pilot run 20 epochs uninterrupted, then decide.

---

## 6. Cloud fallback plan (only if local is impractical)

Trigger conditions (all measured, not guessed):
- Pilot at §4.5 shows > 15 min/epoch → local full run would take > 50 hours → move to cloud.
- Pilot OOMs even at `--hidden 32` → cloud gives you a bigger GPU.
- You want to run a small hyperparameter sweep (e.g. hidden ∈ {64, 128, 192}, lr ∈ {1e-3, 3e-4}) that would take too long locally.

Otherwise, **do not go to cloud** — the local RTX 2060 should handle the baseline run per the repo's own estimate.

### 6.1 Option A — Google Colab free (default cloud choice)

- Typical allocation: NVIDIA T4, 15 GB VRAM (~2× RTX 2060 raw FP32, way more VRAM).
- Session time-out: ~12 hours idle, 12 hours total on the free tier (Google publishes this; policy shifts, verify).
- Cost: $0.

**Workflow:** upload the whole repo (excluding `data/Dataset.zip`) to Google Drive; `data_processing.py` locally, then upload `data/processed/*.pt` + `norm_stats.pt` (~a few GB) to Drive; open a Colab notebook, mount Drive, install requirements, run `train.py`, download `models/baseline_best.pt` back.

This alone probably closes the gap for you. Try Colab free before spending any money.

### 6.2 Option B — Colab Pro (pay-per-compute-unit)

- ~$10/month for Pro, ~$50/month for Pro+; both are subscriptions, not per-run.
- Not a natural fit for a one-shot training campaign under $15.

### 6.3 Option C — NVIDIA Brev / Lambda / RunPod

- Approximate current prices (verify before booking — these move):
  - NVIDIA T4: ~$0.30–$0.50/hr
  - NVIDIA A10 (24 GB): ~$0.60–$0.80/hr
  - NVIDIA A100 40 GB: ~$1.20–$1.80/hr
- $15 / $0.80 ≈ **18 hours on an A10**, which is enough to comfortably run a baseline plus a small sweep with headroom.
- Pick A10 over A100 for this workload — the graph is small enough that A100's extra bandwidth is wasted; you'd pay 2× for maybe 1.3× throughput.

**Budget-safe pattern:**
1. Launch instance, `git clone` the repo, `pip install -r requirements.txt`.
2. `rsync` up `data/processed/*.pt` (do NOT upload the raw 10 GB `Dataset.zip` — that's what `data_processing.py` was for).
3. `python src/train.py --epochs 200` — capture wall time.
4. Copy `models/baseline_best.pt` back.
5. **Stop the instance the second training finishes.** Set a personal alarm; forgotten instances are how $15 becomes $150.

### 6.4 Rough $15 budget math

| Path | Est. cost | Fits budget? |
|---|---|---|
| Local RTX 2060, 200 epochs | $0 (electricity) | ✅ |
| Colab free, 200 epochs | $0 | ✅ |
| Brev A10, 200 epochs (assume ~4h) | ~$3 | ✅, big margin |
| Brev A10, sweep of 4 configs × 200 epochs | ~$12 | ✅, tight |
| Brev A100, sweep of 4 configs × 200 epochs | ~$20 | ❌ — too much |

**Recommendation:** local → Colab free → A10 only if you want the sweep. Never A100 for this project size.

---

## 7. Risks and known unknowns

Named honestly so we can plan around them.

- **Per-epoch wall time on RTX 2060 for these graphs is unknown to me.** The repo's "a few hours for 200 epochs" is an external claim I cannot verify without running the pilot. Everything at §4.6 depends on the pilot's number, which is why the pilot is a gate not a suggestion.
- **CUDA wheel choice.** Resolved — `gnn_surrogate` already runs torch 2.8.0+cu128 with `torch.cuda.is_available() == True` on this RTX 2060 (verified 2026-07-25).
- **Delaunay edges through the airfoil interior.** Not a bug — a documented, deliberate simplification in this repo. If the field/error maps in §4.7 show artifacts hugging the surface, we can filter Delaunay edges that cross the surface as a targeted follow-up. Do not fold that into this run.
- **Drag is pressure-only.** CL numbers are trustworthy; CD is not compared to viscous-inclusive CFD. Report as "pressure drag" in the write-up.
- **Val split is fixed 20 sims off the end of the train list**, not randomized. Trades IID validation for reproducibility — noted in `CLAUDE.md`. Do not "fix" this by shuffling.
- **`train.py` has no `--resume` flag.** If a long run gets interrupted (power, OS update, mistake), you restart from scratch — the checkpoint only stores the best model weights, not the optimizer/scheduler state. Adding `--resume` is a small deliberate change; ask before doing it, per Rule 4.

---

## 8. Confidence

**Confidence: Medium-High.**

*High* on: the current state of the repo (measured), the correctness of the ordering of steps 4.1–4.9 (matches this repo's own PROGRESS.md next-up list), and the "batch size = 1 is a hard constraint" claim (documented in this repo).

*Medium* on: the wall-time estimates in §5 and §6 — those depend on GPU throughput I have not measured. The pilot at §4.5 is designed to convert those medium-confidence guesses into measured numbers before you commit to a long run.

*Low* on: exact current cloud pricing — that changes month to month; check the vendor site before booking.
