# EDA Metrics Review — Issues, Impact, Fixes

Review of `src/eda_metrics.py` and its outputs in `results/eda_metrics/`, conducted 2026-07-28. This document catalogues six top-priority issues found. Each entry states the problem, the evidence (what proves the problem is real), how it will affect the pipeline downstream if left unfixed, and the concrete fix to apply.

The script itself is comprehensive and well structured — it covers all the EDA topics that a portfolio reviewer would ask for, reuses the project's own `postprocess.force_coefficients` instead of duplicating logic, and its outputs already validate several important physics assumptions (e.g. `Cp_max ≈ 1.0` at every stagnation point, `avg_degree ≈ 6` for the Delaunay graph). The issues below are the residual quality problems that would undermine trust in the numbers if a reviewer went digging.

Fixes are ordered by priority: bugs first, then presentation, then a diagnostic add. Numbering matches the tags used in the review conversation.

---

## B1 — Master table includes the 20 validation simulations

### Problem

Lines 625–626 correctly split `train_only_raw = train_raw[:-20]` (the 180 simulations meant for training) from `val_raw = train_raw[-20:]` (the 20 held out for validation, mirroring what `data_processing.py` does). But line 657 then calls `build_master_table(train_raw, train_names)` — passing the full 200-simulation set, val included.

### Evidence

- `results/eda_metrics/master_metrics_table.csv` has 200 data rows (201 lines including the header). If the val holdout were respected, it would have 180.
- The `train_only_raw` variable is created on line 626 but never used inside `build_master_table`.

### Pipeline impact

The master table drives every summary plot after it: `plot_flight_params`, `plot_cl_cd`, `plot_cp_analysis`, `plot_delaunay_aggregated`, `plot_mesh_density_vs_nodes`, `plot_boundary_layer_analysis`, and `plot_correlation_heatmap`. Every one of these summaries currently reflects statistics that include the 20 validation simulations.

This is a small val-leak in the *analysis* rather than the training loop, but it matters in three concrete ways:

1. If you use the master-table summaries (e.g. observed C_D range, wake-turbulence distribution) to inform hyperparameter or normalization choices, the choices will be tuned against the val set the model will later be scored on.
2. Correlation heatmap coefficients are computed on 200 rows including val, so any effect size reported (e.g. "AoA vs C_L r = 0.96") is not a pure training-set observation.
3. In a portfolio write-up, "the training-set distribution has these properties" is a common sentence — it will be false as currently produced.

### Fix

Change line 657 to pass the 180-sim split.

---

## B2 — Negative C_D values in the master CSV point to a surface-ordering artefact in `postprocess.order_surface`

### Problem

The pressure-integrated drag coefficient `CD` returned by `postprocess.force_coefficients` is negative for a subset of simulations in the master CSV. Pressure drag on a normally-oriented airfoil in freestream should be non-negative — it can be very small when viscous shear dominates (viscous shear is not included here), but a *sign* flip indicates the surface normal was integrated in the wrong direction over at least part of the contour.

### Evidence

- Row 3 of `master_metrics_table.csv`: `CD = -0.01647`, `L_over_D = -78.14`. Row 5: `CD = 0.00173` (tiny positive — well inside the pressure-only noise floor).
- `postprocess.order_surface` orders surface points by angle around the *centroid* of the surface points and assumes CCW ordering follows. That assumption breaks whenever the airfoil polygon is not star-shaped with respect to its centroid — which can happen for thin, highly-cambered, or reflexed sections, all of which appear in the AirfRANS shape family. When one segment is out of order, its outward-normal-times-length vector `(dy, -dx)` flips sign, and its contribution to the force integral cancels neighbouring segments instead of adding to them.

### Pipeline impact

If this is not surfaced, three things break:

1. Any C_D metric produced from these sims — the drag polar in `aero_drag_polar.png`, `plot_cl_cd`, correlations involving C_D — is polluted with garbage rows. R² on C_D at test time (from `evaluate.py`) will be measured against a truth signal that is itself wrong for those cases, giving misleading "the model failed on drag" impressions when in fact the label is wrong.
2. `L/D` is unreliable wherever C_D is near zero or negative, so `plot_cl_cd`'s `aero_ld_vs_aoa.png` and the correlation with L/D are similarly polluted.
3. In the portfolio write-up, if you cite a lift/drag correlation from these summaries, a reviewer with an aero background will spot the sign issue and lose trust in the whole analysis.

The root cause lives in `postprocess.order_surface`, which is out of scope for a single-file `eda_metrics.py` fix. What can be done here is to *diagnose and surface* the affected simulations so you can (a) exclude them from downstream summaries, (b) fix `order_surface` later using a more robust ordering (e.g. `shapely.geometry.Polygon` with orientation check, or nearest-neighbour chaining along the surface).

### Fix

Add a small diagnostic function that counts and lists simulations with negative or suspiciously-small C_D, and saves the report as `cd_diagnostics.json`. This does not repair `postprocess.order_surface` — it just makes the problem visible so the affected sims can be excluded from write-up statistics.

---

## B3 — `np.nan_to_num` in the correlation heatmap silently substitutes 0 for missing values

### Problem

`plot_correlation_heatmap` replaces every NaN, +Inf, and −Inf with 0 before calling `np.corrcoef`:

```python
data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
```

Substituting 0 is not a neutral choice for a correlation calculation. Zero is a valid value in the domain of every column, so the substitution poisons the correlation instead of removing the affected rows.

### Evidence

- The master CSV computes `L_over_D = cl / cd` (line 262 of `eda_metrics.py`, guarded against division by zero with a NaN fallback). Combined with B2, several rows have `L_over_D` values whose reliability is already questionable.
- Any row where C_D is exactly zero produces `L_over_D = NaN`, which `np.nan_to_num` then replaces with 0 — a value that is not just wrong but is inside the plausible range of L/D and therefore invisible.

### Pipeline impact

The correlation heatmap is the single densest summary figure in the EDA output. A reviewer will look at each cell as a signal. A substituted-zero row pulls correlations toward zero and, more importantly, introduces spurious mid-range correlations if the affected column has any variance from the substituted rows. The claim "L/D correlates weakly with everything" then becomes untestable: is it really weakly correlated, or is it that the substituted zeros are diluting a real signal?

### Fix

Replace the substitution with a row-mask: drop simulations that have any non-finite value in the selected columns before computing the correlation. Log how many rows were dropped so the reader knows.

---

## P1 — Missing colorbars on `aero_cl_vs_aoa`, `aero_cd_vs_aoa`, `aero_ld_vs_aoa`

### Problem

All three of the AoA-scatter plots colour their points by inlet velocity (`c=vel, cmap="viridis"`), but only `aero_drag_polar.png` calls `fig.colorbar(...)` to draw the legend for that colouring. The other three plots use colour to encode a real variable but never tell the reader what it means.

### Evidence

- Inspection of `results/eda_metrics/aero_cl_vs_aoa.png` (opened in the review): points are visibly coloured on a viridis scale but there is no colorbar on the figure.
- Same pattern applies to `aero_cd_vs_aoa` and `aero_ld_vs_aoa` per the code (`plot_cl_cd`, lines 438–460).

### Pipeline impact

Low-severity but real. In a portfolio deliverable, a reader who sees colour-encoded points without a colorbar has three choices: assume the colour is decorative (loses the third dimension of the plot), guess what the colour means (usually wrong), or dismiss the plot as sloppy. All three outcomes are avoided by three lines of code.

### Fix

Add `fig.colorbar(sc, ax=ax, label="Inlet Vel [m/s]")` on each of the three plots.

---

## P2 — Correlation heatmap contains collinear column pairs that fill space with r = 1.00

### Problem

The `keys` list in `plot_correlation_heatmap` includes both `inlet_velocity_m_s` and `reynolds`, and both `n_nodes` and `mesh_density_nodes_per_m2`. Each of these pairs is exactly collinear by construction:

- `reynolds = inlet_velocity_m_s * 1.0 / 1.56e-5` — a constant multiple of inlet velocity.
- `mesh_density_nodes_per_m2 = n_nodes / (x_range * y_range)` — since the AirfRANS bounding box is essentially constant across simulations, this is a constant multiple of `n_nodes`.

`n_nodes` is also nearly collinear with `boundary_layer_fraction` (r = 0.96 in the plot) because the boundary-layer threshold `d < 0.05 m` is a fixed geometric criterion — more nodes in the mesh means more of them fall inside a fixed near-wall band.

### Evidence

- Direct inspection of `results/eda_metrics/correlation_heatmap.png`: cells `Inlet Vel × Reynolds`, `Nodes × Mesh Density` show r = 1.00; `Nodes × BL Fraction` shows r = 0.96.
- Code inspection of `parse_sim_name` and `mesh_density` confirms the analytic relationships.

### Pipeline impact

The heatmap is a 23×23 grid with 529 cells. When two columns are collinear, they duplicate all their off-diagonal correlations — every other column produces the same row of r-values for the collinear pair. That means roughly 2 × 23 = 46 cells are exact duplicates of another cell for each collinear pair, and the eye has to filter through them to find real signal.

Beyond visual clutter, correlations with two collinear columns are not statistically independent — you cannot cite "Reynolds correlates with C_L" and "Inlet Vel correlates with C_L" as separate findings; they are one finding printed twice. If the heatmap is used to select features for later analysis, keeping both would double-count the same effect.

### Fix

Drop `reynolds` and `mesh_density_nodes_per_m2` from the `keys` and `labels` lists. Leave `boundary_layer_fraction` in — r = 0.96 is high but not perfect, and it has a distinct physical interpretation.

---

## M1 — `naca_params` are parsed but never decomposed into a named `thickness_pct` column

### Problem

`parse_sim_name` extracts a `naca_params` list (3 numbers for NACA 4-digit, 4 for NACA 5-digit) from every simulation filename but discards it — the master table row dict at line 288 stores `naca_series` (4 or 5) but not any of the individual geometry parameters. The most informative one — airfoil thickness as a percentage of chord — is available in every simulation and is trivial to extract.

### Evidence

- `parse_sim_name` returns `"naca_params": naca_params` (line 67) but `build_master_table` never references `fp["naca_params"]` (lines 288–329).
- Per the AirfRANS naming convention (paper: arXiv:2212.07564; docs: airfrans.readthedocs.io), thickness in percent of chord is the last element of the parameter tuple for both NACA-4 (position 3) and NACA-5 (position 4). This lets a single indexing rule — `naca_params[-1]` — extract thickness across both series.

### Pipeline impact

Without a `thickness_pct` column, correlations of C_L, C_D, or C_p_min with airfoil geometry are impossible. The reviewer question "does thin-airfoil performance differ from thick-airfoil performance in your dataset?" cannot be answered from the current CSV. This is exactly the kind of analysis a portfolio deliverable is expected to include — the training set spans thickness variations, and a competent write-up should show how the target variables shift with thickness.

Extracting only `thickness_pct` (not the full geometry) is a deliberate scope choice: the mapping from `naca_params[0]` and `naca_params[1]` to physical properties (max camber, camber position, design lift index) differs between NACA-4 and NACA-5 and would require verifying against the AirfRANS reference implementation before use. Thickness is the safe, high-value first addition.

### Fix

- Add `thickness_pct = naca_params[-1]` to the return dict of `parse_sim_name`.
- Add `"thickness_pct": fp["thickness_pct"]` to the row built in `build_master_table`.
- Optionally also include `thickness_pct` in the heatmap `keys`/`labels` so the correlation with C_L, C_D and C_p can be visualised.

---

## Re-running after fixes

The fixes above are pure code changes to `src/eda_metrics.py`. The existing outputs in `results/eda_metrics/` were produced before the fixes and are therefore **stale after any of the fixes land**. To regenerate:

```bash
conda activate gnn_surrogate
python src/eda_metrics.py
```

Expected runtime: 30–60 minutes (dominated by the Delaunay triangulation and pressure integration on 180 sims). All output files in `results/eda_metrics/` will be overwritten.

Files that will change:
- `master_metrics_table.csv` — 180 rows instead of 200 (B1), gains `thickness_pct` column (M1).
- `correlation_heatmap.png` — 21×21 instead of 23×23 (P2), rows dropped rather than zero-substituted (B3).
- `aero_cl_vs_aoa.png`, `aero_cd_vs_aoa.png`, `aero_ld_vs_aoa.png` — now include colorbars for inlet velocity (P1).
- `cd_diagnostics.json` — new file listing sims with negative or noise-floor C_D (B2).
- All the other summary plots — recomputed on 180 sims rather than 200 (B1). Numerical differences will be small but the values are now honest.
