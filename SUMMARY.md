# Aerodynamic Prediction with GNN — one-page summary

## What this project is

A neural network that predicts the airflow around 2D airfoils **from a picture of the mesh**, instead of running a full CFD solver. The model is a **Graph Neural Network**: each mesh point is a node, physically adjacent points are connected by edges, and the network learns to output four fields at every point — two velocity components, a pressure-related quantity, and a turbulence viscosity.

It's trained on **AirfRANS**, a public dataset of 200 pre-computed CFD simulations designed for the "scarce data" regime — i.e. it's meant to be *hard*, not to look easy.

On top of the baseline, the model also produces an **uncertainty estimate** for every prediction, using a technique called Monte Carlo Dropout. That way, when the model says "here's the pressure field," it also says "and here's how much I trust each part of it." That second part matters more than the first for real engineering use.

## What state the project is in (2026-07-25)

- All the code is written and passes internal smoke tests on made-up data.
- The real dataset (10 GB, 1000 CFD simulations) is downloaded and unpacked on disk.
- **The model has not been trained on the real data yet.** That's the next step.
- Hardware: a personal RTX 2060 laptop GPU with 6 GB of memory. Enough for this project.

## The plan, in plain terms

1. Set up the Python environment (~10 min).
2. Run the two smoke tests to make sure the code works in that environment (~2 min).
3. Run exploratory data analysis on the real dataset — produces some sanity-check plots (~5 min).
4. Convert the raw CFD data into the graph format the neural network expects (~15 min).
5. Do a **short pilot training run** (20 epochs, small model) to measure how fast the GPU is. This tells us whether the full run is feasible locally. (~1 hour)
6. Do the **full training run** (200 epochs, full-size model). Expected to take a few hours on the RTX 2060 — verified in step 5 before committing. Runs unattended; a live dashboard shows loss curves in the browser.
7. **Evaluate** the trained model on 200 held-out test cases. Produces error metrics, predicted-vs-true lift/drag scatter plots, and detailed comparison plots for the best, median, and worst test cases.
8. **Run the uncertainty analysis** — 50 slightly-different predictions per test case, aggregate their spread. Produces uncertainty maps and pressure plots with error bands.
9. Write up the portfolio piece: the metrics from steps 7 and 8, four or five headline plots, one paragraph on where the model is uncertain and whether that uncertainty is trustworthy.

## Cost

Local first — the RTX 2060 should be enough. If it turns out to be too slow (we'll know from the pilot in step 5), **Google Colab's free tier** is the next stop; it gives a T4 GPU with more memory than the RTX 2060, at zero cost. Only if we want to run a hyperparameter sweep do we touch paid cloud, and even then the whole thing fits comfortably under **$15** on an A10 GPU (~$0.60–$0.80/hour, needs about 4 hours).

## What "success" looks like

- The trained model beats "predict the mean" on all four output fields.
- Lift coefficient predicted with **R² > 0.9** (very good) or > 0.7 (still worth writing up) versus the CFD ground truth.
- The uncertainty estimate is not just noise: predictions the model flags as uncertain actually turn out to be less accurate.

The full technical run plan, with the exact commands, is in `plan.md` in the same folder.
