"""Train the baseline GNN on the processed AirfRANS graphs.

Run:  python src/train.py [--epochs 200] [--batch-size 1] [--hidden 128]
Saves the checkpoint with the lowest validation loss to models/baseline_best.pt
and TensorBoard logs to results/runs/.

Memory note (6 GB GPU): AirfRANS graphs have ~180k nodes each, so the default
batch size is 1. Reduce --hidden if you still hit OOM.
"""
import argparse
import time
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import BaselineGNN

PROJECT = Path(__file__).resolve().parents[1]


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total, n_nodes = 0.0, 0
    with torch.enable_grad() if training else torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            loss = torch.nn.functional.mse_loss(pred, batch.y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * batch.num_nodes
            n_nodes += batch.num_nodes
    return total / n_nodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=PROJECT / "data" / "processed")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--checkpoint", type=Path, default=PROJECT / "models" / "baseline_best.pt")
    ap.add_argument("--log-dir", type=Path, default=PROJECT / "results" / "runs")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = torch.load(args.data_dir / "train.pt", weights_only=False)
    val_set = torch.load(args.data_dir / "val.pt", weights_only=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    print(f"device={device}  train={len(train_set)}  val={len(val_set)}")

    model = BaselineGNN(hidden=args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=10
    )
    writer = SummaryWriter(args.log_dir / time.strftime("%Y%m%d-%H%M%S"))

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, device, optimizer)
        val_loss = run_epoch(model, val_loader, device)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("lr", lr, epoch)
        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"state_dict": model.state_dict(), "hidden": args.hidden},
                       args.checkpoint)
            marker = "  *saved*"
        print(f"epoch {epoch:4d}  train {train_loss:.5f}  val {val_loss:.5f}  "
              f"lr {lr:.1e}  {time.time() - t0:.1f}s{marker}")
    writer.close()
    print(f"Best validation loss: {best_val:.5f}  ->  {args.checkpoint}")


if __name__ == "__main__":
    main()
