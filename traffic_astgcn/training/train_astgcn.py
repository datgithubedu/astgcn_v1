# training/train_astgcn.py

import os
from pathlib import Path
import json
import pandas as pd
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from traffic_astgcn.data.canonical import CanonicalTrafficData
from traffic_astgcn.data.adapters.astgcn_adapter import ASTGCNAdapter
from traffic_astgcn.models.astgcn import ASTGCN
from traffic_astgcn.utils.metrics import mae, rmse, mape, r2_score


# ===========================================================
# Helper: Load ALL nodes.json để build adjacency
# ===========================================================
def load_all_nodes(runs_root: str | Path) -> pd.DataFrame:
    runs_root = Path(runs_root)
    rows = []

    for d in sorted(runs_root.iterdir()):
        if not d.is_dir():
            continue

        nodes_file = d / "nodes.json"
        if not nodes_file.exists():
            continue

        with nodes_file.open("r", encoding="utf-8") as f:
            nodes = json.load(f)

        for n in nodes:
            rows.append(
                {
                    "node_id": n["node_id"],
                    "lat": n["lat"],
                    "lon": n["lon"],
                }
            )

    df = pd.DataFrame(rows).drop_duplicates("node_id").reset_index(drop=True)
    return df


# ===========================================================
# Evaluate real-scale metrics (inverse scaling per node)
# ===========================================================
def evaluate(model, loader, A_norm, device, node_scaler):
    model.eval()
    A_t = A_norm.to(device)

    preds = []
    trues = []

    with torch.no_grad():
        for Xh, Xd, Xw, Y in loader:
            Xh = Xh.to(device)
            Xd = Xd.to(device)
            Xw = Xw.to(device)
            Y = Y.to(device)

            Y_hat = model(Xh, Xd, Xw, A_t)

            preds.append(Y_hat.cpu().numpy())
            trues.append(Y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)   # (B, N, T)
    trues = np.concatenate(trues, axis=0)   # (B, N, T)

    # ---------- Inverse standard scaling per-node ----------
    B, N, T = preds.shape
    preds_rs = preds.reshape(-1, N)
    trues_rs = trues.reshape(-1, N)

    preds_real = node_scaler.inverse_transform(preds_rs).reshape(B, N, T)
    trues_real = node_scaler.inverse_transform(trues_rs).reshape(B, N, T)

    return {
        "MAE": mae(preds_real, trues_real),
        "RMSE": rmse(preds_real, trues_real),
        "MAPE": mape(preds_real, trues_real),
        "R2": r2_score(preds_real, trues_real),
    }


# ===========================================================
# MAIN TRAINING PIPELINE
# ===========================================================
def main():
    print("========== ASTGCN TRAINING START ==========")

    # ----------------------------------------------------
    # Config paths
    # ----------------------------------------------------
    CANONICAL_PATH = "data/processed/canonical.parquet"
    RUNS_FOLDER = "data/raw/runs"

    # ----------------------------------------------------
    # TensorBoard Writer
    # ----------------------------------------------------
    log_dir = "runs/astgcn"
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logging to: {log_dir}")

    # ----------------------------------------------------
    # Load canonical dataset
    # ----------------------------------------------------
    print("Loading canonical data...")
    canonical = CanonicalTrafficData.from_parquet(CANONICAL_PATH)

    # ----------------------------------------------------
    # Load nodes → build adjacency
    # ----------------------------------------------------
    print("Loading nodes for adjacency matrix...")
    nodes_df = load_all_nodes(RUNS_FOLDER)

    # ----------------------------------------------------
    # Data Adapter → DataLoader
    # (adapter sẽ tự fit scaler per-node)
    # ----------------------------------------------------
    adapter = ASTGCNAdapter(
        seq_len_h=12,
        seq_len_d=12,
        seq_len_w=12,
        horizon=12,
        time_per_day=48,
        day_lag=1,
        week_lag=7,
        batch_size=16,
    )

    print("Building DataLoaders...")
    train_loader, A_np = adapter(canonical, split="train", nodes_df=nodes_df)
    A_norm = torch.tensor(A_np, dtype=torch.float32)

    val_loader, _ = adapter(canonical, split="val", nodes_df=nodes_df)
    test_loader, _ = adapter(canonical, split="test", nodes_df=nodes_df)

    # Node scaler từ adapter (per-node)
    node_scaler = adapter.node_scaler

    # ----------------------------------------------------
    # Model setup
    # ----------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    N = A_norm.shape[0]

    # Lấy sample để biết T_out
    Xh_sample, _, _, Y_sample = next(iter(train_loader))
    T_out = Y_sample.shape[-1]

    model = ASTGCN(
        num_nodes=N,
        in_channels=1,
        out_timesteps=T_out,
        K=3,
        t_kernel=Xh_sample.shape[-1],
        channels=32,
        blocks=2,
    ).to(device)

    A_t = A_norm.to(device)

    # ----------------------------------------------------
    # Training Config
    # ----------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = torch.nn.MSELoss()

    best_val_loss = float("inf")
    epochs = 50

    # ----------------------------------------------------
    # Training Loop (TensorBoard + checkpoint)
    # ----------------------------------------------------
    for ep in range(1, epochs + 1):

        # -------------------- TRAIN --------------------
        model.train()
        train_losses = []

        for Xh, Xd, Xw, Y in train_loader:
            Xh = Xh.to(device)
            Xd = Xd.to(device)
            Xw = Xw.to(device)
            Y = Y.to(device)

            optimizer.zero_grad()

            Y_hat = model(Xh, Xd, Xw, A_t)
            loss = criterion(Y_hat, Y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        writer.add_scalar("loss/train", train_loss, ep)

        # -------------------- VALIDATION --------------------
        model.eval()
        val_losses = []

        with torch.no_grad():
            for Xh, Xd, Xw, Y in val_loader:
                Xh = Xh.to(device)
                Xd = Xd.to(device)
                Xw = Xw.to(device)
                Y = Y.to(device)

                Y_hat = model(Xh, Xd, Xw, A_t)
                val_loss = criterion(Y_hat, Y)

                val_losses.append(val_loss.item())

        val_loss = np.mean(val_losses)
        writer.add_scalar("loss/val", val_loss, ep)

        print(f"Epoch {ep:03d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "astgcn_best.pth")
            print("  => Saved best model")

    writer.close()

    # ----------------------------------------------------
    # Load best model before test
    # ----------------------------------------------------
    print("\nLoading best model checkpoint...")
    model.load_state_dict(torch.load("astgcn_best.pth", map_location=device))

    # ----------------------------------------------------
    # Evaluate on TEST set (real scale)
    # ----------------------------------------------------
    print("\n========== TEST EVALUATION ==========")
    metrics = evaluate(model, test_loader, A_t, device, node_scaler)

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print("=====================================")


if __name__ == "__main__":
    main()
