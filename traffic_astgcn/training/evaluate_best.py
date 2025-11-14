import torch
import numpy as np
from pathlib import Path

from traffic_astgcn.data.canonical import CanonicalTrafficData
from traffic_astgcn.data.adapters.astgcn_adapter import ASTGCNAdapter
from traffic_astgcn.models.astgcn import ASTGCN
from traffic_astgcn.utils.metrics import mae, rmse, mape, r2_score

import pandas as pd
import json


def load_all_nodes(runs_root):
    runs_root = Path(runs_root)
    rows = []

    for d in sorted(runs_root.iterdir()):
        if (d / "nodes.json").exists():
            with open(d / "nodes.json", "r", encoding="utf-8") as f:
                nodes = json.load(f)
            rows.extend(nodes)

    df = pd.DataFrame(rows).drop_duplicates("node_id")
    return df[["node_id", "lat", "lon"]]


def main():
    print("=== EVALUATE BEST MODEL ===")

    CANONICAL_PATH = "data/processed/canonical.parquet"
    RUNS_FOLDER = "data/raw/runs"

    canonical = CanonicalTrafficData.from_parquet(CANONICAL_PATH)
    nodes_df = load_all_nodes(RUNS_FOLDER)

    adapter = ASTGCNAdapter(
        seq_len_h=12,
        seq_len_d=12,
        seq_len_w=12,
        horizon=12,
        time_per_day=48,
    )

    _, _ = adapter(canonical, split="train", nodes_df=nodes_df)
    test_loader, A_np = adapter(canonical, split="test", nodes_df=nodes_df)

    A_norm = torch.tensor(A_np, dtype=torch.float32)

    N = A_norm.shape[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get T_out
    _, _, _, Ysample = next(iter(test_loader))
    T_out = Ysample.shape[-1]

    model = ASTGCN(
        num_nodes=N,
        in_channels=1,
        out_timesteps=T_out,
        K=3,
        t_kernel=12,
        channels=32,
        blocks=2,
    ).to(device)

    print("Loading checkpoint astgcn_best.pth ...")
    model.load_state_dict(torch.load("astgcn_best.pth", map_location=device))
    model.eval()

    preds = []
    trues = []

    scaler = adapter.node_scaler

    with torch.no_grad():
        for Xh, Xd, Xw, Y in test_loader:
            Xh = Xh.to(device)
            Xd = Xd.to(device)
            Xw = Xw.to(device)
            Y = Y.to(device)

            Y_hat = model(Xh, Xd, Xw, A_norm.to(device))

            preds.append(Y_hat.cpu().numpy())
            trues.append(Y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    # inverse scale
    B, N, T = preds.shape
    preds_rs = preds.reshape(-1, N)
    trues_rs = trues.reshape(-1, N)

    preds_real = scaler.inverse_transform(preds_rs).reshape(B, N, T)
    trues_real = scaler.inverse_transform(trues_rs).reshape(B, N, T)

    print("\n=== TEST METRICS ===")
    print("MAE :", mae(preds_real, trues_real))
    print("RMSE:", rmse(preds_real, trues_real))
    print("MAPE:", mape(preds_real, trues_real))
    print("R²  :", r2_score(preds_real, trues_real))

    print("\nDONE.")


if __name__ == "__main__":
    main()
