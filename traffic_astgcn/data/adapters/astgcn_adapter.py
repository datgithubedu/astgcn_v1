import numpy as np
import pandas as pd
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.neighbors import NearestNeighbors
from scipy import sparse

from traffic_astgcn.data.canonical import CanonicalTrafficData
from sklearn.preprocessing import StandardScaler


# ------------------------
# Dataset cho PyTorch
# ------------------------
class ASTGCDataset(Dataset):
    def __init__(self, Xh, Xd, Xw, Y):
        self.Xh = Xh
        self.Xd = Xd
        self.Xw = Xw
        self.Y  = Y

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        return (
            torch.tensor(self.Xh[idx], dtype=torch.float32),
            torch.tensor(self.Xd[idx], dtype=torch.float32),
            torch.tensor(self.Xw[idx], dtype=torch.float32),
            torch.tensor(self.Y[idx],  dtype=torch.float32),
        )


# ------------------------
# Build adjacency
# ------------------------
def build_adjacency_from_nodes(nodes_df: pd.DataFrame, k=8):
    nodes_df = nodes_df.drop_duplicates("node_id").sort_values("node_id")

    coords_rad = np.radians(nodes_df[["lat", "lon"]].to_numpy())

    nbrs = NearestNeighbors(n_neighbors=k+1, metric="haversine").fit(coords_rad)
    distances, indices = nbrs.kneighbors(coords_rad)

    R = 6371000.0
    distances_m = distances * R
    sigma = distances_m[:, 1:].mean()

    rows, cols, data = [], [], []

    for i in range(len(nodes_df)):
        for j, d in zip(indices[i, 1:], distances_m[i, 1:]):
            w = np.exp(-(d**2) / (2 * sigma**2))
            rows.append(i)
            cols.append(j)
            data.append(w)

    A = sparse.coo_matrix((data, (rows, cols)), shape=(len(nodes_df), len(nodes_df)))
    A = A.maximum(A.T).toarray().astype(np.float32)
    return A


# ------------------------
# Adapter chính
# ------------------------
@dataclass
class ASTGCNAdapter:
    seq_len_h: int = 12
    seq_len_d: int = 12
    seq_len_w: int = 12
    horizon: int   = 12
    time_per_day: int = 48
    day_lag: int = 1
    week_lag: int = 7
    batch_size: int = 16

    # -------------------------------------------------
    def _edge_to_node_timeseries(self, canonical: CanonicalTrafficData):
        df = canonical.edge_df.copy()

        # map node → index
        nodes = sorted(df["node_a_id"].unique())
        node2idx = {n: i for i, n in enumerate(nodes)}
        df["node_idx"] = df["node_a_id"].map(node2idx)

        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Pivot thành (time, nodes)
        pv = df.pivot_table(index="timestamp", columns="node_idx", values="speed_kmh")
        pv = pv.sort_index().interpolate(limit_direction="both").ffill().bfill()

        # 🌟 Sửa quan trọng: scale per-node (không dùng canonical.speed_scaler)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled = scaler.fit_transform(pv.values)   # (T, N_nodes)

        # lưu scaler để inference sau này
        self.node_scaler = scaler

        return scaled.astype(np.float32), pv.index.to_numpy(), np.array(nodes)


    # -------------------------------------------------
    def _build_sequences(self, pv_scaled: np.ndarray):
        T_total, N = pv_scaled.shape
        Th, Td, Tw, horizon = self.seq_len_h, self.seq_len_d, self.seq_len_w, self.horizon
        time_per_day = self.time_per_day

        Xh_list, Xd_list, Xw_list, Y_list = [], [], [], []

        t_start = max(
            Tw + self.week_lag * time_per_day,
            Td + self.day_lag * time_per_day,
            Th
        )

        for t in range(t_start, T_total - horizon):
            Xh = pv_scaled[t - Th: t]
            Xd = pv_scaled[t - self.day_lag*time_per_day - Td: t - self.day_lag*time_per_day]
            Xw = pv_scaled[t - self.week_lag*time_per_day - Tw: t - self.week_lag*time_per_day]
            Y  = pv_scaled[t: t + horizon]

            Xh_list.append(Xh)
            Xd_list.append(Xd)
            Xw_list.append(Xw)
            Y_list.append(Y)

        Xh = np.transpose(np.stack(Xh_list), (0, 2, 1))
        Xd = np.transpose(np.stack(Xd_list), (0, 2, 1))
        Xw = np.transpose(np.stack(Xw_list), (0, 2, 1))
        Y  = np.transpose(np.stack(Y_list),  (0, 2, 1))

        return Xh, Xd, Xw, Y

    # -------------------------------------------------
    def __call__(self, canonical: CanonicalTrafficData, split: str, nodes_df: pd.DataFrame):
        pv_scaled, timestamps, node_ids = self._edge_to_node_timeseries(canonical)
        Xh, Xd, Xw, Y = self._build_sequences(pv_scaled)

        n = Xh.shape[0]
        n_train = int(n * 0.7)
        n_val = int(n * 0.1)

        if split == "train":
            sl = slice(0, n_train)
        elif split == "val":
            sl = slice(n_train, n_train + n_val)
        else:
            sl = slice(n_train + n_val, n)

        dataset = ASTGCDataset(Xh[sl], Xd[sl], Xw[sl], Y[sl])
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=(split == "train"))

        if nodes_df is not None:
            # Lấy node_id theo thứ tự đã pivot
            used_nodes = pd.DataFrame({"node_id": node_ids})

            # Join để lấy lat/lon đúng thứ tự
            merged = used_nodes.merge(nodes_df, on="node_id", how="left")

            # Nếu thiếu lat/lon → lỗi
            if merged[["lat", "lon"]].isna().any().any():
                raise ValueError("Missing lat/lon for some pivot nodes. Check nodes.json consistency.")

            A = build_adjacency_from_nodes(merged)
        else:
            raise ValueError("nodes_df must not be None")

        return loader, A
