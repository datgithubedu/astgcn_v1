# models/astgcn.py
"""
ASTGCN model cho bài toán Traffic Speed / Congestion Prediction.

Được tách ra từ notebook astgcn-merge-3, chỉnh lại cho phù hợp với
pipeline CanonicalTrafficData + ASTGCNAdapter:

- Input cho model:
    Xh, Xd, Xw: (batch_size, N_nodes, T_in)
- A_norm: adjacency chuẩn hoá (N, N)

- Output:
    Y_hat: (batch_size, N_nodes, T_out)

Model gồm 3 nhánh:
    - Nhánh "recent" (Xh)
    - Nhánh "daily"  (Xd)
    - Nhánh "weekly" (Xw)
Mỗi nhánh gồm nhiều STBlock (Spatial-Temporal block) với attention.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Utility: Scaled Laplacian & Chebyshev polynomials
# ---------------------------------------------------------------------


def scaled_laplacian(A: torch.Tensor) -> torch.Tensor:
    """
    Tính scaled Laplacian L_tilde từ adjacency A.

    A: (N, N) torch.Tensor, đã chuẩn hoá đối xứng (hoặc raw adjacency cũng được).
    Trả về:
        L_tilde: (N, N)
    """
    if not isinstance(A, torch.Tensor):
        A = torch.tensor(A, dtype=torch.float32)

    A = A.float()
    device = A.device
    N = A.shape[0]

    # L = D - A
    D = torch.diag(torch.sum(A, dim=1))
    L = D - A

    # Ước lượng lambda_max bằng eigvals (nếu lỗi thì fallback = 2.0)
    try:
        eigs = torch.linalg.eigvals(L)
        lambda_max = torch.max(torch.real(eigs)).item()
    except Exception:
        lambda_max = 2.0

    if lambda_max <= 0:
        lambda_max = 2.0

    # L_tilde = 2 L / lambda_max - I
    L_tilde = (2.0 / lambda_max) * L - torch.eye(N, device=device)
    return L_tilde


def chebyshev_polynomials(L_tilde: torch.Tensor, K: int) -> List[torch.Tensor]:
    """
    Tạo danh sách Chebyshev polynomials T_k(L_tilde) cho k=0..K-1.

    Trả về:
        [T0, T1, ..., T_{K-1}] mỗi phần tử shape (N, N)
    """
    N = L_tilde.shape[0]
    T_k: List[torch.Tensor] = []

    T0 = torch.eye(N, dtype=L_tilde.dtype, device=L_tilde.device)
    T_k.append(T0)

    if K == 1:
        return T_k

    T1 = L_tilde
    T_k.append(T1)

    for _ in range(2, K):
        Tkp = 2 * L_tilde @ T_k[-1] - T_k[-2]
        T_k.append(Tkp)

    return T_k


# ---------------------------------------------------------------------
# Spatial Attention
# ---------------------------------------------------------------------


class SpatialAttention(nn.Module):
    """
    Spatial Attention như trong paper ASTGCN (phiên bản đơn giản hoá).

    Input:
        x: (B, N, C, T)

    Output:
        S: (B, N, N) attention theo không gian (nodes)
    """

    def __init__(self, in_channels: int, num_nodes: int, time_len: int):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(in_channels, in_channels))
        self.W2 = nn.Parameter(torch.randn(in_channels, in_channels))
        self.W3 = nn.Parameter(torch.randn(in_channels, in_channels))
        self.Vs = nn.Parameter(torch.randn(num_nodes, num_nodes))
        self.bs = nn.Parameter(torch.zeros(1))

        self.num_nodes = num_nodes
        self.time_len = time_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C, T)
        B, N, C, T = x.shape

        # Tổng theo thời gian → (B, N, C)
        X_sum_t = torch.sum(x, dim=-1)

        f1 = X_sum_t @ self.W1  # (B, N, C)
        f2 = X_sum_t @ self.W2  # (B, N, C)

        # Pairwise interaction giữa nodes
        tmp = f1 @ self.W3  # (B, N, C)

        # scores[i, j] = <tmp_i, f2_j>
        scores = torch.einsum("bnc,bmc->bnm", tmp, f2)  # (B, N, N)
        scores = scores + self.bs

        # Kết hợp với Vs (global)
        Vs = self.Vs.unsqueeze(0).expand(B, -1, -1)  # (B, N, N)
        S = torch.tanh(scores) * Vs
        S = F.softmax(S, dim=2)  # attention trên chiều j (neighbors)

        return S  # (B, N, N)


# ---------------------------------------------------------------------
# Temporal Attention
# ---------------------------------------------------------------------


class TemporalAttention(nn.Module):
    """
    Temporal Attention đơn giản hoá.

    Input:
        x: (B, N, C, T)

    Output:
        E: (B, T, T) attention theo thời gian
    """

    def __init__(self, in_channels: int, num_nodes: int, time_len: int):
        super().__init__()
        self.U1 = nn.Parameter(torch.randn(in_channels, in_channels))
        self.U2 = nn.Parameter(torch.randn(in_channels, in_channels))
        self.U3 = nn.Parameter(torch.randn(in_channels, in_channels))
        self.Ve = nn.Parameter(torch.randn(time_len, time_len))
        self.be = nn.Parameter(torch.zeros(1))

        self.time_len = time_len
        self.num_nodes = num_nodes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C, T)
        B, N, C, T = x.shape

        # Tổng theo nodes → (B, C, T)
        X_sum_n = torch.sum(x, dim=1)  # (B, C, T)

        # (B, T, C)
        X_t = X_sum_n.permute(0, 2, 1)

        t1 = X_t @ self.U1  # (B, T, C)
        t2 = X_t @ self.U2  # (B, T, C)

        # scores giữa các time step:
        scores = torch.einsum("btc,buc->btu", t1, t2)  # (B, T, T)
        scores = scores + self.be

        Ve = self.Ve.unsqueeze(0).expand(B, -1, -1)  # (B, T, T)
        E = torch.tanh(scores) * Ve
        E = F.softmax(E, dim=2)  # attention trên chiều thời gian

        return E  # (B, T, T)


# ---------------------------------------------------------------------
# Chebyshev Graph Convolution
# ---------------------------------------------------------------------


class ChebGraphConv(nn.Module):
    """
    Chebyshev Graph Convolution:
        x_out = sum_{k=0}^{K-1} T_k(L_tilde) x Theta_k
    """

    def __init__(
        self,
        K: int,
        in_channels: int,
        out_channels: int,
        num_nodes: int,
    ):
        super().__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels

        # (K, C_in, C_out)
        self.Thetas = nn.Parameter(torch.randn(K, in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(
        self,
        x: torch.Tensor,
        cheb_polynomials: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        x: (B, N, C_in) features tại 1 time step
        cheb_polynomials: list T_k(L_tilde) (N, N), k = 0..K-1

        return:
            out: (B, N, C_out)
        """
        B, N, C = x.shape
        assert C == self.in_channels

        out = torch.zeros(B, N, self.out_channels, device=x.device)

        for k in range(self.K):
            T_k = cheb_polynomials[k]  # (N, N)

            # support @ x: (N, N) @ (B, N, C) → (B, N, C)
            support_x = torch.einsum("nm,bmc->bnc", T_k, x)

            # (B, N, C_in) @ (C_in, C_out) → (B, N, C_out)
            out += torch.einsum("bnc,co->bno", support_x, self.Thetas[k])

        out = out + self.bias
        return out  # (B, N, C_out)


# ---------------------------------------------------------------------
# ST-Block (Spatial–Temporal Block)
# ---------------------------------------------------------------------


class STBlock(nn.Module):
    """
    Một STBlock gồm:
        - Temporal Attention
        - Spatial Attention
        - Chebyshev GraphConv (cho từng time step)
        - Temporal convolution (1x3) trên time axis
        - Residual connection + LayerNorm
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        num_nodes: int,
        t_kernel: int,
    ):
        super().__init__()
        self.spatial_att = SpatialAttention(in_channels, num_nodes, time_len=t_kernel)
        self.temporal_att = TemporalAttention(in_channels, num_nodes, time_len=t_kernel)

        self.cheb_K = K
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.gconv = ChebGraphConv(K, in_channels, out_channels, num_nodes)

        # Temporal conv: (B, outC, N, T) → conv 1x3 theo T
        self.temp_conv = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(1, 3),
            padding=(0, 1),
        )

        # Residual 1x1 conv: (B, C_in, N, T) → (B, outC, N, T)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

        # LayerNorm trên (N, C) cho mỗi time step
        self.layer_norm = nn.LayerNorm([num_nodes, out_channels])

    def forward(
        self,
        x: torch.Tensor,
        cheb_polynomials: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        x: (B, N, C, T)
        cheb_polynomials: list T_k(L_tilde)
        return:
            out: (B, N, out_channels, T)
        """
        B, N, C, T = x.shape

        # ---- Temporal Attention ----
        E = self.temporal_att(x)  # (B, T, T)

        # reshape để nhân attention trên T
        x_perm = x.permute(0, 2, 1, 3).reshape(B, C * N, T)  # (B, C*N, T)
        x_t_att = x_perm @ E  # (B, C*N, T)
        x_t_att = x_t_att.reshape(B, C, N, T).permute(0, 2, 1, 3)  # (B, N, C, T)

        # ---- Spatial Attention ----
        S = self.spatial_att(x_t_att)  # (B, N, N)

        # ---- Graph conv cho từng time step ----
        g_out_ts = []
        for t in range(T):
            x_t = x_t_att[..., t]  # (B, N, C)

            # Áp spatial att: x_t_s = S @ x_t
            x_t_s = torch.einsum("bnm,bmc->bnc", S, x_t)  # (B, N, C)

            g_t = self.gconv(x_t_s, cheb_polynomials)  # (B, N, outC)
            g_out_ts.append(g_t.unsqueeze(-1))

        g_out = torch.cat(g_out_ts, dim=-1)  # (B, N, outC, T)

        # ---- Temporal conv ----
        g_out_t = g_out.permute(0, 2, 1, 3)  # (B, outC, N, T)
        g_conv = self.temp_conv(g_out_t)  # (B, outC, N, T)

        # ---- Residual ----
        res = self.residual_conv(x.permute(0, 2, 1, 3))  # (B, outC, N, T)
        out = F.relu(g_conv + res)
        out = out.permute(0, 2, 1, 3)  # (B, N, outC, T)

        # ---- LayerNorm trên (N, C) cho từng T ----
        B_, N_, C_, T_ = out.shape
        out = out.permute(0, 3, 1, 2)  # (B, T, N, C)
        out = self.layer_norm(out)
        out = out.permute(0, 2, 3, 1)  # (B, N, C, T)

        return out


# ---------------------------------------------------------------------
# ASTGCN Model (3-branch fusion)
# ---------------------------------------------------------------------


class ASTGCN(nn.Module):
    """
    ASTGCN với 3 nhánh:
        - Recent (Xh)
        - Daily  (Xd)
        - Weekly (Xw)

    Tham số chính:
        num_nodes: số node N
        in_channels: số feature / node (1 nếu chỉ dùng speed)
        out_timesteps: số bước thời gian cần forecast (horizon)
        K: Chebyshev order
        t_kernel: độ dài chuỗi input (T_in), dùng cho attention
        channels: số channel bên trong STBlock
        blocks: số STBlock chồng nhau cho mỗi nhánh
    """

    def __init__(
        self,
        num_nodes: int,
        in_channels: int = 1,
        out_timesteps: int = 12,
        K: int = 3,
        t_kernel: int = 12,
        channels: int = 64,
        blocks: int = 2,
    ):
        super().__init__()

        self.N = num_nodes
        self.in_channels = in_channels
        self.out_timesteps = out_timesteps
        self.K = K

        # 3 nhánh: recent, daily, weekly
        self.branch_h = nn.ModuleList(
            [
                STBlock(
                    in_channels if i == 0 else channels,
                    channels,
                    K,
                    num_nodes,
                    t_kernel,
                )
                for i in range(blocks)
            ]
        )
        self.branch_d = nn.ModuleList(
            [
                STBlock(
                    in_channels if i == 0 else channels,
                    channels,
                    K,
                    num_nodes,
                    t_kernel,
                )
                for i in range(blocks)
            ]
        )
        self.branch_w = nn.ModuleList(
            [
                STBlock(
                    in_channels if i == 0 else channels,
                    channels,
                    K,
                    num_nodes,
                    t_kernel,
                )
                for i in range(blocks)
            ]
        )

        # Map channels → out_timesteps (conv 1x1)
        self.final_conv_h = nn.Conv2d(channels, out_timesteps, kernel_size=(1, 1))
        self.final_conv_d = nn.Conv2d(channels, out_timesteps, kernel_size=(1, 1))
        self.final_conv_w = nn.Conv2d(channels, out_timesteps, kernel_size=(1, 1))

        # Fusion weights (learnable) cho từng node, từng bước thời gian
        self.weight_h = nn.Parameter(torch.ones(num_nodes, out_timesteps))
        self.weight_d = nn.Parameter(torch.ones(num_nodes, out_timesteps))
        self.weight_w = nn.Parameter(torch.ones(num_nodes, out_timesteps))

    # ----- helper cho từng nhánh -----
    def _forward_branch(
        self,
        x: torch.Tensor,
        cheb_polynomials: List[torch.Tensor],
        branch_blocks: nn.ModuleList,
        final_conv: nn.Conv2d,
    ) -> torch.Tensor:
        """
        x: (B, N, C, T)
        return:
            out: (B, N, out_timesteps)
        """
        out = x
        for blk in branch_blocks:
            out = blk(out, cheb_polynomials)  # (B, N, channels, T)

        # (B, channels, N, T)
        out_t = out.permute(0, 2, 1, 3)
        out_mapped = final_conv(out_t)  # (B, outT, N, T)

        # Lấy trung bình theo trục thời gian cuối cùng → (B, outT, N)
        out_mean = out_mapped.mean(dim=-1)
        out_mean = out_mean.permute(0, 2, 1)  # (B, N, outT)

        return out_mean

    # ----- forward tổng -----
    def forward(
        self,
        Xh: torch.Tensor,
        Xd: torch.Tensor,
        Xw: torch.Tensor,
        A_norm: torch.Tensor,
    ) -> torch.Tensor:
        """
        Xh, Xd, Xw: (B, N, T_in)
        A_norm: (N, N) adjacency đã chuẩn hoá

        Trả về:
            Y_hat: (B, N, out_timesteps)
        """
        device = Xh.device
        A_norm = A_norm.to(device)

        # Chebyshev supports
        L_tilde = scaled_laplacian(A_norm)
        cheb_polys = chebyshev_polynomials(L_tilde.to(device), self.K)

        # Thêm channel dimension: C = 1
        Xh_ = Xh.unsqueeze(2)  # (B, N, 1, T)
        Xd_ = Xd.unsqueeze(2)
        Xw_ = Xw.unsqueeze(2)

        out_h = self._forward_branch(
            Xh_, cheb_polys, self.branch_h, self.final_conv_h
        )  # (B, N, outT)
        out_d = self._forward_branch(
            Xd_, cheb_polys, self.branch_d, self.final_conv_d
        )
        out_w = self._forward_branch(
            Xw_, cheb_polys, self.branch_w, self.final_conv_w
        )

        # Fusion bằng learnable weights
        W_h = torch.sigmoid(self.weight_h).unsqueeze(0)  # (1, N, outT)
        W_d = torch.sigmoid(self.weight_d).unsqueeze(0)
        W_w = torch.sigmoid(self.weight_w).unsqueeze(0)

        Y_hat = W_h * out_h + W_d * out_d + W_w * out_w  # (B, N, outT)
        return Y_hat


# ---------------------------------------------------------------------
# (Optional) Simple training loop – dùng cho script training
# ---------------------------------------------------------------------


def train_model(
    model: ASTGCN,
    train_loader,
    val_loader,
    A_norm: torch.Tensor,
    device: str = "cuda",
    epochs: int = 30,
    lr: float = 1e-3,
):
    """
    Vòng lặp train đơn giản, MSELoss trên không gian chuẩn hoá.

    Gợi ý dùng trong training/train_astgcn.py:
        from models.astgcn import ASTGCN, train_model
    """
    device = torch.device(device)
    model = model.to(device)
    A_t = A_norm.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_val = float("inf")

    for ep in range(1, epochs + 1):
        # -------- Train --------
        model.train()
        train_loss = 0.0
        n_batch = 0

        for Xh, Xd, Xw, Y in train_loader:
            Xh = Xh.to(device)
            Xd = Xd.to(device)
            Xw = Xw.to(device)
            Y = Y.to(device)

            optimizer.zero_grad()
            Y_hat = model(Xh, Xd, Xw, A_t)  # (B, N, outT)

            loss = criterion(Y_hat, Y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_loss += loss.item()
            n_batch += 1

        train_loss /= max(1, n_batch)

        # -------- Validation --------
        model.eval()
        val_loss = 0.0
        n_batch = 0

        with torch.no_grad():
            for Xh, Xd, Xw, Y in val_loader:
                Xh = Xh.to(device)
                Xd = Xd.to(device)
                Xw = Xw.to(device)
                Y = Y.to(device)

                Y_hat = model(Xh, Xd, Xw, A_t)
                loss = criterion(Y_hat, Y)

                val_loss += loss.item()
                n_batch += 1

        val_loss /= max(1, n_batch)

        print(f"Epoch {ep:03d} | Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "astgcn_best.pth")
            print("  (saved best model)")

    return model
