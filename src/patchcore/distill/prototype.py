"""M4: 学習可能プロトタイプ（soft-min）スコアラ。

教師の異常スコア（パッチ特徴量 → 最近傍 L2 距離）を、K 個の学習可能な中心
`c_k` への soft-min 距離で近似する：

    score(x) ≈ -T · logsumexp_k( -‖x - c_k‖² / T )   （T→0 で min‖x-c_k‖²）

利点：
  - soft-min は最近傍距離の構造に一致し、単調＝max プーリング後のランキング目的と整合。
  - メモリは中心 K×D のみで一定（継続学習でラウンドを重ねても増えない）。
  - 中心をコアセットで初期化でき、微分可能な「学習コアセット」として M2 と連続。
"""
import numpy as np
import torch
import torch.nn as nn

from patchcore.distill import Scorer
from patchcore.distill.student import _EPS
from patchcore.distill.train_student import _ranking_loss


class PrototypeNet(nn.Module):
    """K 個の学習可能中心への soft-min 二乗距離を返す。"""

    def __init__(self, init_centers: np.ndarray, temp: float = 1.0):
        super().__init__()
        self.centers = nn.Parameter(torch.from_numpy(np.ascontiguousarray(init_centers, dtype=np.float32)))
        self.temp = float(temp)

    def forward(self, x):
        # x: [B, D] → 二乗距離 [B, K] → soft-min（T→0 で min）
        d2 = torch.cdist(x, self.centers) ** 2
        return -self.temp * torch.logsumexp(-d2 / self.temp, dim=1)


def train_prototype(
    bank: np.ndarray,
    label_index,
    init_centers: np.ndarray,
    device,
    temp: float = 1.0,
    n_steps: int = 2000,
    batch: int = 4096,
    lr: float = 1e-2,
    sigma_max: float = 2.0,
    far_frac: float = 0.1,
    seed: int = 0,
    loss: str = "huber",
    log_every: int = 300,
) -> PrototypeNet:
    """教師バンクの距離写像に合わせて中心を微調整する（学習データ生成は生徒と同一）。"""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    n_total, dim = bank.shape
    feat_std = bank.std(axis=0, keepdims=True).astype(np.float32) + _EPS
    feat_mean = bank.mean(axis=0, keepdims=True).astype(np.float32)

    model = PrototypeNet(init_centers, temp=temp).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    huber = nn.SmoothL1Loss()

    model.train()
    for step in range(n_steps):
        idx = rng.integers(0, n_total, size=batch)
        sigma = rng.random((batch, 1)).astype(np.float32) * sigma_max
        noise = rng.standard_normal((batch, dim)).astype(np.float32) * feat_std * sigma
        queries = bank[idx] + noise

        n_far = int(batch * far_frac)
        if n_far > 0:
            far = feat_mean + rng.standard_normal((n_far, dim)).astype(
                np.float32
            ) * feat_std * (sigma_max * 2.0)
            queries[:n_far] = far

        queries = np.ascontiguousarray(queries, dtype=np.float32)
        labels, _ = label_index.search(queries, 1)  # squared L2（教師と同一指標）

        x = torch.from_numpy(queries).to(device)
        pred = model(x)  # 予測 二乗距離
        if loss == "ranking":
            target = torch.from_numpy(labels[:, 0].astype(np.float32)).to(device)
            step_loss = _ranking_loss(pred, target)
        else:
            target = torch.from_numpy(np.log(labels[:, 0] + _EPS).astype(np.float32)).to(device)
            step_loss = huber(torch.log(pred.clamp_min(0) + _EPS), target)

        optimizer.zero_grad()
        step_loss.backward()
        optimizer.step()

        if log_every and (step % log_every == 0 or step == n_steps - 1):
            print(f"  [proto] step {step+1}/{n_steps}  loss={step_loss.item():.4f}")

    model.eval()
    return model


class PrototypeScorer(Scorer):
    """学習済みプロトタイプ中心で soft-min 距離を算出する。"""

    def __init__(self, model: PrototypeNet, device, infer_batch: int = 65536, name=None):
        self.k = model.centers.shape[0]
        self.name = name or f"M4_proto_{self.k}"
        self.model = model.to(device).eval()
        self.device = device
        self.infer_batch = infer_batch
        n_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        self._size_mb = n_bytes / (1024 * 1024)

    def score_patches(self, features):
        x = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
        preds = []
        with torch.no_grad():
            for start in range(0, len(x), self.infer_batch):
                chunk = x[start : start + self.infer_batch].to(self.device)
                preds.append(self.model(chunk).clamp_min(0).cpu())
        return torch.cat(preds).numpy()

    def memory_mb(self):
        return self._size_mb
