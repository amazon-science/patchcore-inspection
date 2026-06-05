"""M3: 生徒ネットへの蒸留。

教師バンク（faiss）が与える「パッチ特徴量 → 最近傍 L2 距離」写像を、定数サイズの
MLP に回帰学習させる。推論時は faiss バンク不要で、メモリは MLP 重みのみ（継続学習で
ラウンドを重ねても増えない）。
"""
import numpy as np
import torch
import torch.nn as nn

from patchcore.distill import Scorer

_EPS = 1e-6


class StudentMLP(nn.Module):
    """パッチ特徴量 [*, in_dim] → log(距離+eps) を回帰する MLP。"""

    def __init__(self, in_dim: int = 1024, hidden=(512, 256)):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def model_size_mb(model: nn.Module) -> float:
    n_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    n_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    return n_bytes / (1024 * 1024)


class StudentScorer(Scorer):
    """学習済み生徒 MLP でパッチ毎スコア（= 予測距離）を算出する。"""

    def __init__(self, model: StudentMLP, device, infer_batch: int = 65536):
        self.name = "M3_student"
        self.model = model.to(device).eval()
        self.device = device
        self.infer_batch = infer_batch
        self._size_mb = model_size_mb(model)

    def score_patches(self, features):
        x = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
        preds = []
        with torch.no_grad():
            for start in range(0, len(x), self.infer_batch):
                chunk = x[start : start + self.infer_batch].to(self.device)
                # 多様体から大きく外れた点での expm1 オーバーフロー(inf/nan)を防ぐ
                log_dist = self.model(chunk).clamp_max(40.0)
                preds.append(torch.expm1(log_dist).clamp_min(0).cpu())
        return torch.cat(preds).numpy()

    def memory_mb(self):
        return self._size_mb
