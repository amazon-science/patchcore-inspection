"""蒸留 Loss（設計書 §10）。

L = lambda_map * L_map(A_S, A_T) + lambda_score * L_score(s_S, s_T)

- L_map: MSE または SmoothL1。異常領域が小さい場合に備え、Teacher map に基づく
  重み付き map loss（w_i = 1 + alpha * normalize(A_T_i)）を選択可能。
- L_score: MSE。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

MAP_LOSS_TYPES = ("mse", "smooth_l1")


def _elementwise_map_loss(a_s: torch.Tensor, a_t: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "mse":
        return (a_s - a_t) ** 2
    if kind == "smooth_l1":
        return F.smooth_l1_loss(a_s, a_t, reduction="none")
    raise ValueError(f"未知の map_loss: {kind}（候補: {MAP_LOSS_TYPES}）")


def _per_sample_minmax(a_t: torch.Tensor) -> torch.Tensor:
    """各サンプルの map を [0,1] に min-max 正規化（重み計算用）。"""
    flat = a_t.reshape(a_t.shape[0], -1)
    lo = flat.min(dim=1, keepdim=True).values
    hi = flat.max(dim=1, keepdim=True).values
    norm = (flat - lo) / (hi - lo + 1e-8)
    return norm.reshape_as(a_t)


def map_loss(
    a_s: torch.Tensor,
    a_t: torch.Tensor,
    kind: str = "smooth_l1",
    weighted: bool = False,
    alpha: float = 5.0,
) -> torch.Tensor:
    """異常マップ loss。a_s, a_t は同形状（[N,1,H,W] 等）。"""
    elem = _elementwise_map_loss(a_s, a_t, kind)
    if weighted:
        w = 1.0 + alpha * _per_sample_minmax(a_t)
        return (w * elem).mean()
    return elem.mean()


def score_loss(s_s: torch.Tensor, s_t: torch.Tensor) -> torch.Tensor:
    """画像スコア loss（MSE）。s_s, s_t は [N]。"""
    return F.mse_loss(s_s, s_t)


class DistillLoss(nn.Module):
    """map loss と score loss の重み付き和（設計書 §10.1）。"""

    def __init__(
        self,
        map_loss_kind: str = "smooth_l1",
        lambda_map: float = 1.0,
        lambda_score: float = 0.1,
        use_weighted_map_loss: bool = True,
        anomaly_weight_alpha: float = 5.0,
    ):
        super().__init__()
        self.map_loss_kind = map_loss_kind
        self.lambda_map = lambda_map
        self.lambda_score = lambda_score
        self.use_weighted_map_loss = use_weighted_map_loss
        self.anomaly_weight_alpha = anomaly_weight_alpha

    def forward(self, a_s, a_t, s_s, s_t):
        l_map = map_loss(
            a_s,
            a_t,
            kind=self.map_loss_kind,
            weighted=self.use_weighted_map_loss,
            alpha=self.anomaly_weight_alpha,
        )
        l_score = score_loss(s_s, s_t)
        total = self.lambda_map * l_map + self.lambda_score * l_score
        return total, {"loss": total.detach(), "map": l_map.detach(), "score": l_score.detach()}
