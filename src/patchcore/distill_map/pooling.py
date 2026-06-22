"""異常マップ A_S から画像レベル異常スコア s_S を算出する pooling。

設計書 §11 / §5.1 に対応。``max`` / ``topk_mean`` / ``mean`` を実装する。
入力 amap は ``[N, H, W]`` または ``[N, 1, H, W]`` を受け付け、出力は ``[N]``。
"""
import torch

POOLING_TYPES = ("max", "topk_mean", "mean")


def _flatten_maps(amap: torch.Tensor) -> torch.Tensor:
    """[N,H,W] / [N,1,H,W] → [N, H*W] に整形する。"""
    if amap.dim() == 4:
        if amap.shape[1] != 1:
            raise ValueError(f"channel 次元は 1 を想定（got {amap.shape[1]}）")
        amap = amap[:, 0]
    if amap.dim() != 3:
        raise ValueError(f"amap は [N,H,W] か [N,1,H,W] を想定（got {tuple(amap.shape)}）")
    return amap.reshape(amap.shape[0], -1)


def score_from_map(
    amap: torch.Tensor, pooling: str = "max", topk_ratio: float = 0.01
) -> torch.Tensor:
    """異常マップから画像スコア [N] を算出する。

    Args:
        amap: 異常マップ ``[N,H,W]`` または ``[N,1,H,W]``。
        pooling: ``max`` / ``topk_mean`` / ``mean``。
        topk_ratio: ``topk_mean`` のとき上位何割の画素を平均するか（0<r<=1）。
    """
    flat = _flatten_maps(amap)
    if pooling == "max":
        return flat.max(dim=1).values
    if pooling == "mean":
        return flat.mean(dim=1)
    if pooling == "topk_mean":
        n = flat.shape[1]
        k = max(1, int(round(n * topk_ratio)))
        topk = flat.topk(k, dim=1).values
        return topk.mean(dim=1)
    raise ValueError(f"未知の pooling: {pooling}（候補: {POOLING_TYPES}）")
