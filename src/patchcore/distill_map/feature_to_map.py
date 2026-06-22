"""Pattern A Student: 特徴マップ F_T → 異常マップ A_S（設計書 §6）。

PatchCore backbone の中間特徴（target_embed_dimension に集約された
``[C, H', W']`` 格子）を入力に、最近傍探索を置き換える decoder で異常マップを
予測する。画像 backbone を持たないため非常に軽量。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from patchcore.distill_map import pooling


class _UpBlock(nn.Module):
    """x2 アップサンプル + Conv3x3 + ReLU。"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.act(self.conv(x))


class FeatureToAnomalyMapStudent(nn.Module):
    """特徴格子 [N,C,H',W'] → 異常マップ [N,1,target_map_size,target_map_size]。"""

    def __init__(
        self,
        in_channels: int = 1024,
        reduce_channels: int = 256,
        decoder_channels: int = 128,
        num_up: int = 3,
        target_map_size: int = 224,
        output_activation: str = "none",
        score_pooling: str = "max",
        topk_ratio: float = 0.01,
    ):
        super().__init__()
        if output_activation not in ("none", "sigmoid"):
            raise ValueError(f"未知の output_activation: {output_activation}")
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, reduce_channels, 1),
            nn.ReLU(inplace=True),
        )
        ups = []
        ch = reduce_channels
        for _ in range(num_up):
            ups.append(_UpBlock(ch, decoder_channels))
            ch = decoder_channels
        self.decoder = nn.Sequential(*ups)
        self.head = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels, 1, 1),
        )
        self.target_map_size = target_map_size
        self.output_activation = output_activation
        self.score_pooling = score_pooling
        self.topk_ratio = topk_ratio

    def forward(self, feat):
        x = self.reduce(feat.float())
        x = self.decoder(x)
        amap = self.head(x)
        amap = F.interpolate(
            amap, size=(self.target_map_size, self.target_map_size),
            mode="bilinear", align_corners=False,
        )
        if self.output_activation == "sigmoid":
            amap = torch.sigmoid(amap)
        return amap

    def score(self, amap):
        return pooling.score_from_map(
            amap, pooling=self.score_pooling, topk_ratio=self.topk_ratio
        )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: nn.Module) -> float:
    n_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return n_bytes / (1024 ** 2)


def build_student(**kwargs) -> FeatureToAnomalyMapStudent:
    return FeatureToAnomalyMapStudent(**kwargs)
