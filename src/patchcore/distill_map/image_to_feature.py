"""蒸留方法③ 前段 Student: 画像 → Teacher 特徴マップ F_T を回帰する。

ImageNet 事前学習の重い backbone（WideResNet50 embed）を、学習可能な軽量
ResNet18-FPN で置き換えることを狙う。出力は Teacher の中間特徴格子
``[N, out_channels, H', W']``（既定 1024×28×28）に厳密に一致させ、蒸留方法②で
学習済みの後段 ``FeatureToAnomalyMapStudent`` を凍結してそのまま重ねられるようにする。

損失は F_T への特徴回帰（純 MSE 等）を想定。異常マップ／スコア損失は用いない。
既存 :mod:`patchcore.distill_map.student_fpn` の ``ResNet18Features`` / ``FPN`` を流用する。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from patchcore.distill_map.student_fpn import FPN, ResNet18Features


class FeatureHead(nn.Module):
    """Conv3x3-ReLU-Conv3x3-ReLU-Conv1x1 → out_channels の特徴グリッド。"""

    def __init__(self, in_channels: int = 128, mid_channels: int = 256,
                 out_channels: int = 1024):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 1),
        )

    def forward(self, x):
        return self.body(x)


class ImageToFeatureStudent(nn.Module):
    """画像 → Teacher 特徴グリッド ``[N, out_channels, H', W']`` を予測する前段。

    forward は F_T と同解像度・同チャンネルの特徴を返す。``output_activation="relu"``
    を選ぶと、post-ReLU で非負の Teacher 特徴に合わせて出力も非負に制約できる。
    """

    def __init__(
        self,
        out_channels: int = 1024,
        target_feat_size=28,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        fpn_out_channels: int = 128,
        decoder_channels: int = 256,
        output_activation: str = "none",
    ):
        super().__init__()
        if output_activation not in ("none", "relu"):
            raise ValueError(f"未知の output_activation: {output_activation}")
        self.backbone = ResNet18Features(pretrained=pretrained, freeze=freeze_backbone)
        self.fpn = FPN(ResNet18Features.OUT_CHANNELS, fpn_out_channels)
        self.head = FeatureHead(fpn_out_channels, decoder_channels, out_channels)
        self.out_channels = out_channels
        self.target_feat_size = target_feat_size
        self.output_activation = output_activation

    def _target_hw(self):
        s = self.target_feat_size
        return (s, s) if isinstance(s, int) else tuple(s)

    def forward(self, x):
        feats = self.backbone(x)
        pyramid = self.fpn(feats)
        y = self.head(pyramid[0])  # 最細レベル（top-down 集約済み）
        size = self._target_hw()
        if tuple(y.shape[-2:]) != size:
            y = F.interpolate(y, size=size, mode="bilinear", align_corners=False)
        if self.output_activation == "relu":
            y = F.relu(y)
        return y


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: nn.Module) -> float:
    n_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return n_bytes / (1024 ** 2)


def build_student(**kwargs) -> ImageToFeatureStudent:
    """config（dict / argparse 由来）から前段 Student を構築する薄いファクトリ。"""
    return ImageToFeatureStudent(**kwargs)
