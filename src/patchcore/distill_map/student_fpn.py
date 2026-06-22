"""Pattern B Student: ResNet18 + FPN + Decoder で画像→異常マップを予測する。

設計書 §7/§8 に対応。ResNet18(ImageNet pretrained) の C2-C5 を FPN で統合し、
prediction head で 1ch の異常マップ A_S を出力する。出力は target_map_size に
bilinear 補間して Teacher map A_T と解像度を揃える（既定: target=teacher=224）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

from patchcore.distill_map import pooling


class ResNet18Features(nn.Module):
    """ResNet18 から C2,C3,C4,C5 を取り出す特徴抽出器。"""

    OUT_CHANNELS = (64, 128, 256, 512)  # C2, C3, C4, C5

    def __init__(self, pretrained: bool = True, freeze: bool = False):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=weights)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1  # C2  (64, H/4)
        self.layer2 = net.layer2  # C3  (128, H/8)
        self.layer3 = net.layer3  # C4  (256, H/16)
        self.layer4 = net.layer4  # C5  (512, H/32)
        self.frozen = freeze
        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:  # freeze 時は BN 統計も固定
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def forward(self, x):
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c2, c3, c4, c5]


class FPN(nn.Module):
    """Top-down FPN。各 C2-C5 を out_channels に揃えて統合する。"""

    def __init__(self, in_channels=(64, 128, 256, 512), out_channels: int = 128):
        super().__init__()
        self.lateral = nn.ModuleList(
            [nn.Conv2d(c, out_channels, 1) for c in in_channels]
        )
        self.smooth = nn.ModuleList(
            [nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels]
        )

    def forward(self, feats):
        laterals = [lat(f) for lat, f in zip(self.lateral, feats)]
        # top-down: 粗い層から細かい層へ加算
        for i in range(len(laterals) - 1, 0, -1):
            up = F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest"
            )
            laterals[i - 1] = laterals[i - 1] + up
        return [smooth(l) for smooth, l in zip(self.smooth, laterals)]


class PredictionHead(nn.Module):
    """Conv3x3-ReLU-Conv3x3-ReLU-Conv1x1 → 1ch 異常マップ（設計書 §8.3）。"""

    def __init__(self, in_channels: int = 128, mid_channels: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, 1),
        )

    def forward(self, x):
        return self.body(x)


class ImageToMapStudent(nn.Module):
    """画像→異常マップ Student（Pattern B）。

    forward は異常マップ A_S を ``[N,1,target_map_size,target_map_size]`` で返す。
    画像スコア s_S は :meth:`score` で pooling して得る。
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        fpn_out_channels: int = 128,
        decoder_channels: int = 128,
        target_map_size: int = 224,
        output_activation: str = "none",
        score_pooling: str = "max",
        topk_ratio: float = 0.01,
    ):
        super().__init__()
        if output_activation not in ("none", "sigmoid"):
            raise ValueError(f"未知の output_activation: {output_activation}")
        self.backbone = ResNet18Features(pretrained=pretrained, freeze=freeze_backbone)
        self.fpn = FPN(ResNet18Features.OUT_CHANNELS, fpn_out_channels)
        self.head = PredictionHead(fpn_out_channels, decoder_channels)
        self.target_map_size = target_map_size
        self.output_activation = output_activation
        self.score_pooling = score_pooling
        self.topk_ratio = topk_ratio

    def forward(self, x):
        feats = self.backbone(x)
        pyramid = self.fpn(feats)
        amap = self.head(pyramid[0])  # 最細 P2 を使用
        amap = F.interpolate(
            amap,
            size=(self.target_map_size, self.target_map_size),
            mode="bilinear",
            align_corners=False,
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


def build_student(**kwargs) -> ImageToMapStudent:
    """config（dict / argparse 由来）から Student を構築する薄いファクトリ。"""
    return ImageToMapStudent(**kwargs)
