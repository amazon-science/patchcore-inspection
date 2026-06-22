"""precompute した Teacher target の読み込み・突合・正規化ユーティリティ。

Student 学習・評価から共通利用する。画像は既存 MVTecDataset の前処理を
そのまま使うことで、Teacher target 生成時と Student 入力の前処理を一致させる
（設計書 §21.1）。
"""
import glob
import os
import re

import PIL.Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_eval_transform(resize: int = 256, imagesize: int = 224):
    """MVTecDataset と同一の画像前処理（設計書 §21.1 で Teacher と一致必須）。"""
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(imagesize),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class ImageFolderDataset(Dataset):
    """フォルダ内の画像を MVTec 同一前処理で読む（precompute の --image_dir 用）。"""

    def __init__(self, folder: str, resize: int = 256, imagesize: int = 224, is_anomaly: int = 1):
        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        self.paths = sorted(p for e in exts for p in glob.glob(os.path.join(folder, e)))
        if not self.paths:
            raise RuntimeError(f"画像が見つかりません: {folder}")
        self.transform = build_eval_transform(resize, imagesize)
        self.is_anomaly = is_anomaly

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = self.transform(PIL.Image.open(path).convert("RGB"))
        return {
            "image": image,
            "mask": torch.zeros([1, image.shape[1], image.shape[2]]),
            "is_anomaly": self.is_anomaly,
            "image_name": os.path.basename(path),
            "image_path": path,
        }


def sample_id(image_name: str) -> str:
    """image_name（'classname/split/type/000.png'）をファイル名 ID に変換。"""
    return re.sub(r"[^0-9A-Za-z_.-]", "__", image_name)


def _image_name_from_path(image_path: str) -> str:
    return "/".join(image_path.split("/")[-4:])


class TeacherTargetDataset(Dataset):
    """MVTecDataset の各画像に、precompute 済み Teacher target を対応付ける。

    返り値: ``{"image", "teacher_map" [1,H,W], "teacher_score", "label", "sample_id"}``。
    対応する ``.pt`` が存在する画像のみを対象にする（スモークの部分 precompute に対応）。
    """

    def __init__(self, base_dataset, targets_split_dir: str, load_feature: bool = False):
        self.base = base_dataset
        self.dir = targets_split_dir
        self.load_feature = load_feature
        avail = {
            f[:-3] for f in os.listdir(targets_split_dir) if f.endswith(".pt")
        }
        self.items = []
        for i, tup in enumerate(base_dataset.data_to_iterate):
            image_path = tup[2]
            sid = sample_id(_image_name_from_path(image_path))
            if sid in avail:
                self.items.append((i, sid))
        if not self.items:
            raise RuntimeError(
                f"対応する Teacher target が見つかりません: {targets_split_dir}"
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, j):
        base_idx, sid = self.items[j]
        item = self.base[base_idx]
        rec = torch.load(os.path.join(self.dir, sid + ".pt"), weights_only=False)
        amap = rec["teacher_anomaly_map"]
        if amap.dim() == 2:
            amap = amap.unsqueeze(0)  # [1,H,W]
        out = {
            "image": item["image"],
            "teacher_map": amap.float(),
            "teacher_score": torch.tensor(
                float(rec["teacher_anomaly_score"]), dtype=torch.float32
            ),
            "label": int(rec["label"]),
            "sample_id": sid,
            "image_path": rec.get("image_path", sid),
        }
        if self.load_feature:
            if "teacher_feature_map" not in rec:
                raise KeyError(
                    f"teacher_feature_map がありません（precompute を --save_feature で実行）: {sid}"
                )
            out["teacher_feature"] = rec["teacher_feature_map"].float()
        return out


class DirectImageTargetDataset(Dataset):
    """precompute 済み split を、画像を image_path から直接読んで対応付ける。

    疑似異常など MVTecDataset の index に対応しない画像群（precompute の
    --image_dir で生成）を Student 学習に使うためのデータセット。返り値は
    :class:`TeacherTargetDataset` と同一キー。
    """

    def __init__(self, targets_split_dir: str, resize: int = 256, imagesize: int = 224,
                 load_feature: bool = False):
        self.dir = targets_split_dir
        self.files = sorted(glob.glob(os.path.join(targets_split_dir, "*.pt")))
        if not self.files:
            raise RuntimeError(f"target .pt が見つかりません: {targets_split_dir}")
        self.transform = build_eval_transform(resize, imagesize)
        self.load_feature = load_feature
        # fit_score_stats 互換のため items=(idx, sid) を用意
        self.items = [(i, os.path.basename(f)[:-3]) for i, f in enumerate(self.files)]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, j):
        rec = torch.load(self.files[j], weights_only=False)
        image = self.transform(PIL.Image.open(rec["image_path"]).convert("RGB"))
        amap = rec["teacher_anomaly_map"]
        if amap.dim() == 2:
            amap = amap.unsqueeze(0)
        out = {
            "image": image,
            "teacher_map": amap.float(),
            "teacher_score": torch.tensor(
                float(rec["teacher_anomaly_score"]), dtype=torch.float32
            ),
            "label": int(rec["label"]),
            "sample_id": os.path.basename(self.files[j])[:-3],
            "image_path": rec["image_path"],
        }
        if self.load_feature:
            out["teacher_feature"] = rec["teacher_feature_map"].float()
        return out


def _dataset_scores(dataset):
    """dataset（.items/.dir を持つ）から Teacher スコア list を集める。"""
    return [
        float(torch.load(os.path.join(dataset.dir, sid + ".pt"), weights_only=False)[
            "teacher_anomaly_score"
        ])
        for _, sid in dataset.items
    ]


def fit_score_stats(dataset) -> dict:
    """学習データ（単一 or リスト）の Teacher スコアから正規化統計を算出する。"""
    datasets = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    vals = []
    for d in datasets:
        vals.extend(_dataset_scores(d))
    scores = torch.tensor(vals, dtype=torch.float64)
    return {
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "min": float(scores.min()),
        "max": float(scores.max()),
    }


class TargetNormalizer:
    """Teacher map / score を同一アフィン変換で正規化する。

    map と score は同じ距離スケールなので、同一変換で正規化することで
    ``s_T ≈ pool(A_T)`` のスケール対応を保つ（設計書 §21.5）。
    mode: ``none`` / ``minmax`` / ``zscore``。統計は学習データのスコアから算出。
    """

    MODES = ("none", "minmax", "zscore")

    def __init__(self, mode: str, stats: dict):
        if mode not in self.MODES:
            raise ValueError(f"未知の normalize mode: {mode}（候補: {self.MODES}）")
        self.mode = mode
        self.stats = stats

    def _shift_scale(self):
        if self.mode == "minmax":
            lo, hi = self.stats["min"], self.stats["max"]
            return lo, (hi - lo + 1e-8)
        if self.mode == "zscore":
            return self.stats["mean"], (self.stats["std"] + 1e-8)
        return 0.0, 1.0

    def apply(self, x):
        shift, scale = self._shift_scale()
        return (x - shift) / scale

    def to_dict(self):
        return {"mode": self.mode, "stats": self.stats}
