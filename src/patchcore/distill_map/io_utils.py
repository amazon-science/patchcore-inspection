"""precompute した Teacher target の読み込み・突合・正規化ユーティリティ。

Student 学習・評価から共通利用する。画像は既存 MVTecDataset の前処理を
そのまま使うことで、Teacher target 生成時と Student 入力の前処理を一致させる
（設計書 §21.1）。
"""
import os
import re

import torch
from torch.utils.data import Dataset


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

    def __init__(self, base_dataset, targets_split_dir: str):
        self.base = base_dataset
        self.dir = targets_split_dir
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
        return {
            "image": item["image"],
            "teacher_map": amap.float(),
            "teacher_score": torch.tensor(
                float(rec["teacher_anomaly_score"]), dtype=torch.float32
            ),
            "label": int(rec["label"]),
            "sample_id": sid,
        }


def fit_score_stats(dataset: TeacherTargetDataset) -> dict:
    """学習データの Teacher スコアから正規化統計を算出する。"""
    scores = torch.tensor(
        [
            torch.load(os.path.join(dataset.dir, sid + ".pt"), weights_only=False)[
                "teacher_anomaly_score"
            ]
            for _, sid in dataset.items
        ],
        dtype=torch.float64,
    )
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
