"""precompute した Teacher target の読み込み・突合・正規化ユーティリティ。

Student 学習・評価から共通利用する。画像は既存 MVTecDataset の前処理を
そのまま使うことで、Teacher target 生成時と Student 入力の前処理を一致させる
（設計書 §21.1）。
"""
import glob
import json
import os
import re

import numpy as np
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
                 load_feature: bool = False, max_count: int = None, target_scale: float = 1.0):
        self.dir = targets_split_dir
        self.files = sorted(glob.glob(os.path.join(targets_split_dir, "*.pt")))
        if not self.files:
            raise RuntimeError(f"target .pt が見つかりません: {targets_split_dir}")
        if max_count:
            self.files = self.files[:max_count]
        self.target_scale = float(target_scale)
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
            "teacher_map": amap.float() * self.target_scale,
            "teacher_score": torch.tensor(
                float(rec["teacher_anomaly_score"]) * self.target_scale, dtype=torch.float32
            ),
            "label": int(rec["label"]),
            "sample_id": os.path.basename(self.files[j])[:-3],
            "image_path": rec["image_path"],
        }
        if self.load_feature:
            out["teacher_feature"] = rec["teacher_feature_map"].float()
        return out


def _dataset_scores(dataset):
    """dataset（.items/.dir を持つ）から Teacher スコア list を集める（フォールバック）。"""
    return [
        float(torch.load(os.path.join(dataset.dir, sid + ".pt"), weights_only=False)[
            "teacher_anomaly_score"
        ])
        for _, sid in dataset.items
    ]


def _packed_stats(dataset):
    """PackedTargetDataset（get_scores を持つ）から統計を返す。無ければ None。"""
    if not hasattr(dataset, "get_scores"):
        return None
    s = np.asarray(dataset.get_scores(), dtype=np.float64)
    if s.size == 0:
        return None
    return {"count": int(s.size), "mean": float(s.mean()), "std": float(s.std()),
            "min": float(s.min()), "max": float(s.max())}


def _split_stats_from_json(dataset):
    """dataset.dir に対応する precompute の stats.json から split 統計を返す。

    見つからなければ None。allfeat（特徴入りで .pt が巨大）でも .pt を読まずに
    正規化統計を得るための高速経路。
    """
    packed = _packed_stats(dataset)
    if packed is not None:
        return packed
    run_root = os.path.dirname(os.path.normpath(dataset.dir))
    split = os.path.basename(os.path.normpath(dataset.dir))
    path = os.path.join(run_root, "stats.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        stats = json.load(f)
    s = stats.get(split)
    if not s or s.get("score_mean") is None:
        return None
    return {
        "count": int(s["count"]), "mean": float(s["score_mean"]),
        "std": float(s["score_std"]), "min": float(s["score_min"]),
        "max": float(s["score_max"]),
    }


def fit_score_stats(dataset) -> dict:
    """学習データ（単一 or リスト）の Teacher スコアから正規化統計を算出する。

    各 dataset の precompute stats.json があればそれを正確に合成（.pt を読まない）。
    無い dataset があれば、その分だけ .pt を走査するフォールバックに切り替える。
    """
    datasets = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    per = [_split_stats_from_json(d) for d in datasets]
    if all(p is not None for p in per):  # 高速経路：stats.json から厳密合成
        n = sum(p["count"] for p in per)
        mean = sum(p["count"] * p["mean"] for p in per) / n
        # pooled variance: E[x^2] - mean^2、E[x^2]_i = std_i^2 + mean_i^2
        ex2 = sum(p["count"] * (p["std"] ** 2 + p["mean"] ** 2) for p in per) / n
        var = max(0.0, ex2 - mean ** 2)
        return {
            "mean": mean, "std": var ** 0.5,
            "min": min(p["min"] for p in per), "max": max(p["max"] for p in per),
        }
    vals = []  # フォールバック：.pt を走査
    for d in datasets:
        vals.extend(_dataset_scores(d))
    scores = torch.tensor(vals, dtype=torch.float64)
    return {
        "mean": float(scores.mean()), "std": float(scores.std()),
        "min": float(scores.min()), "max": float(scores.max()),
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


# ---------------------------------------------------------------------------
# 高速化：packed memmap 保存/読み込み・正常の RAM 常駐
# （[[pseudo-anomaly-scale-experiment]] の A2/B1）。
# ---------------------------------------------------------------------------

_NORM_MEAN_T = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_NORM_STD_T = torch.tensor(IMAGENET_STD).view(3, 1, 1)
PACK_MARKER = "packed_meta.json"


class PackedWriter:
    """Student target を packed memmap にストリーミング書き込みする。

    数百万サンプルでも per-file ``.pt`` を作らず、``images``(uint8, resize+crop 済み) /
    ``features``(fp16) / ``targets``(fp16 異常マップ) / ``scores`` / ``labels`` を
    事前確保した memmap に 1 サンプルずつ書く。学習時は :class:`PackedTargetDataset`。
    """

    def __init__(self, out_dir, count, imagesize, map_size, save_image=True,
                 feature_shape=None):
        os.makedirs(out_dir, exist_ok=True)
        self.dir = out_dir
        self.count = int(count)
        self.imagesize = int(imagesize)
        self.map_size = [int(map_size[0]), int(map_size[1])]
        self.save_image = save_image
        self.feature_shape = list(feature_shape) if feature_shape else None
        self.images = (np.lib.format.open_memmap(
            os.path.join(out_dir, "images.npy"), mode="w+", dtype=np.uint8,
            shape=(self.count, 3, self.imagesize, self.imagesize)) if save_image else None)
        self.features = (np.lib.format.open_memmap(
            os.path.join(out_dir, "features.npy"), mode="w+", dtype=np.float16,
            shape=(self.count, *self.feature_shape)) if self.feature_shape else None)
        self.targets = np.lib.format.open_memmap(
            os.path.join(out_dir, "targets.npy"), mode="w+", dtype=np.float16,
            shape=(self.count, self.map_size[0], self.map_size[1]))
        self.scores = np.lib.format.open_memmap(
            os.path.join(out_dir, "scores.npy"), mode="w+", dtype=np.float32,
            shape=(self.count,))
        self.labels = np.lib.format.open_memmap(
            os.path.join(out_dir, "labels.npy"), mode="w+", dtype=np.int8,
            shape=(self.count,))
        self._paths = []

    def write(self, i, target, score, label, image=None, feature=None, image_path=None):
        if self.images is not None and image is not None:
            self.images[i] = np.asarray(image, dtype=np.uint8)
        if self.features is not None and feature is not None:
            self.features[i] = np.asarray(feature, dtype=np.float16)
        self.targets[i] = np.asarray(target, dtype=np.float16)
        self.scores[i] = float(score)
        self.labels[i] = int(label)
        self._paths.append(image_path or "")

    def close(self, extra_meta=None):
        for a in (self.images, self.features, self.targets, self.scores, self.labels):
            if a is not None:
                a.flush()
        s = self.scores.astype(np.float64)
        meta = {
            "count": self.count, "imagesize": self.imagesize, "map_size": self.map_size,
            "save_image": bool(self.images is not None),
            "feature_shape": self.feature_shape,
            "score_mean": float(s.mean()) if len(s) else None,
            "score_std": float(s.std()) if len(s) else None,
            "score_min": float(s.min()) if len(s) else None,
            "score_max": float(s.max()) if len(s) else None,
            "n_anomaly": int((self.labels[:] == 1).sum()),
        }
        if extra_meta:
            meta.update(extra_meta)
        with open(os.path.join(self.dir, PACK_MARKER), "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        with open(os.path.join(self.dir, "image_paths.txt"), "w") as f:
            f.write("\n".join(self._paths))
        return meta


def is_packed(path):
    return os.path.isfile(os.path.join(path, PACK_MARKER))


class PackedTargetDataset(Dataset):
    """:class:`PackedWriter` が書いた packed memmap を Student 学習で読む。

    返り値は :class:`TeacherTargetDataset` と同一キー。``in_memory=True`` で全 memmap を
    RAM に載せる（正常など少数を常駐させ IO を消す）。
    """

    def __init__(self, folder, load_feature=False, in_memory=False, max_count=None,
                 target_scale=1.0, subset="stride"):
        self.dir = folder
        with open(os.path.join(folder, PACK_MARKER)) as f:
            self.meta = json.load(f)
        self.load_feature = load_feature
        # A(マスク直接)ターゲットを PatchCore 異常スケールへ整合させる倍率
        self.target_scale = float(target_scale)
        mmap = None if in_memory else "r"
        self.targets = np.load(os.path.join(folder, "targets.npy"), mmap_mode=mmap)
        self.scores = np.load(os.path.join(folder, "scores.npy"), mmap_mode=mmap)
        self.labels = np.load(os.path.join(folder, "labels.npy"), mmap_mode=mmap)
        self.images = (np.load(os.path.join(folder, "images.npy"), mmap_mode=mmap)
                       if self.meta.get("save_image") else None)
        self.features = (np.load(os.path.join(folder, "features.npy"), mmap_mode=mmap)
                         if load_feature and self.meta.get("feature_shape") else None)
        total = int(self.meta["count"])
        # スケール実験のサブセット：source基点並び（1正常画像→連番変種）のため、
        # 先頭N件(head)だと使われる正常画像の種類が減り「枚数」と「背景多様性」が
        # 交絡する。既定の stride は等間隔抽出で全ソースを均等に含める。
        if max_count and int(max_count) < total:
            n = int(max_count)
            if subset == "head":
                self.indices = np.arange(n, dtype=np.int64)
            else:
                self.indices = (np.arange(n, dtype=np.int64) * (total / n)).astype(np.int64)
            self.count = n
        else:
            self.indices = None
            self.count = total
        self.subset = subset
        # fit_score_stats / 互換のため items=(idx, sid) を用意
        self.items = [(i, f"packed_{i:08d}") for i in range(self.count)]

    def _row(self, j):
        return int(self.indices[j]) if self.indices is not None else int(j)

    def __len__(self):
        return self.count

    def get_scores(self):
        s = (np.asarray(self.scores)[self.indices] if self.indices is not None
             else np.asarray(self.scores[:self.count]))
        return s * self.target_scale

    def __getitem__(self, j):
        i = self._row(j)
        amap = torch.from_numpy(np.array(self.targets[i], dtype=np.float32)).unsqueeze(0)
        amap = amap * self.target_scale
        out = {
            "teacher_map": amap,
            "teacher_score": torch.tensor(float(self.scores[i]) * self.target_scale,
                                          dtype=torch.float32),
            "label": int(self.labels[i]),
            "sample_id": f"packed_{i:08d}",
            "image_path": f"{self.dir}#packed_{i:08d}",
        }
        if self.images is not None:
            img = torch.from_numpy(np.array(self.images[i], dtype=np.float32)).div_(255.0)
            out["image"] = (img - _NORM_MEAN_T) / _NORM_STD_T
        if self.features is not None:
            out["teacher_feature"] = torch.from_numpy(
                np.array(self.features[i], dtype=np.float32))
        return out


class InMemoryCacheDataset(Dataset):
    """任意 dataset の全サンプルを一度だけ __getitem__ して RAM に常駐させる。

    正常（少数）を常駐させ、1:1 サンプラで毎バッチ引かれる正常側の IO を消す（B1）。
    """

    def __init__(self, base, verbose=True):
        self.base = base
        self.dir = getattr(base, "dir", None)
        self.items = getattr(base, "items", None)
        if verbose:
            print(f"正常常駐キャッシュ構築: {len(base)} 件を RAM へ...", flush=True)
        self.cache = [base[i] for i in range(len(base))]
        if verbose:
            print("正常常駐キャッシュ完了", flush=True)

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, j):
        return self.cache[j]
