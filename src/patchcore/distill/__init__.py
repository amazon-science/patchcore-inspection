"""教師 PatchCore メモリバンクの蒸留によるメモリ削減のための共通基盤。

教師の異常スコアは「各パッチ特徴量の最近傍 L2 距離（の平均）」。この
`パッチ特徴量 → スコア` の写像を、巨大な faiss バンク以外の軽量表現へ
差し替え可能にするための共通インターフェース `Scorer` と、各手法を同一基準
（AUROC × メモリ × FPS）で評価するためのユーティリティを提供する。
"""
import abc
import os

import faiss
import numpy as np


def extract_bank_vectors(faiss_index, batch: int = 100_000) -> np.ndarray:
    """faiss index の全ベクトルを numpy 配列として取り出す（reconstruct_n）。

    教師バンクに格納された正常パッチ特徴量そのもの。蒸留の学習データになる。
    """
    ntotal = faiss_index.ntotal
    dim = faiss_index.d
    out = np.empty((ntotal, dim), dtype=np.float32)
    for start in range(0, ntotal, batch):
        n = min(batch, ntotal - start)
        out[start : start + n] = faiss_index.reconstruct_n(start, n)
    return out


def file_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0.0


def index_size_mb(index) -> float:
    """faiss index のシリアライズ後バイト数を MB で返す（GPU index は CPU へ複製）。"""
    cpu_index = index
    try:
        cpu_index = faiss.index_gpu_to_cpu(index)
    except Exception:
        pass
    data = np.asarray(faiss.serialize_index(cpu_index))
    return data.nbytes / (1024 * 1024)


class Scorer(abc.ABC):
    """パッチ特徴量 → パッチ毎異常スコア の写像（メモリ削減手法の共通 IF）。"""

    name = "base"

    @abc.abstractmethod
    def score_patches(self, features: np.ndarray) -> np.ndarray:
        """features: [M, D] → per-patch scores: [M]（高いほど異常）。"""

    @abc.abstractmethod
    def memory_mb(self) -> float:
        """推論時に常駐させるメモリフットプリント（MB）。"""
