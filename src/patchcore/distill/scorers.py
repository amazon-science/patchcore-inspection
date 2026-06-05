"""faiss ベースのスコアラ（M0: 教師, M1: PQ量子化, M2: コアセット再圧縮）。"""
import faiss
import numpy as np

from patchcore.distill import Scorer, index_size_mb

_GPU_RES = None


def _gpu_res():
    global _GPU_RES
    if _GPU_RES is None:
        _GPU_RES = faiss.StandardGpuResources()
    return _GPU_RES


def _to_gpu(index, float16: bool = False):
    if hasattr(faiss, "StandardGpuResources"):
        options = faiss.GpuClonerOptions()
        # IVFPQ を GPU に載せる際は float16 ルックアップテーブルが必須
        # （shared memory 上限。既存 ApproximateFaissNN と同設定）
        options.useFloat16 = float16
        return faiss.index_cpu_to_gpu(_gpu_res(), 0, index, options)
    return index


def _search_scores(index, features, n_nn):
    query = np.ascontiguousarray(features, dtype=np.float32)
    distances, _ = index.search(query, n_nn)
    return distances.mean(axis=-1)


class FlatScorer(Scorer):
    """M0: 教師そのもの。厳密 L2（IndexFlatL2）で最近傍距離を計算する基準手法。"""

    def __init__(self, bank: np.ndarray, n_nn: int = 1, on_gpu: bool = True):
        self.name = "M0_teacher_flat"
        self.n_nn = n_nn
        cpu_index = faiss.IndexFlatL2(bank.shape[1])
        cpu_index.add(bank)
        self._size_mb = index_size_mb(cpu_index)
        self.index = _to_gpu(cpu_index) if on_gpu else cpu_index

    def score_patches(self, features):
        return _search_scores(self.index, features, self.n_nn)

    def memory_mb(self):
        return self._size_mb


class PQScorer(Scorer):
    """M1: IVFPQ による積量子化でバンクを圧縮（既存 ApproximateFaissNN と同設定）。"""

    def __init__(
        self,
        bank: np.ndarray,
        n_nn: int = 1,
        on_gpu: bool = True,
        nlist: int = 512,
        m: int = 64,
        nbits: int = 8,
        nprobe: int = 16,
    ):
        dim = bank.shape[1]
        ntotal = len(bank)
        # 学習点が少ない（スモーク等）場合は nlist を縮小
        nlist = max(1, min(nlist, ntotal // 40)) if ntotal < nlist * 40 else nlist
        self.name = f"M1_pq_m{m}_nbits{nbits}"
        self.n_nn = n_nn
        quantizer = faiss.IndexFlatL2(dim)
        cpu_index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
        cpu_index.train(bank)
        cpu_index.add(bank)
        cpu_index.nprobe = min(nprobe, nlist)
        self._size_mb = index_size_mb(cpu_index)
        self.index = _to_gpu(cpu_index, float16=True) if on_gpu else cpu_index
        try:
            self.index.nprobe = min(nprobe, nlist)
        except Exception:
            pass

    def score_patches(self, features):
        return _search_scores(self.index, features, self.n_nn)

    def memory_mb(self):
        return self._size_mb


class CoresetScorer(Scorer):
    """M2: コアセットを再サンプリングしてバンクの件数を削減（厳密 L2 のまま）。"""

    def __init__(self, bank: np.ndarray, sampler, n_nn: int = 1, on_gpu: bool = True):
        reduced = np.ascontiguousarray(sampler.run(bank), dtype=np.float32)
        self.n_vectors = len(reduced)
        self.name = f"M2_coreset_{self.n_vectors}"
        self.n_nn = n_nn
        cpu_index = faiss.IndexFlatL2(reduced.shape[1])
        cpu_index.add(reduced)
        self._size_mb = index_size_mb(cpu_index)
        self.index = _to_gpu(cpu_index) if on_gpu else cpu_index

    def score_patches(self, features):
        return _search_scores(self.index, features, self.n_nn)

    def memory_mb(self):
        return self._size_mb
