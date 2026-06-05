"""生徒 MLP の学習：教師 faiss バンクを使った距離回帰（知識蒸留）。

学習データは教師バンク特徴量から動的に生成する：
  - 正常多様体近傍：バンクベクトル + ガウス摂動（σ を 0〜sigma_max で変化）
  - 異常側（多様体から遠い点）：特徴空間全体を覆う広いガウスから少数サンプル
各クエリの正解距離（squared L2）は教師バンク（IndexFlatL2）への最近傍探索で付与する。
画像フォワードは不要なので高速。
"""
import numpy as np
import torch
import torch.nn as nn

from patchcore.distill.student import StudentMLP, _EPS


def train_student(
    bank: np.ndarray,
    label_index,
    device,
    hidden=(512, 256),
    n_steps: int = 3000,
    batch: int = 4096,
    lr: float = 1e-3,
    sigma_max: float = 2.0,
    far_frac: float = 0.1,
    seed: int = 0,
    log_every: int = 300,
) -> StudentMLP:
    """教師バンクから距離回帰タスクを生成して生徒 MLP を学習し返す。

    Args:
        bank: [N, D] 教師バンク特徴量（正常パッチ）。
        label_index: 正解距離を引く faiss index（教師バンクの IndexFlatL2、GPU 可）。
        device: torch デバイス。
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    n_total, dim = bank.shape
    feat_std = bank.std(axis=0, keepdims=True).astype(np.float32) + _EPS  # [1, D]
    feat_mean = bank.mean(axis=0, keepdims=True).astype(np.float32)

    model = StudentMLP(dim, hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss()  # Huber

    model.train()
    for step in range(n_steps):
        idx = rng.integers(0, n_total, size=batch)
        sigma = rng.random((batch, 1)).astype(np.float32) * sigma_max
        noise = rng.standard_normal((batch, dim)).astype(np.float32) * feat_std * sigma
        queries = bank[idx] + noise

        n_far = int(batch * far_frac)
        if n_far > 0:
            far = feat_mean + rng.standard_normal((n_far, dim)).astype(
                np.float32
            ) * feat_std * (sigma_max * 2.0)
            queries[:n_far] = far

        queries = np.ascontiguousarray(queries, dtype=np.float32)
        labels, _ = label_index.search(queries, 1)  # squared L2（教師と同一指標）
        target = np.log(labels[:, 0] + _EPS).astype(np.float32)

        x = torch.from_numpy(queries).to(device)
        y = torch.from_numpy(target).to(device)
        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if log_every and (step % log_every == 0 or step == n_steps - 1):
            print(f"  [student] step {step+1}/{n_steps}  loss={loss.item():.4f}")

    model.eval()
    return model
