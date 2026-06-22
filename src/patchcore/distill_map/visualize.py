"""異常マップの可視化（設計書 §16）。

入力画像・Teacher map・Student map・差分・overlay を 1 枚のパネルにまとめて保存する。
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"

# MVTecDataset の Normalize に対応（ImageNet 統計）
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denorm_image(img: torch.Tensor) -> np.ndarray:
    """正規化済み画像テンソル [3,H,W] を表示用 [H,W,3] (0-1) に戻す。"""
    x = img.detach().cpu() * _STD + _MEAN
    x = x.clamp(0, 1).permute(1, 2, 0).numpy()
    return x


def save_panel(out_dir, sample_id, image, teacher_map, student_map, score_info=None):
    """1 サンプルの可視化パネルを ``<out_dir>/<sample_id>_panel.png`` に保存。"""
    os.makedirs(out_dir, exist_ok=True)
    img = denorm_image(image)
    t = teacher_map.detach().cpu().float().numpy()
    s = student_map.detach().cpu().float().squeeze().numpy()
    diff = np.abs(t - s)
    vmax = max(float(t.max()), float(s.max()), 1e-6)

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    axes[0].imshow(img); axes[0].set_title("入力画像")
    im1 = axes[1].imshow(t, cmap="jet", vmin=0, vmax=vmax); axes[1].set_title("Teacher A_T")
    im2 = axes[2].imshow(s, cmap="jet", vmin=0, vmax=vmax); axes[2].set_title("Student A_S")
    im3 = axes[3].imshow(diff, cmap="jet"); axes[3].set_title("|A_T - A_S|")
    axes[4].imshow(img)
    axes[4].imshow(s, cmap="jet", alpha=0.5, vmin=0, vmax=vmax)
    axes[4].set_title("overlay (A_S)")
    for ax in axes:
        ax.set_xlabel("x [px]"); ax.set_ylabel("y [px]")
    for im, ax in ((im1, axes[1]), (im2, axes[2]), (im3, axes[3])):
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if score_info:
        fig.suptitle(score_info, fontsize=11)
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{sample_id}_panel.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def save_roc(out_path, labels, teacher_scores, student_scores, aurocs):
    """Teacher vs Student の image-level ROC を 1 枚に重ねる。"""
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(5, 5))
    for name, sc, color in (
        ("Teacher", teacher_scores, "C0"),
        ("Student", student_scores, "C1"),
    ):
        fpr, tpr, _ = roc_curve(labels, sc)
        ax.plot(fpr, tpr, color=color, label=f"{name} (AUROC={aurocs[name]:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("image-level ROC: Teacher vs Student")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
