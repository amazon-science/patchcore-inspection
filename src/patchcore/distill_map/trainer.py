"""Student 蒸留学習ループ（設計書 §14.3）。

precompute 済み Teacher target を教師信号に、Student（画像→異常マップ）を
``L = lambda_map*L_map + lambda_score*L_score`` で学習する。AdamW + cosine、
AMP、early stopping、学習曲線出力に対応。
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

from patchcore.distill_map import io_utils
from patchcore.distill_map.losses import DistillLoss

matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"


def split_train_val(dataset, val_ratio: float, seed: int):
    """index ベースで train/val に分割する（val_ratio<=0 で val なし）。"""
    n = len(dataset)
    if val_ratio <= 0:
        return dataset, None
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = max(1, int(round(n * val_ratio)))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def _run_epoch(student, loader, crit, normalizer, device, optimizer=None, scaler=None):
    train = optimizer is not None
    student.train(train)
    totals = {"loss": 0.0, "map": 0.0, "score": 0.0}
    n = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        a_t = normalizer.apply(batch["teacher_map"].to(device, non_blocking=True))
        s_t = normalizer.apply(batch["teacher_score"].to(device, non_blocking=True))
        bs = images.shape[0]
        with torch.set_grad_enabled(train):
            with torch.autocast("cuda", enabled=scaler is not None):
                a_s = student(images)
                s_s = student.score(a_s)
                loss, logs = crit(a_s, a_t, s_s, s_t)
        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        for k in totals:
            totals[k] += float(logs[k]) * bs
        n += bs
    return {k: v / max(n, 1) for k, v in totals.items()}


def _plot_curve(history, out_path):
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [h["train_loss"] for h in history], label="train total")
    ax.plot(epochs, [h["train_map"] for h in history], label="train map", ls="--")
    ax.plot(epochs, [h["train_score"] for h in history], label="train score", ls=":")
    if history[0].get("val_loss") is not None:
        ax.plot(epochs, [h["val_loss"] for h in history], label="val total", color="k")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("蒸留学習曲線")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def train_student(student, full_dataset, normalizer, device, args, run_dir):
    """Student を学習し、best/last checkpoint・学習曲線・train_log.csv を残す。"""
    train_set, val_set = split_train_val(full_dataset, args.val_ratio, args.seed)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=False,
    )
    val_loader = (
        DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                   num_workers=args.num_workers)
        if val_set is not None else None
    )
    print(f"train={len(train_set)}  val={len(val_set) if val_set else 0}")

    student.to(device)
    crit = DistillLoss(
        map_loss_kind=args.map_loss,
        lambda_map=args.lambda_map,
        lambda_score=args.lambda_score,
        use_weighted_map_loss=args.use_weighted_map_loss,
        anomaly_weight_alpha=args.anomaly_weight_alpha,
    )
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "train_log.csv")
    history = []
    best_metric = float("inf")
    bad_epochs = 0

    def _ckpt_payload():
        return {
            "student_state_dict": student.state_dict(),
            "normalizer": normalizer.to_dict(),
            "student_kwargs": getattr(args, "student_kwargs", {}),
        }

    for epoch in range(1, args.epochs + 1):
        tr = _run_epoch(student, train_loader, crit, normalizer, device,
                        optimizer=optimizer, scaler=scaler)
        va = (_run_epoch(student, val_loader, crit, normalizer, device)
              if val_loader is not None else None)
        scheduler.step()
        row = {
            "epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
            "train_loss": tr["loss"], "train_map": tr["map"], "train_score": tr["score"],
            "val_loss": va["loss"] if va else None,
            "val_map": va["map"] if va else None,
            "val_score": va["score"] if va else None,
        }
        history.append(row)
        monitor = va["loss"] if va else tr["loss"]
        msg = (f"[{epoch:3d}/{args.epochs}] train={tr['loss']:.4f} "
               f"(map={tr['map']:.4f} score={tr['score']:.4f})")
        if va:
            msg += f"  val={va['loss']:.4f}"
        torch.save(_ckpt_payload(), os.path.join(ckpt_dir, "last.ckpt"))
        if monitor < best_metric - 1e-6:
            best_metric = monitor
            bad_epochs = 0
            torch.save(_ckpt_payload(), os.path.join(ckpt_dir, "best.ckpt"))
            msg += "  *best"
        else:
            bad_epochs += 1
        print(msg, flush=True)
        if args.early_stopping and bad_epochs >= args.patience:
            print(f"early stopping at epoch {epoch} (patience={args.patience})")
            break

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    _plot_curve(history, os.path.join(run_dir, "train_curve.png"))
    print(f"best monitor={best_metric:.4f}  checkpoints={ckpt_dir}")
    return {"history": history, "best_metric": best_metric, "ckpt_dir": ckpt_dir}
