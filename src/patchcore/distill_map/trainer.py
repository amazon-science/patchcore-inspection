"""Student 蒸留学習ループ（設計書 §14.3）。

precompute 済み Teacher target を教師信号に、Student（画像→異常マップ）を
``L = lambda_map*L_map + lambda_score*L_score`` で学習する。AdamW + cosine、
AMP、early stopping、学習曲線出力に対応。

疑似異常スケール実験向けに以下を追加（[[pseudo-anomaly-scale-experiment]]）:
- normal / pseudo を別々に受け取り、各々を train と **hold-out（学習に使わない・
  オーバーサンプルしない）** に分割。
- 正常の過学習検知：train_normal（probe）と holdout_normal の map 損失ギャップ曲線
  ``overfit_normal.png`` と、holdout 正常スコア分布（collapse 検知）を毎 epoch 記録。
- 1:1（``--normal_pseudo_ratio``）の比率固定サンプラ（WeightedRandomSampler）。
- 高速化：pin_memory / persistent_workers / prefetch / channels_last / cudnn.benchmark。
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Subset,
    WeightedRandomSampler,
)

from patchcore.distill_map.losses import DistillLoss

matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"


def _split_indices(n, val_ratio, seed):
    """index を train/holdout に分割して返す（val_ratio<=0 で holdout なし）。"""
    if val_ratio <= 0 or n <= 1:
        return list(range(n)), []
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_hold = max(1, int(round(n * val_ratio)))
    return perm[n_hold:], perm[:n_hold]


def split_train_val(dataset, val_ratio: float, seed: int):
    """後方互換：単一 dataset を index ベースで train/val に分割する。"""
    tr, ho = _split_indices(len(dataset), val_ratio, seed)
    if not ho:
        return dataset, None
    return Subset(dataset, tr), Subset(dataset, ho)


def _loader(dataset, batch_size, num_workers, shuffle=False, sampler=None,
            drop_last=False, prefetch_factor=4, pin_memory=False):
    kw = dict(batch_size=batch_size, num_workers=num_workers, drop_last=drop_last,
              pin_memory=pin_memory)
    if sampler is not None:
        kw["sampler"] = sampler
    else:
        kw["shuffle"] = shuffle
    if num_workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kw)


def _ratio_sampler(n_normal, n_pseudo, ratio, steps_per_epoch, batch_size, seed):
    """normal:pseudo を ``ratio:1`` に保つ WeightedRandomSampler を作る。

    ConcatDataset([normal_train, pseudo_train]) を前提に、index 0..n_normal-1 を
    normal、以降を pseudo とみなす。既定 num_samples は各 pseudo を約1回・normal を
    約 ratio 倍サンプルする長さ（steps_per_epoch>0 なら固定長に上書き）。
    """
    w = torch.empty(n_normal + n_pseudo, dtype=torch.double)
    w[:n_normal] = ratio * n_pseudo / max(n_normal, 1)
    w[n_normal:] = 1.0
    if steps_per_epoch and steps_per_epoch > 0:
        num_samples = steps_per_epoch * batch_size
    else:
        num_samples = int(round(n_pseudo * (1.0 + ratio)))
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(w, num_samples=num_samples, replacement=True, generator=g)


def _forward_batch(student, batch, crit, normalizer, device, input_key, feature_mode,
                   use_amp, channels_last, want_scores=False):
    """1 バッチの forward。loss と logs、（want_scores 時）student スコアを返す。"""
    images = batch[input_key].to(device, non_blocking=True)
    if channels_last and images.dim() == 4:
        images = images.contiguous(memory_format=torch.channels_last)
    with torch.autocast("cuda", enabled=use_amp):
        if feature_mode:
            f_t = batch["teacher_feature"].to(device, non_blocking=True)
            f_s = student(images)
            loss = F.mse_loss(f_s, f_t)
            return loss, {"loss": float(loss), "feat": float(loss)}, None
        a_t = normalizer.apply(batch["teacher_map"].to(device, non_blocking=True))
        s_t = normalizer.apply(batch["teacher_score"].to(device, non_blocking=True))
        a_s = student(images)
        s_s = student.score(a_s)
        loss, logs = crit(a_s, a_t, s_s, s_t)
    scores = s_s.detach().float().flatten() if want_scores else None
    return loss, logs, scores


def _run_epoch(student, loader, crit, normalizer, device, optimizer=None, scaler=None,
               input_key="image", feature_mode=False, channels_last=False,
               collect_scores=False):
    """1 epoch 分の学習 or 評価。平均 logs（と collect_scores 時スコア統計）を返す。"""
    train = optimizer is not None
    student.train(train)
    totals = {"loss": 0.0, "feat": 0.0} if feature_mode else {"loss": 0.0, "map": 0.0, "score": 0.0}
    n = 0
    score_chunks = []
    use_amp = scaler is not None
    for batch in loader:
        bs = batch[input_key].shape[0]
        with torch.set_grad_enabled(train):
            loss, logs, scores = _forward_batch(
                student, batch, crit, normalizer, device, input_key, feature_mode,
                use_amp, channels_last, want_scores=collect_scores and not feature_mode,
            )
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
        if scores is not None:
            score_chunks.append(scores.cpu())
        n += bs
    out = {k: v / max(n, 1) for k, v in totals.items()}
    if collect_scores and score_chunks:
        s = torch.cat(score_chunks)
        out["score_mean"] = float(s.mean())
        out["score_std"] = float(s.std())
    return out


def _plot_curve(history, out_path):
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(6, 4))
    if "train_feat" in history[0]:
        ax.plot(epochs, [h["train_feat"] for h in history], label="train feat MSE")
        if history[0].get("val_feat") is not None:
            ax.plot(epochs, [h["val_feat"] for h in history], label="val feat MSE", color="k")
    else:
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


def _plot_overfit(history, out_path):
    """正常の過学習可視化：(左) train/holdout の map 損失ギャップ、(右) 正常スコア分布。"""
    ep = [h["epoch"] for h in history]
    has_score = history[0].get("holdout_normal_score_mean") is not None
    fig, axes = plt.subplots(1, 2 if has_score else 1, figsize=(11 if has_score else 6, 4))
    axes = axes if has_score else [axes]
    tr = [h.get("train_normal_map") for h in history]
    ho = [h.get("holdout_normal_map") for h in history]
    axes[0].plot(ep, tr, label="train 正常 (probe)", color="#1f77b4")
    axes[0].plot(ep, ho, label="hold-out 正常 (未使用)", color="#d62728")
    axes[0].fill_between(ep, tr, ho, color="#d62728", alpha=0.12, label="gap（過学習指標）")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("map 損失（正常）")
    axes[0].set_title("正常の train/hold-out ギャップ"); axes[0].legend(); axes[0].grid(alpha=0.3)
    if has_score:
        mean = [h["holdout_normal_score_mean"] for h in history]
        std = [h["holdout_normal_score_std"] for h in history]
        lo = [m - s for m, s in zip(mean, std)]
        hi = [m + s for m, s in zip(mean, std)]
        axes[1].plot(ep, mean, color="#2ca02c", label="hold-out 正常スコア mean")
        axes[1].fill_between(ep, lo, hi, color="#2ca02c", alpha=0.15, label="±std")
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("Student 異常スコア（正常）")
        axes[1].set_title("正常スコア分布（0 へ張り付き=collapse）")
        axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def train_student(student, normal_ds, normalizer, device, args, run_dir, pseudo_ds=None):
    """Student を学習し、best/last checkpoint・学習曲線・過学習図・train_log.csv を残す。

    normal_ds / pseudo_ds を別々に受け取り、各々を train と hold-out に分割する。
    """
    input_key = getattr(args, "input_key", "image")
    feature_mode = getattr(args, "pattern", "image_to_map") == "image_to_feature"
    seed = args.seed
    val_ratio = args.val_ratio
    num_workers = args.num_workers
    prefetch = getattr(args, "prefetch_factor", 4)
    ratio = getattr(args, "normal_pseudo_ratio", 1.0)
    steps_per_epoch = getattr(args, "steps_per_epoch", 0)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    channels_last = bool(getattr(args, "channels_last", 1)) and device.type == "cuda"
    student.to(device)
    if channels_last:
        student.to(memory_format=torch.channels_last)

    # --- normal / pseudo を train と hold-out（未使用・非オーバーサンプル）に分割 ---
    n_tr_idx, n_ho_idx = _split_indices(len(normal_ds), val_ratio, seed)
    normal_train = Subset(normal_ds, n_tr_idx)
    normal_hold = Subset(normal_ds, n_ho_idx) if n_ho_idx else None
    # train 側 probe（過学習ギャップ用に normal_train の固定部分集合）
    probe_idx = n_tr_idx[: min(len(n_tr_idx), getattr(args, "probe_size", 2000))]
    normal_probe = Subset(normal_ds, probe_idx) if probe_idx else None

    pseudo_train = pseudo_hold = None
    if pseudo_ds is not None:
        p_tr_idx, p_ho_idx = _split_indices(len(pseudo_ds), val_ratio, seed)
        pseudo_train = Subset(pseudo_ds, p_tr_idx)
        pseudo_hold = Subset(pseudo_ds, p_ho_idx) if p_ho_idx else None

    pin = device.type == "cuda"
    if pseudo_train is not None:
        train_concat = ConcatDataset([normal_train, pseudo_train])
        sampler = _ratio_sampler(len(normal_train), len(pseudo_train), ratio,
                                 steps_per_epoch, args.batch_size, seed)
        train_loader = _loader(train_concat, args.batch_size, num_workers, sampler=sampler,
                               drop_last=True, prefetch_factor=prefetch, pin_memory=pin)
        print(f"train: normal {len(normal_train)} + pseudo {len(pseudo_train)}"
              f"（比率 normal:pseudo={ratio:.2g}:1, num_samples={len(sampler)}）")
    else:
        train_loader = _loader(normal_train, args.batch_size, num_workers, shuffle=True,
                               drop_last=True, prefetch_factor=prefetch, pin_memory=pin)
        print(f"train: normal {len(normal_train)}（pseudo なし）")

    def mk_eval(ds):
        return _loader(ds, args.batch_size, num_workers, shuffle=False,
                       prefetch_factor=prefetch, pin_memory=pin) if ds and len(ds) else None
    normal_hold_loader = mk_eval(normal_hold)
    normal_probe_loader = mk_eval(normal_probe)
    pseudo_hold_loader = mk_eval(pseudo_hold)

    crit = DistillLoss(
        map_loss_kind=args.map_loss, lambda_map=args.lambda_map,
        lambda_score=args.lambda_score, use_weighted_map_loss=args.use_weighted_map_loss,
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
            "pattern": getattr(args, "pattern", "image_to_map"),
        }

    def _eval(loader, collect_scores=False):
        if loader is None:
            return None
        return _run_epoch(student, loader, crit, normalizer, device, input_key=input_key,
                          feature_mode=feature_mode, channels_last=channels_last,
                          collect_scores=collect_scores)

    loss_key = "feat" if feature_mode else "map"
    for epoch in range(1, args.epochs + 1):
        tr = _run_epoch(student, train_loader, crit, normalizer, device,
                        optimizer=optimizer, scaler=scaler, input_key=input_key,
                        feature_mode=feature_mode, channels_last=channels_last)
        nho = _eval(normal_hold_loader, collect_scores=True)
        npr = _eval(normal_probe_loader)
        pho = _eval(pseudo_hold_loader)
        scheduler.step()

        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
               "train_loss": tr["loss"]}
        if feature_mode:
            row["train_feat"] = tr["feat"]
        else:
            row["train_map"] = tr["map"]; row["train_score"] = tr["score"]
        # 過学習検知（正常 train probe vs holdout）
        row["train_normal_map"] = npr[loss_key] if npr else None
        row["holdout_normal_map"] = nho[loss_key] if nho else None
        row["holdout_normal_loss"] = nho["loss"] if nho else None
        row["holdout_pseudo_map"] = pho[loss_key] if pho else None
        row["holdout_pseudo_loss"] = pho["loss"] if pho else None
        if nho and "score_mean" in nho:
            row["holdout_normal_score_mean"] = nho["score_mean"]
            row["holdout_normal_score_std"] = nho["score_std"]
        if npr and nho:
            row["overfit_gap"] = (nho[loss_key] - npr[loss_key])
        history.append(row)

        # early stopping は hold-out（正常＋疑似）を監視。無ければ train。
        mon_parts = [v for v in (row.get("holdout_normal_loss"),
                                 row.get("holdout_pseudo_loss")) if v is not None]
        monitor = sum(mon_parts) / len(mon_parts) if mon_parts else tr["loss"]

        msg = f"[{epoch:3d}/{args.epochs}] train={tr['loss']:.4f}"
        if not feature_mode:
            msg += f" (map={tr['map']:.4f} score={tr['score']:.4f})"
        if nho:
            msg += f"  ho_normal_{loss_key}={nho[loss_key]:.4f}"
        if npr and nho:
            msg += f"  gap={row['overfit_gap']:+.4f}"
        if pho:
            msg += f"  ho_pseudo_{loss_key}={pho[loss_key]:.4f}"

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

    # 揃わない列があり得るので全 key の和集合で書き出す
    keys = []
    for h in history:
        for k in h:
            if k not in keys:
                keys.append(k)
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)
    _plot_curve(history, os.path.join(run_dir, "train_curve.png"))
    if any(h.get("holdout_normal_map") is not None for h in history):
        _plot_overfit(history, os.path.join(run_dir, "overfit_normal.png"))
    print(f"best monitor={best_metric:.4f}  checkpoints={ckpt_dir}")
    return {"history": history, "best_metric": best_metric, "ckpt_dir": ckpt_dir}
