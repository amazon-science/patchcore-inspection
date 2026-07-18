"""PatchCore Teacher の異常マップ A_T・スコア s_T を事前計算して保存する。

設計書 §9（Option 2: Precomputed Teacher Targets）/ §18.1 に対応。
学習済み PatchCore を 1 度 forward し、各サンプルの異常マップとスコアを
``.pt`` で保存する。Teacher 出力は**生のまま保存**し、正規化（map min-max /
score z-score 等）は学習時に行う（柔軟・可逆のため）。スコアの統計量は
``stats.json`` に併せて残す。

CLAUDE.md「実行・結果出力の規約」に従い、出力は ``results/distill_map/precomputed/
<YYYYMMDD_HHMMSS>/`` 配下に隔離し、``meta.json`` / ``log.txt`` / サニティ図を残す。
保護対象（./dataset, /disk/data_area4）には書き込まない。
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import patchcore.common
import patchcore.patchcore
import patchcore.runlog as runlog
from patchcore.datasets import mvtec
from patchcore.distill_map.io_utils import sample_id as _sample_id

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"

SPLIT_MAP = {"train": mvtec.DatasetSplit.TRAIN, "test": mvtec.DatasetSplit.TEST}


def parse_args():
    p = argparse.ArgumentParser(description="PatchCore Teacher の異常マップ/スコアを precompute")
    p.add_argument("teacher", nargs="?", default=None,
                   help="教師モデルのフォルダ（patchcore_params.pkl を含む）。"
                        "--map_from_mask 使用時は不要")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset_path", default=os.path.join(REPO_ROOT, "dataset"))
    p.add_argument("--classname", default="anomaly_dataset")
    p.add_argument("--resize", type=int, default=256)
    p.add_argument("--imagesize", type=int, default=224)
    p.add_argument(
        "--splits", nargs="+", default=["train", "test"], choices=list(SPLIT_MAP)
    )
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--no_faiss_on_gpu", action="store_true", help="faiss を CPU で動かす")
    p.add_argument(
        "--max_per_split", type=int, default=None, help="全 split 共通の処理枚数上限（スモーク用）"
    )
    p.add_argument("--max_train", type=int, default=None, help="train の枚数上限（max_per_split より優先）")
    p.add_argument("--max_test", type=int, default=None, help="test の枚数上限（max_per_split より優先）")
    p.add_argument(
        "--save_feature", action="store_true",
        help="Pattern A 用に Teacher 特徴マップ F_T([C,H',W'] float16) も保存する",
    )
    p.add_argument("--image_dir", default=None,
                   help="任意画像フォルダを処理（疑似異常等）。--splits の代わりに使う")
    p.add_argument("--image_dir_split", default="pseudo", help="--image_dir の保存 split 名")
    # 疑似異常スケール実験：Teacher をスキップし描画マスクを直接ターゲット化（①用）
    p.add_argument("--map_from_mask", default=None,
                   help="マスクフォルダ。指定時は faiss NN を回さず、ぼかしマスクを"
                        " teacher_anomaly_map にする（--image_dir=異常画像と併用）")
    p.add_argument("--mask_blur_sigma", type=float, default=2.0,
                   help="マスク→ターゲットのガウシアンぼかし σ[px]（0で無効）")
    p.add_argument("--pack", action="store_true",
                   help="per-file .pt でなく packed memmap で保存（大規模向け）")
    p.add_argument(
        "--out",
        default=os.path.join(REPO_ROOT, "results", "distill_map", "precomputed"),
        help="出力親ディレクトリ（この下に timestamp フォルダを作る）",
    )
    p.add_argument(
        "--assets_dir",
        default=os.path.join(REPO_ROOT, "meeting_logs", "roadmap", "assets", "20260622"),
        help="サニティ図の転記先（ロードマップ用）。空文字で無効化",
    )
    return p.parse_args()


def _predict_with_features(pc, images):
    """pc._predict を再現しつつ、特徴格子 F_T([B,C,H',W']) も返す（backbone 1回）。"""
    images = images.to(torch.float).to(pc.device)
    _ = pc.forward_modules.eval()
    batchsize = images.shape[0]
    with torch.no_grad():
        features, patch_shapes = pc._embed(images, provide_patch_shapes=True)
        features = np.asarray(features)  # [B*P, C]
        patch_scores = image_scores = pc.anomaly_scorer.predict([features])[0]
        image_scores = pc.patch_maker.unpatch_scores(image_scores, batchsize=batchsize)
        image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
        image_scores = pc.patch_maker.score(image_scores)
        patch_scores = pc.patch_maker.unpatch_scores(patch_scores, batchsize=batchsize)
        scales = patch_shapes[0]
        patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])
        masks = pc.anomaly_segmentor.convert_to_segmentation(patch_scores)
    c = features.shape[-1]
    feat_grid = features.reshape(batchsize, scales[0], scales[1], c).transpose(0, 3, 1, 2)
    return list(image_scores), list(masks), feat_grid  # feat_grid: [B,C,H',W']


def _process_split(pc, device, args, split_name, split_dir, dataset=None):
    os.makedirs(split_dir, exist_ok=True)
    if dataset is None:
        dataset = mvtec.MVTecDataset(
            args.dataset_path,
            classname=args.classname,
            resize=args.resize,
            imagesize=args.imagesize,
            split=SPLIT_MAP[split_name],
            seed=args.seed,
        )
    limit = {"train": args.max_train, "test": args.max_test}.get(split_name)
    if limit is None:
        limit = args.max_per_split
    n_total = len(dataset)
    if limit:
        n_total = min(limit, n_total)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    scores_all, labels_all = [], []
    map_h = map_w = None
    feat_shape = None
    done = 0
    for batch in loader:
        if done >= n_total:
            break
        images = batch["image"]
        image_names = batch["image_name"]
        full_paths = batch.get("image_path", image_names)
        is_anomaly = batch["is_anomaly"].numpy().tolist()
        masks_gt = batch.get("mask")

        if args.save_feature:
            scores, amaps, feat_grid = _predict_with_features(pc, images)
        else:
            scores, amaps = pc._predict(images)
            feat_grid = None
        for i in range(len(scores)):
            if done >= n_total:
                break
            amap = np.asarray(amaps[i], dtype=np.float32)
            if map_h is None:
                map_h, map_w = amap.shape
            sid = _sample_id(image_names[i])
            record = {
                "image_path": full_paths[i],
                "image_name": image_names[i],
                "teacher_anomaly_map": torch.from_numpy(amap),
                "teacher_anomaly_score": float(scores[i]),
                "label": int(is_anomaly[i]),
            }
            if feat_grid is not None:
                record["teacher_feature_map"] = torch.from_numpy(feat_grid[i]).half()
                if feat_shape is None:
                    feat_shape = list(feat_grid[i].shape)
            if masks_gt is not None and split_name == "test":
                record["mask"] = masks_gt[i].float().squeeze()
            torch.save(record, os.path.join(split_dir, sid + ".pt"))
            scores_all.append(float(scores[i]))
            labels_all.append(int(is_anomaly[i]))
            done += 1
        print(f"  [{split_name}] {done}/{n_total} 枚保存", flush=True)

    scores_arr = np.asarray(scores_all, dtype=np.float64)
    stats = {
        "count": int(len(scores_all)),
        "n_anomaly": int(sum(labels_all)),
        "map_size": [int(map_h), int(map_w)] if map_h else None,
        "feature_shape": feat_shape,
        "score_mean": float(scores_arr.mean()) if len(scores_arr) else None,
        "score_std": float(scores_arr.std()) if len(scores_arr) else None,
        "score_min": float(scores_arr.min()) if len(scores_arr) else None,
        "score_max": float(scores_arr.max()) if len(scores_arr) else None,
    }
    print(f"  [{split_name}] 完了: {stats}")
    return stats


def _list_images(folder):
    import glob
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    return sorted(p for e in exts for p in glob.glob(os.path.join(folder, e)))


def _find_mask(masks_dir, img_path):
    """異常画像に対応するマスクを basename（拡張子違いも許容）で探す。"""
    base = os.path.basename(img_path)
    cand = os.path.join(masks_dir, base)
    if os.path.isfile(cand):
        return cand
    stem = os.path.splitext(base)[0]
    for e in (".png", ".jpg", ".jpeg", ".bmp"):
        c = os.path.join(masks_dir, stem + e)
        if os.path.isfile(c):
            return c
    return None


def _mask_to_target(mask_img, resize, imagesize, sigma):
    """マスク PIL(L) を画像と同一幾何(Resize→CenterCrop)で整列し、ぼかして [0,1] float。"""
    from torchvision import transforms
    m = transforms.Resize(resize, interpolation=transforms.InterpolationMode.NEAREST)(mask_img)
    m = transforms.CenterCrop(imagesize)(m)
    arr = (np.asarray(m, dtype=np.float32) > 127).astype(np.float32)
    if sigma and sigma > 0:
        from scipy.ndimage import gaussian_filter
        arr = gaussian_filter(arr, sigma=sigma)
    return np.clip(arr, 0.0, 1.0)


def _process_mask_split(args, split_name, split_dir):
    """Teacher をスキップし、ぼかしマスクを teacher_anomaly_map として保存する（①用）。"""
    import PIL.Image
    from torchvision import transforms

    from patchcore.distill_map.io_utils import PackedWriter
    from patchcore.distill_map.io_utils import sample_id as _sid

    os.makedirs(split_dir, exist_ok=True)
    img_paths = _list_images(args.image_dir)
    if not img_paths:
        raise RuntimeError(f"異常画像が見つかりません: {args.image_dir}")
    limit = args.max_train or args.max_per_split
    n_total = min(limit, len(img_paths)) if limit else len(img_paths)
    img_paths = img_paths[:n_total]
    S = args.imagesize
    resize_img = transforms.Resize(args.resize)
    crop = transforms.CenterCrop(S)

    writer = None
    if args.pack:
        writer = PackedWriter(split_dir, n_total, imagesize=S, map_size=(S, S), save_image=True)
    scores, labels = [], []
    for i, ip in enumerate(img_paths):
        mp = _find_mask(args.map_from_mask, ip)
        if mp is None:
            raise RuntimeError(f"対応マスクが見つかりません: {ip}")
        target = _mask_to_target(PIL.Image.open(mp).convert("L"),
                                 args.resize, S, args.mask_blur_sigma)
        score = float(target.max())
        if writer is not None:
            pim = crop(resize_img(PIL.Image.open(ip).convert("RGB")))
            arr = np.asarray(pim, dtype=np.uint8).transpose(2, 0, 1)  # [3,S,S]
            writer.write(i, target=target, score=score, label=1, image=arr, image_path=ip)
        else:
            rec = {
                "image_path": ip, "image_name": os.path.basename(ip),
                "teacher_anomaly_map": torch.from_numpy(target),
                "teacher_anomaly_score": score, "label": 1,
            }
            torch.save(rec, os.path.join(split_dir, _sid(os.path.basename(ip)) + ".pt"))
        scores.append(score)
        labels.append(1)
        if (i + 1) % 5000 == 0:
            print(f"  [{split_name}] {i + 1}/{n_total} 保存", flush=True)

    s = np.asarray(scores, dtype=np.float64)
    stats = {
        "count": int(len(scores)), "n_anomaly": int(sum(labels)),
        "map_size": [S, S], "feature_shape": None,
        "score_mean": float(s.mean()) if len(s) else None,
        "score_std": float(s.std()) if len(s) else None,
        "score_min": float(s.min()) if len(s) else None,
        "score_max": float(s.max()) if len(s) else None,
    }
    if writer is not None:
        writer.close(extra_meta={"source": "map_from_mask", "blur_sigma": args.mask_blur_sigma,
                                 "n_anomaly": int(sum(labels))})
    print(f"  [{split_name}] 完了(map_from_mask): {stats}")
    return stats


def _mask_sanity_figure(split_dir, run_dir, args):
    """先頭数件の (異常画像, ぼかしターゲット, オーバーレイ) を並べる。"""
    import PIL.Image
    from patchcore.distill_map.io_utils import (
        IMAGENET_MEAN, IMAGENET_STD, PackedTargetDataset, is_packed)
    rows = []
    if is_packed(split_dir):
        ds = PackedTargetDataset(split_dir, in_memory=False)
        mean = np.array(IMAGENET_MEAN).reshape(1, 1, 3)
        std = np.array(IMAGENET_STD).reshape(1, 1, 3)
        for j in range(min(4, len(ds))):
            it = ds[j]
            img = it["image"].numpy().transpose(1, 2, 0)
            img = np.clip(img * std + mean, 0, 1)
            rows.append((img, it["teacher_map"].numpy()[0]))
    else:
        import glob
        from torchvision import transforms
        for f in sorted(glob.glob(os.path.join(split_dir, "*.pt")))[:4]:
            rec = torch.load(f, weights_only=False)
            im = transforms.CenterCrop(args.imagesize)(
                transforms.Resize(args.resize)(PIL.Image.open(rec["image_path"]).convert("RGB")))
            rows.append((np.asarray(im) / 255.0, rec["teacher_anomaly_map"].numpy()))
    if not rows:
        return None
    fig, axes = plt.subplots(len(rows), 3, figsize=(9, 3 * len(rows)))
    if len(rows) == 1:
        axes = axes[None, :]
    for r, (img, tgt) in enumerate(rows):
        axes[r, 0].imshow(img); axes[r, 0].set_title("異常画像")
        im = axes[r, 1].imshow(tgt, cmap="jet", vmin=0, vmax=1)
        axes[r, 1].set_title("ぼかしターゲット")
        axes[r, 2].imshow(img); axes[r, 2].imshow(tgt, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        axes[r, 2].set_title("オーバーレイ")
        for c in range(3):
            axes[r, c].set_xlabel("x [px]"); axes[r, c].set_ylabel("y [px]")
    fig.suptitle(f"map_from_mask サニティ（σ={args.mask_blur_sigma}）", fontsize=11)
    fig.tight_layout()
    out = os.path.join(run_dir, "mask_targets_sanity.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _sanity_figure(run_dir, args):
    """test split から正常/異常を各2枚選び、入力と A_T ヒートマップを並べる。"""
    test_dir = os.path.join(run_dir, "test")
    if not os.path.isdir(test_dir):
        return None
    files = sorted(os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".pt"))
    picked = {"good": [], "anomaly": []}
    for f in files:
        rec = torch.load(f, map_location="cpu", weights_only=False)
        key = "anomaly" if rec["label"] == 1 else "good"
        if len(picked[key]) < 2:
            picked[key].append(rec)
        if len(picked["good"]) >= 2 and len(picked["anomaly"]) >= 2:
            break
    rows = picked["good"] + picked["anomaly"]
    if not rows:
        return None

    fig, axes = plt.subplots(len(rows), 1, figsize=(4, 3 * len(rows)))
    if len(rows) == 1:
        axes = [axes]
    for ax, rec in zip(axes, rows):
        amap = rec["teacher_anomaly_map"].numpy()
        im = ax.imshow(amap, cmap="jet")
        kind = "異常" if rec["label"] == 1 else "正常"
        ax.set_title(f"{kind}: s_T={rec['teacher_anomaly_score']:.3f}", fontsize=9)
        ax.set_xlabel("x [px]")
        ax.set_ylabel("y [px]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="異常度 A_T")
    fig.suptitle("Teacher 異常マップ サニティ（正常 vs 異常）", fontsize=10)
    fig.tight_layout()
    out_path = os.path.join(run_dir, "teacher_targets_sanity.png")
    fig.savefig(out_path, dpi=150)
    if args.assets_dir:
        os.makedirs(args.assets_dir, exist_ok=True)
        fig.savefig(os.path.join(args.assets_dir, "teacher_targets_sanity.png"), dpi=150)
    plt.close(fig)
    return out_path


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    on_gpu = not args.no_faiss_on_gpu
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    run_dir = runlog.make_run_dir(args.out)
    runlog.start_tee(run_dir)
    print(f"出力先: {run_dir}")
    print(f"Device: {device}  faiss_on_gpu={on_gpu}")

    all_stats = {}
    # --- ①用の高速経路：Teacher をスキップし描画マスクを直接ターゲット化 ---
    if args.map_from_mask:
        if not args.image_dir:
            raise SystemExit("--map_from_mask は --image_dir（異常画像フォルダ）と併用してください")
        if args.save_feature:
            raise SystemExit("--map_from_mask + --save_feature は未対応（②③実装時に追加予定）")
        split_name = args.image_dir_split
        split_dir = os.path.join(run_dir, split_name)
        print(f"map_from_mask: 画像={args.image_dir} / マスク={args.map_from_mask} "
              f"→ split='{split_name}' (σ={args.mask_blur_sigma}, pack={args.pack})")
        all_stats[split_name] = _process_mask_split(args, split_name, split_dir)
        with open(os.path.join(run_dir, "stats.json"), "w") as f:
            json.dump(all_stats, f, ensure_ascii=False, indent=2)
        sanity = _mask_sanity_figure(split_dir, run_dir, args)
        print(f"サニティ図: {sanity}")
        runlog.write_meta(run_dir, extra={
            "map_from_mask": args.map_from_mask, "image_dir": args.image_dir,
            "image_dir_split": split_name, "mask_blur_sigma": args.mask_blur_sigma,
            "pack": args.pack, "resize": args.resize, "imagesize": args.imagesize,
            "seed": args.seed, "stats": all_stats,
        })
        print(f"完了: {run_dir}")
        return

    if not args.teacher:
        raise SystemExit("teacher（教師モデルのフォルダ）を指定してください（--map_from_mask 以外では必須）")
    nn_method = patchcore.common.FaissNN(on_gpu, args.num_workers)
    pc = patchcore.patchcore.PatchCore(device)
    pc.load_from_path(args.teacher, device=device, nn_method=nn_method)
    print(f"教師ロード完了: {args.teacher}")

    if args.image_dir:
        from patchcore.distill_map.io_utils import ImageFolderDataset
        split_name = args.image_dir_split
        ds = ImageFolderDataset(args.image_dir, resize=args.resize, imagesize=args.imagesize)
        print(f"画像フォルダ処理: {args.image_dir} → split='{split_name}' ({len(ds)} 枚)")
        split_dir = os.path.join(run_dir, split_name)
        all_stats[split_name] = _process_split(
            pc, device, args, split_name, split_dir, dataset=ds
        )
    else:
        for split_name in args.splits:
            split_dir = os.path.join(run_dir, split_name)
            all_stats[split_name] = _process_split(pc, device, args, split_name, split_dir)

    with open(os.path.join(run_dir, "stats.json"), "w") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)

    sanity = _sanity_figure(run_dir, args)
    print(f"サニティ図: {sanity}")

    runlog.write_meta(
        run_dir,
        extra={
            "teacher": args.teacher,
            "classname": args.classname,
            "resize": args.resize,
            "imagesize": args.imagesize,
            "splits": args.splits,
            "seed": args.seed,
            "stats": all_stats,
        },
    )
    print(f"完了: {run_dir}")


if __name__ == "__main__":
    main()
