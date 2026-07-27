#!/usr/bin/env bash
# 疑似B方式10x の ②③ を **nvme 上の特徴** で実行する（run_pseudoB_23.sh の高速版）。
#
# 経緯：疑似特徴 384GB を NFS(data_area4) から乱択で読むと実効 17.7MB/s しか出ず、
# ② が 1epoch=97分（102.8GB/epoch）で 30epoch に 2日近くかかった。nvme は約100倍
# 速い（1,731MB/s）ため、特徴を nvme へコピーしてから学習する。条件（10x・
# steps_per_epoch）は一切変えない。
#
# 前提：/tmp/t-hori/bfai/pseudoB_feat_local/ に features/targets/scores/labels/images
#       が揃っていること（images は B方式 pack への symlink）。
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=./src
PY=.venv/bin/python

GPU="${GPU:-1}"
EPOCHS="${EPOCHS:-30}"
STEPS="${STEPS:-2000}"
OUT="${OUT:-results/distill_map/pseudoB_10x}"
TFEAT="${TFEAT:-/tmp/t-hori/bfai/allfeat_local/20260623_042006}"   # 正常/評価(特徴あり, nvme)
PB="${PB:-/tmp/t-hori/bfai/pseudoB_feat_local}"                    # 疑似B 10x 特徴(nvme)
TEACHER="${TEACHER:-results/ContinualLearning/20260526_比較検討/20260527_064537_N1000_P5/round_21}"
FS_OUT="${FS_OUT:-/disk/data_area4/t-hori/bfai/student_feat}"      # F_S は大容量なので data_area4

mkdir -p "$OUT"
common=(--gpu "$GPU" --epochs "$EPOCHS" --batch_size 32
        --normal_pseudo_ratio 1.0 --steps_per_epoch "$STEPS" --out "$OUT")

# ---- 1) ② feature_to_map（特徴→異常map）----
echo "[1] ② feature_to_map 学習開始 $(date +%H:%M:%S)"
$PY bin/train_distill_map.py "$TFEAT" --pseudo_dir "$PB" \
  --pattern feature_to_map --name P2B_feature_to_map "${common[@]}"
R2=$(ls -1dt "$OUT"/P2B_feature_to_map_*/ | head -1)
echo "[1] 完了 $R2"

echo "[1e] ② 評価 $(date +%H:%M:%S)"
$PY bin/eval_distill_map.py "$TFEAT" --checkpoint "${R2}checkpoints/best.ckpt" \
  --name eval_P2B_feature_to_map --gpu "$GPU" --out "$OUT"

# ---- 2) ③前段 image_to_feature（画像→特徴map）----
echo "[2] ③前段 image_to_feature 学習開始 $(date +%H:%M:%S)"
$PY bin/train_distill_map.py "$TFEAT" --pseudo_dir "$PB" \
  --pattern image_to_feature --name P3B_front "${common[@]}"
FRONTR=$(ls -1dt "$OUT"/P3B_front_*/ | head -1)
FRONT="${FRONTR}checkpoints/best.ckpt"
echo "[2] 完了 $FRONT"

# ---- 3) 前段出力 F_S を precompute（③後段の高速化）----
echo "[3] F_S precompute $(date +%H:%M:%S)"
$PY bin/precompute_student_features.py --front_checkpoint "$FRONT" \
  --targets_dir "$TFEAT" --pseudo_dir "$PB" \
  --gpu "$GPU" --batch_size 64 --num_workers 8 --name P3B_FS --out "$FS_OUT"
FS=$(ls -1dt "$FS_OUT"/P3B_FS_*/ | head -1)
echo "[3] 完了 $FS"

# ---- 4) ③後段 feature_to_map on F_S ----
echo "[4] ③後段 学習開始 $(date +%H:%M:%S)"
$PY bin/train_distill_map.py "$FS" --pseudo_dir "${FS}pseudo" \
  --pattern feature_to_map --name P3B_back --patience 5 "${common[@]}"
BACKR=$(ls -1dt "$OUT"/P3B_back_*/ | head -1)
echo "[4] 完了 $BACKR"

# ---- 5) ③ 連結評価（前段＋後段）----
echo "[5] ③ 連結評価 $(date +%H:%M:%S)"
$PY bin/eval_distill_map.py "$TFEAT" --checkpoint "${BACKR}checkpoints/best.ckpt" \
  --front_checkpoint "$FRONT" --name eval_P3B_twostage --gpu "$GPU" --num_vis 8 --out "$OUT"

# ---- 5b) 表3行目「画像→特徴map」: 前段 F_S を Teacher メモリバンクに通す ----
echo "[5b] ③前段 → メモリバンク評価 $(date +%H:%M:%S)"
GPU="$GPU" OUT="$OUT" TFEAT="$TFEAT" FRONT="$FRONT" TEACHER="$TEACHER" \
  bash bin/run_pseudoB_row3.sh || echo "[5b] 失敗（後で個別実行）"

echo "全完了 $(date '+%m/%d %H:%M:%S')"
