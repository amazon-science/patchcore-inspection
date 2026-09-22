#!/usr/bin/env bash
# 疑似異常「形状」バリアントの 1x スクリーニング（生成 → B方式教師 → ① 学習 → 評価）。
#
# 目的：疑似異常の *形* だけを差し替え、前回の B方式10x（AUROC 0.8215）と同じ後段工程で
# 精度が動くかを 1x（23,888枚）で先に判定する。後段（教師・学習・評価）は一切変えない。
#
# 使い方：
#   VARIANT=S2 SHAPE_ARGS="--blob_shape_family irregular" GPU=0 bash bin/run_shape_screen_1x.sh
#   VARIANT=S0 PSEUDO_DIR=<既存B pack> PSEUDO_MAX=23888 GPU=1 bash bin/run_shape_screen_1x.sh  # 対照(生成なし)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=./src
PY=.venv/bin/python

VARIANT="${VARIANT:?VARIANT を指定（S0/S2/S3 等）}"
GPU="${GPU:-0}"
SHAPE_ARGS="${SHAPE_ARGS:-}"
PER_IMAGE="${PER_IMAGE:-1}"                  # 1x = 正常1枚につき1枚
SEED="${SEED:-0}"
WORKERS="${WORKERS:-14}"
EPOCHS="${EPOCHS:-30}"
STEPS="${STEPS:-2000}"
BATCH="${BATCH:-32}"
ROOT="${ROOT:-/tmp/t-hori/bfai/shape_screen_1x}"          # 生成物はローカル nvme（NFS 乱読回避）
OUT="${OUT:-results/distill_map/shape_screen_1x}"
TEACHER="${TEACHER:-results/ContinualLearning/20260526_比較検討/20260527_064537_N1000_P5/round_21}"
TARGETS="${TARGETS:-results/distill_map/precomputed/20260622_222843_all}"
PSEUDO_DIR="${PSEUDO_DIR:-}"                 # 指定時は生成をスキップして既存 pack を使う
PSEUDO_MAX="${PSEUDO_MAX:-0}"

mkdir -p "$OUT" "$ROOT"
echo "=== 形状スクリーニング 1x: VARIANT=$VARIANT GPU=$GPU SHAPE_ARGS='$SHAPE_ARGS' ==="

if [ -z "$PSEUDO_DIR" ]; then
  # ---- 1) 疑似異常画像の生成（形のみ差し替え。色・サイズ・個数は全条件共通）----
  echo "[1] 疑似異常生成 $(date +%H:%M:%S)"
  $PY scripts/generate_pseudo_anomalies.py --per_image "$PER_IMAGE" --pack --types blob \
      --seed "$SEED" --workers "$WORKERS" --out "$ROOT/$VARIANT" $SHAPE_ARGS
  GEN=$(ls -1dt "$ROOT/$VARIANT"/*/ | head -1)
  PACK="${GEN}pseudo"
  echo "[1] 完了 $PACK"

  # ---- 2) B方式の正解（PatchCore 距離マップ）----
  # faiss は可視デバイスの 0 番に載るため CUDA_VISIBLE_DEVICES で GPU を選び --gpu 0 を渡す
  echo "[2] B方式 教師ターゲット生成 $(date +%H:%M:%S)"
  BDIR="$ROOT/${VARIANT}_B"
  CUDA_VISIBLE_DEVICES="$GPU" $PY scripts/make_pseudo_targets_patchcore.py "$TEACHER" \
      --pack "$PACK" --out "$BDIR" --gpu 0 --batch_size 16
  PSEUDO_DIR="$BDIR"
  echo "[2] 完了 $PSEUDO_DIR"
fi

# ---- 3) ① image_to_map の学習（条件は B方式10x と同一）----
echo "[3] ① 学習 $(date +%H:%M:%S)"
extra=()
[ "$PSEUDO_MAX" -gt 0 ] && extra+=(--pseudo_max "$PSEUDO_MAX")
$PY bin/train_distill_map.py "$TARGETS" --pseudo_dir "$PSEUDO_DIR" \
    --pattern image_to_map --name "${VARIANT}_1x_image_to_map" --gpu "$GPU" \
    --epochs "$EPOCHS" --batch_size "$BATCH" --normal_pseudo_ratio 1.0 \
    --steps_per_epoch "$STEPS" --out "$OUT" "${extra[@]}"
RUN=$(ls -1dt "$OUT/${VARIANT}_1x_image_to_map"_*/ | head -1)
echo "[3] 完了 $RUN"

# ---- 4) 評価（前回と同一の評価セット）----
echo "[4] 評価 $(date +%H:%M:%S)"
$PY bin/eval_distill_map.py "$TARGETS" --checkpoint "${RUN}checkpoints/best.ckpt" \
    --name "eval_${VARIANT}_1x" --gpu "$GPU" --out "$OUT"
echo "=== $VARIANT 完了 $(date +%H:%M:%S) ==="
