#!/usr/bin/env bash
# 疑似異常「不定形＋境界ぼかし」10x での ②③ 残り工程を通しで実行する。
#
# 前提：② feature_to_map と ③前段 image_to_feature の学習が実行中（または完了済）。
# 流れ：②評価 → F_T を data_area4 へ退避（nvme 確保）→ F_S precompute → ③後段
#       → ③連結評価 → 画像→特徴map 行（bank 評価）
# 退避に mv を使うのは、移動と同時に nvme が空き、F_T も保全できるため（rm 不要）。
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=./src
PY=.venv/bin/python

OUT="${OUT:-results/distill_map/shape_10x_23}"
TFEAT="${TFEAT:-/tmp/t-hori/bfai/allfeat_local/20260623_042006}"
PBF="${PBF:-/tmp/t-hori/bfai/shape_10x/S3_10x_feat}"      # 疑似 F_T（退避対象）
PB="${PB:-/tmp/t-hori/bfai/shape_10x/S3_10x_B}"           # 疑似 packed（画像・教師マップ）
FS_OUT="${FS_OUT:-/tmp/t-hori/bfai/shape_10x/FS}"
ARCHIVE="${ARCHIVE:-/disk/data_area4/t-hori/bfai/S3_10x_feat_soft}"
TEACHER="${TEACHER:-results/ContinualLearning/20260526_比較検討/20260527_064537_N1000_P5/round_21}"
common=(--epochs 30 --batch_size 32 --normal_pseudo_ratio 1.0 --steps_per_epoch 2000 --cache_normal 0 --out "$OUT")

echo "=== stage3 開始 $(date '+%m/%d %H:%M:%S') ==="
# ---- 0) ②・③前段の学習完了を待つ（学習末尾の "best monitor=" を検知）----
for f in run_P2.log run_P3front.log; do
  until grep -q "best monitor=" "$OUT/$f" 2>/dev/null; do sleep 120; done
  echo "[0] $f 学習完了 $(date +%H:%M:%S)"
done

# ---- 1) ② 評価 ----
R2=$(ls -1dt "$OUT"/P2_soft10x_*/ | head -1)
echo "[1] ② 評価 $(date +%H:%M:%S)  $R2"
$PY bin/eval_distill_map.py "$TFEAT" --checkpoint "${R2}checkpoints/best.ckpt" \
  --name eval_P2_soft10x --gpu 1 --out "$OUT" >> "$OUT/run_P2.log" 2>&1
echo "[1] 完了"

# ---- 2) F_T を data_area4 へ退避（mv で移動＝nvme が空く。F_S 用の領域を作る）----
echo "[2] F_T 退避 $(date +%H:%M:%S)  → $ARCHIVE"
mkdir -p "$(dirname "$ARCHIVE")"
mv "$PBF" "$ARCHIVE"
df -h /tmp | tail -1

# ---- 3) F_S precompute（③後段の高速化）----
FRONTR=$(ls -1dt "$OUT"/P3front_soft10x_*/ | head -1)
FRONT="${FRONTR}checkpoints/best.ckpt"
echo "[3] F_S precompute $(date +%H:%M:%S)  front=$FRONT"
$PY bin/precompute_student_features.py --front_checkpoint "$FRONT" \
  --targets_dir "$TFEAT" --pseudo_dir "$PB" \
  --gpu 0 --batch_size 64 --num_workers 8 --name FS_soft10x --out "$FS_OUT" > "$OUT/run_FS.log" 2>&1
FS=$(ls -1dt "$FS_OUT"/FS_soft10x_*/ | head -1)
echo "[3] 完了 $FS"

# ---- 4) ③後段 feature_to_map on F_S ----
echo "[4] ③後段 学習 $(date +%H:%M:%S)"
$PY bin/train_distill_map.py "$FS" --pseudo_dir "${FS}pseudo" \
  --pattern feature_to_map --name P3back_soft10x --patience 5 --gpu 0 "${common[@]}" > "$OUT/run_P3back.log" 2>&1
BACKR=$(ls -1dt "$OUT"/P3back_soft10x_*/ | head -1)
echo "[4] 完了 $BACKR"

# ---- 5) ③ 連結評価（前段＋後段）----
echo "[5] ③ 連結評価 $(date +%H:%M:%S)"
$PY bin/eval_distill_map.py "$TFEAT" --checkpoint "${BACKR}checkpoints/best.ckpt" \
  --front_checkpoint "$FRONT" --name eval_P3_twostage_soft10x --gpu 0 --num_vis 8 \
  --out "$OUT" >> "$OUT/run_P3back.log" 2>&1

# ---- 6) 画像→特徴map 行（前段 F_S を教師メモリバンクへ。異常map評価込み）----
echo "[6] bank 評価 $(date +%H:%M:%S)"
CUDA_VISIBLE_DEVICES=1 $PY bin/eval_front_with_bank.py "$TFEAT" \
  --front_checkpoint "$FRONT" --teacher "$TEACHER" --name eval_front_bank_soft10x \
  --gpu 0 --batch_size 8 --num_workers 4 --out "$OUT" > "$OUT/run_bank.log" 2>&1

echo "=== stage3 完了 $(date '+%m/%d %H:%M:%S') ==="
