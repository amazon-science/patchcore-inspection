#!/usr/bin/env bash
# 疑似異常B方式(10x)で ②(feature_to_map) と ③(二段) を順に学習・評価する。
#
# ①(image_to_map) は別途実行済み（AUROC 0.822）。本スクリプトは②③を連結し、
# 定量評価表の 238,880枚 行（特徴map評価・異常map評価・AUROC/AP/F1max）を埋める。
#
# 前提:
#   - B方式 疑似pack（images+targets(PatchCore距離)+features）が生成済み
#   - RAM 逼迫のため正常特徴は RAM 常駐せず(--cache_normal 0)、ローカル nvme から読む
#     （NFS ランダム読みが遅いため、正常特徴 47GB を nvme へコピーしてから使う）
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=./src
PY=.venv/bin/python

GPU="${GPU:-1}"
EPOCHS="${EPOCHS:-30}"
STEPS="${STEPS:-2000}"
OUT="${OUT:-results/distill_map/pseudoB_10x}"
PB="${PB:-/disk/data_area4/t-hori/bfai/precomputed_pseudoB_feat_10x_20260724_234238}"
TFEAT_SRC="${TFEAT_SRC:-results/distill_map/precomputed_allfeat/20260623_042006}"
LOCAL_ROOT="${LOCAL_ROOT:-/tmp/t-hori/bfai/allfeat_local}"
FS_OUT="${FS_OUT:-/disk/data_area4/t-hori/bfai/student_feat}"
TEACHER="${TEACHER:-results/ContinualLearning/20260526_比較検討/20260527_064537_N1000_P5/round_21}"

mkdir -p "$OUT"

# ---- 0) 正常/評価の特徴(47GB)を nvme へコピー（既にあればスキップ）----
TFEAT="$LOCAL_ROOT/$(basename "$TFEAT_SRC")"
if [[ ! -f "$TFEAT/stats.json" ]]; then
  echo "[0] 正常特徴を nvme へコピー: $TFEAT_SRC → $TFEAT"
  mkdir -p "$TFEAT"
  cp -r "$TFEAT_SRC"/. "$TFEAT"/
  echo "[0] コピー完了: $(du -sh "$TFEAT" | cut -f1)"
else
  echo "[0] コピー済みを再利用: $TFEAT"
fi

latest() { ls -1dt "$1"/ 2>/dev/null | head -1; }

# ---- 1) ② feature_to_map ----
echo "[1] ② feature_to_map 学習開始 $(date +%H:%M:%S)"
$PY bin/train_distill_map.py "$TFEAT" \
  --pseudo_dir "$PB" --pattern feature_to_map --name P2B_feature_to_map \
  --gpu "$GPU" --epochs "$EPOCHS" --batch_size 32 \
  --normal_pseudo_ratio 1.0 --steps_per_epoch "$STEPS" \
  --cache_normal 0 --out "$OUT"
R2=$(latest "$OUT/P2B_feature_to_map_*")
echo "[1] ② 評価 $(date +%H:%M:%S)  run=$R2"
$PY bin/eval_distill_map.py "$TFEAT" \
  --checkpoint "${R2}checkpoints/best.ckpt" \
  --name eval_P2B_feature_to_map --gpu "$GPU" --out "$OUT"

# ---- 2) ③前段 image_to_feature（特徴map評価の列）----
echo "[2] ③前段 image_to_feature 学習開始 $(date +%H:%M:%S)"
$PY bin/train_distill_map.py "$TFEAT" \
  --pseudo_dir "$PB" --pattern image_to_feature --name P3B_front \
  --gpu "$GPU" --epochs "$EPOCHS" --batch_size 32 \
  --normal_pseudo_ratio 1.0 --steps_per_epoch "$STEPS" \
  --cache_normal 0 --out "$OUT"
RF=$(latest "$OUT/P3B_front_*")
FRONT="${RF}checkpoints/best.ckpt"
# 表3行目「画像→特徴map」= 異常mapを出さない行。特徴map評価(feat_MSE/MAE)は前段の
# train_log.csv 最終行(val feat mse/mae)、推論時間/メモリは最終段の bench_networks で取る。

# ---- 3) 前段出力 F_S を一括計算（大容量: data_area4）----
echo "[3] F_S precompute 開始 $(date +%H:%M:%S)"
$PY bin/precompute_student_features.py \
  --front_checkpoint "$FRONT" --targets_dir "$TFEAT" --pseudo_dir "$PB" \
  --gpu "$GPU" --batch_size 64 --num_workers 8 --name P3B_FS --out "$FS_OUT"
FS=$(latest "$FS_OUT/P3B_FS_*")

# ---- 4) ③後段 F_S→異常map を再学習（③B）----
echo "[4] ③後段 学習開始 $(date +%H:%M:%S)  FS=$FS"
$PY bin/train_distill_map.py "$FS" \
  --pseudo_dir "${FS}pseudo" --pattern feature_to_map --name P3B_back \
  --gpu "$GPU" --epochs "$EPOCHS" --patience 5 --batch_size 32 \
  --normal_pseudo_ratio 1.0 --steps_per_epoch "$STEPS" \
  --cache_normal 0 --out "$OUT"
RB=$(latest "$OUT/P3B_back_*")

# ---- 5) ③ 連結評価（前段+後段）----
echo "[5] ③ 連結評価 $(date +%H:%M:%S)"
$PY bin/eval_distill_map.py "$TFEAT" \
  --checkpoint "${RB}checkpoints/best.ckpt" --front_checkpoint "$FRONT" \
  --name eval_P3B_twostage --gpu "$GPU" --num_vis 8 --out "$OUT"

# ---- 6) 効率ベンチ（表の推論時間/メモリ列。①②③前段③後段をまとめて1回）----
echo "[6] 効率ベンチ $(date +%H:%M:%S)"
R1=$(latest "$OUT/P1B_image_to_map_*")
$PY bin/bench_networks.py --teacher "$TEACHER" \
  --p2 "${R1}checkpoints/best.ckpt" \
  --p4 "${R2}checkpoints/best.ckpt" \
  --p5_front "$FRONT" --p5b_back "${RB}checkpoints/best.ckpt" \
  --gpu "$GPU" --name bench_pseudoB --out "$OUT" \
  || echo "[6] bench 失敗（後で個別取得）"

echo "完了 $(date +%H:%M:%S)  出力: $OUT"
