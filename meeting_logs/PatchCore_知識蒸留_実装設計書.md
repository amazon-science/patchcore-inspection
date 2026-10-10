# PatchCore 知識蒸留 実装設計書

## ResNet18-FPN/Decoder Student による異常マップ・異常スコア蒸留

## 1. 目的

既存レポジトリに実装済みの PatchCore を Teacher とし，Student モデルへ知識蒸留を行う。

目的は，PatchCore の推論時に必要な

* 高次元特徴抽出
* memory bank との最近傍探索
* patch-level distance 計算
* anomaly map 生成

を Student モデルで近似し，**推論高速化・メモリ使用量削減・実運用容易化**を達成することである。

本実験では，Student モデルとして **ResNet18 + FPN/Decoder** を用い，以下の2パターンを実装・比較する。

---

## 2. 実験パターン

### Pattern A: Feature-to-AnomalyMap Distillation

PatchCore Teacher の中間特徴，または既存 backbone から得られる特徴マップを Student Decoder に入力し，異常マップを予測する。

```text
Input image
  ↓
Feature extractor
  ↓
Feature map
  ↓
Student FPN/Decoder
  ↓
Student anomaly map
  ↓
Student anomaly score
```

このパターンでは，Student は「特徴マップ → 異常マップ」の変換を学習する。

目的：

* PatchCore の最近傍探索部分を置き換える
* memory bank search を neural network head で近似する
* backbone は既存 PatchCore と共有または固定可能

---

### Pattern B: Image-to-AnomalyMap Distillation

入力画像から Student が直接異常マップを予測する。

```text
Input image
  ↓
ResNet18 backbone
  ↓
FPN/Decoder
  ↓
Student anomaly map
  ↓
Student anomaly score
```

このパターンでは，Student は「画像 → 異常マップ」の end-to-end 予測を学習する。

目的：

* PatchCore 全体を Student で置き換える
* 推論時に Teacher backbone と memory bank を不要にする
* 実運用向けの高速モデルを構築する

---

## 3. 既存 PatchCore 実装との整合性要件

Coding Agent は，既存レポジトリの構造を確認し，以下の既存コンポーネントを可能な限り再利用すること。

### 3.1 再利用対象

既存実装に以下が存在する場合は，新規実装せず再利用する。

* dataset loader
* image preprocessing
* train / test split
* PatchCore model class
* feature extractor
* memory bank
* anomaly map 生成処理
* anomaly score 計算処理
* evaluation metrics
* config system
* logging
* checkpoint save / load
* visualization utility

### 3.2 追加実装の基本方針

既存コードを破壊せず，以下のような拡張として実装する。

```text
models/
  patchcore/
    ...
  distillation/
    student_resnet18_fpn.py
    feature_to_map_student.py
    image_to_map_student.py

trainers/
  patchcore_trainer.py
  distillation_trainer.py

scripts/
  train_patchcore_distill.py
  eval_patchcore_distill.py

configs/
  distillation/
    resnet18_fpn_feature_to_map.yaml
    resnet18_fpn_image_to_map.yaml
```

既存のディレクトリ構造が異なる場合は，レポジトリの命名規則・設計思想に合わせて配置する。

---

## 4. Teacher: PatchCore

### 4.1 Teacher の役割

Teacher は，既存実装の PatchCore を使用する。

Teacher から取得する出力は以下とする。

| 出力                                    | 記号    | 用途                      |
| ------------------------------------- | ----- | ----------------------- |
| Teacher anomaly map                   | `A_T` | Student の異常マップ蒸留        |
| Teacher anomaly score                 | `s_T` | Student の画像スコア蒸留        |
| Patch-level nearest neighbor distance | `D_T` | 必要に応じて距離マップとして蒸留        |
| Intermediate feature map              | `F_T` | Pattern A の入力または特徴蒸留に使用 |

### 4.2 Teacher の固定

蒸留時，PatchCore Teacher は基本的に固定する。

```text
Teacher parameters: frozen
Student parameters: trainable
```

Teacher の memory bank も学習済み・構築済みのものを使用し，蒸留学習中には更新しない。

---

## 5. Student Model

## 5.1 共通方針

Student は，PatchCore が出力する異常マップを再現する dense prediction model とする。

分類モデルではなく，以下を出力する。

```text
Student output:
  - anomaly map A_S
  - anomaly score s_S
```

異常スコア `s_S` は，Student anomaly map `A_S` から算出する。

推奨：

```text
s_S = mean(top-k(A_S))
```

または

```text
s_S = max(A_S)
```

初期実装では `max` と `top-k mean` の両方を選択可能にする。

---

# 6. Pattern A: Feature-to-AnomalyMap Student

## 6.1 入力

Pattern A の Student 入力は，画像ではなく feature map とする。

候補：

```text
F_T = PatchCore backbone intermediate feature
```

または

```text
F = existing feature extractor output
```

PatchCore が複数層特徴を concat している場合，Student 入力もそれに合わせる。

例：

```text
F_T ∈ R^{C×H'×W'}
```

## 6.2 モデル構造

```text
Feature map F_T
  ↓
1×1 Conv for channel reduction
  ↓
FPN / Decoder
  ↓
3×3 Conv + ReLU
  ↓
1×1 Conv
  ↓
Student anomaly map A_S
```

実装名の例：

```python
FeatureToAnomalyMapStudent
```

### 6.3 出力

```text
A_S ∈ R^{1×H_map×W_map}
s_S ∈ R
```

`H_map, W_map` は Teacher anomaly map のサイズに合わせる。

Teacher map と解像度が異なる場合は，Student 出力または Teacher 出力を補間して合わせる。

推奨：

```text
A_T を Student 出力サイズに bilinear interpolation
```

または

```text
A_S を A_T サイズに bilinear interpolation
```

どちらか一方に統一する。

---

# 7. Pattern B: Image-to-AnomalyMap Student

## 7.1 入力

Pattern B の Student 入力は画像とする。

```text
x ∈ R^{3×H×W}
```

グレースケール画像の場合も，既存 PatchCore preprocessing に合わせる。

* 既存 PatchCore が 3ch 入力なら，グレースケールを 3ch replicate する
* normalize / resize / crop は Teacher と同一にする

## 7.2 モデル構造

Student は ResNet18 + FPN/Decoder とする。

```text
Input image
  ↓
ResNet18 backbone
  ├─ C2
  ├─ C3
  ├─ C4
  └─ C5
  ↓
FPN
  ├─ P2
  ├─ P3
  ├─ P4
  └─ P5
  ↓
Decoder / Prediction Head
  ↓
Student anomaly map A_S
  ↓
Pooling
  ↓
Student anomaly score s_S
```

### 7.3 Backbone

ResNet18 は ImageNet pretrained を基本とする。

設定で以下を選択可能にする。

```yaml
student:
  backbone: resnet18
  pretrained: true
  freeze_backbone: false
```

初期実験では `freeze_backbone: false` とし，end-to-end で学習する。

比較実験として，backbone freeze も選択可能にする。

---

# 8. FPN / Decoder 要件

## 8.1 FPN の役割

FPN は，ResNet18 の複数階層特徴を統合し，異常マップ予測に必要な空間情報を保持する。

使用する特徴層は以下を基本とする。

```text
C2, C3, C4, C5
```

入力画像が `224×224` の場合の例：

| Feature |  Size | Channels |
| ------- | ----: | -------: |
| C2      | 56×56 |       64 |
| C3      | 28×28 |      128 |
| C4      | 14×14 |      256 |
| C5      |   7×7 |      512 |

## 8.2 FPN 出力

FPN の出力チャンネルは config で指定する。

推奨初期値：

```yaml
fpn:
  out_channels: 128
```

## 8.3 Prediction Head

異常マップ head は以下を基本とする。

```text
Conv 3×3
ReLU
Conv 3×3
ReLU
Conv 1×1
```

出力チャンネルは1。

```text
A_S ∈ R^{1×H_map×W_map}
```

出力活性化は原則なし。

ただし，Teacher anomaly map を `[0, 1]` 正規化する場合は，`sigmoid` を使用可能にする。

---

# 9. Teacher Target の保存・取得

## 9.1 蒸留ターゲットの作成

蒸留学習時に毎回 Teacher PatchCore を forward してもよいが，計算コストが高いため，以下の2方式を選択可能にする。

### Option 1: On-the-fly Teacher

```text
Training loop 内で Teacher を実行して A_T, s_T を取得する
```

メリット：

* 実装が単純
* データ拡張と整合しやすい

デメリット：

* 学習が遅い

### Option 2: Precomputed Teacher Targets

事前に Teacher 出力を保存する。

保存対象：

```text
sample_id
image_path
A_T
s_T
D_T optional
F_T optional
label if available
mask if available
```

保存形式：

```text
.pt / .npz / .hdf5
```

推奨：

```text
precomputed_targets/{category}/{split}/{sample_id}.pt
```

含める内容：

```python
{
    "image_path": str,
    "teacher_anomaly_map": Tensor,
    "teacher_anomaly_score": Tensor,
    "teacher_distance_map": Tensor optional,
    "teacher_feature_map": Tensor optional,
    "label": int optional,
    "mask": Tensor optional
}
```

初期実装では，precompute script を追加する。

```text
scripts/precompute_patchcore_teacher_outputs.py
```

---

# 10. Loss 設計

## 10.1 基本 Loss

蒸留 Loss は以下とする。

```text
L = λ_map L_map(A_S, A_T)
  + λ_score L_score(s_S, s_T)
```

## 10.2 Anomaly Map Loss

基本は MSE または Smooth L1。

```text
L_map = MSE(A_S, A_T)
```

または

```text
L_map = SmoothL1(A_S, A_T)
```

config で選択可能にする。

```yaml
loss:
  map_loss: mse
```

## 10.3 Weighted Map Loss

異常領域が小さい場合，正常領域が支配的になるため，Teacher anomaly map に基づく重み付き loss を実装する。

```text
L_map = mean(w_i * |A_S(i) - A_T(i)|)
```

重み：

```text
w_i = 1 + α * normalize(A_T(i))
```

config：

```yaml
loss:
  use_weighted_map_loss: true
  anomaly_weight_alpha: 5.0
```

## 10.4 Score Loss

```text
L_score = MSE(s_S, s_T)
```

Teacher score のスケールが大きい場合，score normalization を行う。

```yaml
loss:
  normalize_teacher_score: true
```

## 10.5 Optional: Feature Distillation Loss

Pattern B では，必要に応じて Student の中間特徴と Teacher の特徴を近づける。

```text
L_feat = MSE(project(F_S), F_T)
```

最初の実験では必須ではない。

config で無効化可能にする。

```yaml
loss:
  use_feature_loss: false
```

## 10.6 Optional: Ranking Loss

画像レベルの異常順位を保存したい場合，Teacher score に基づく ranking loss を追加する。

```text
s_T(i) > s_T(j) なら s_S(i) > s_S(j)
```

初期実装では optional とする。

```yaml
loss:
  use_ranking_loss: false
```

---

# 11. Student Score 算出

Student anomaly map から image-level anomaly score を算出する。

実装すべき pooling：

```yaml
score_pooling:
  type: max
```

候補：

```text
max
topk_mean
mean
```

推奨：

```text
max
topk_mean
```

`topk_mean` の場合：

```yaml
score_pooling:
  type: topk_mean
  topk_ratio: 0.01
```

---

# 12. データ要件

## 12.1 必要なデータ種別

本実験で扱うデータは以下の3種類である。

| データ    | 必要性   | 用途                                              |
| ------ | ----- | ----------------------------------------------- |
| 正常画像   | 必須    | PatchCore Teacher の memory bank 構築，Student 蒸留学習 |
| 実異常画像  | 評価に必須 | 最終性能評価，AUROC/AUPRO計算                            |
| 疑似異常画像 | 推奨    | Student に異常パターンを学習させる，map loss の学習を強化           |

結論：

```text
最低限必要：
  - 正常画像
  - 実異常画像 test set

強く推奨：
  - 疑似異常画像
```

## 12.2 正常画像

正常画像は必須。

用途：

1. PatchCore Teacher の memory bank 構築
2. Teacher の正常分布学習
3. Student の training input
4. 正常画像に対する低異常スコア学習

PatchCore は通常，正常画像のみで memory bank を構築するため，正常画像がないと Teacher が成立しない。

## 12.3 実異常画像

実異常画像は，原則 test set のみに存在する前提とする。

用途：

1. 最終評価
2. image-level AUROC
3. pixel-level AUROC
4. AUPRO
5. qualitative visualization

注意：

```text
実異常画像は training には使用しない。
```

ただし，validation set として一部利用する設計にすると，純粋な異常検知評価として弱くなるため，初期設計では test only とする。

## 12.4 疑似異常画像

疑似異常画像は必須ではないが，強く推奨する。

理由：

正常画像のみで Teacher 出力を蒸留すると，Student は主に「正常画像に低い異常マップを出す」ことを学習する。
その結果，実異常画像に対して異常マップを強く出せない可能性がある。

疑似異常画像を用いることで，Student は

```text
正常画像 → 低異常マップ
疑似異常画像 → 高異常マップ
```

の対応を学習できる。

疑似異常画像に対しても Teacher PatchCore を実行し，Teacher anomaly map を蒸留ターゲットとして使用する。

## 12.5 推奨データ構成

```text
dataset/
  category/
    train/
      good/
        *.png

    pseudo/
      anomaly/
        *.png
      masks/
        *.png optional

    test/
      good/
        *.png
      anomaly_type_1/
        *.png
      anomaly_type_2/
        *.png

    ground_truth/
      anomaly_type_1/
        *.png
      anomaly_type_2/
        *.png
```

疑似異常 mask がある場合は保存する。
ただし，本実験の主ターゲットは Teacher anomaly map なので，pseudo mask は補助的に使う。

---

# 13. 疑似異常画像の生成要件

疑似異常画像を使用する場合，以下のいずれかを実装または既存実装を利用する。

候補：

1. CutPaste
2. DRAEM-style synthetic anomaly
3. Perlin noise anomaly
4. Texture blending
5. Random scratch / blob / spot
6. Domain-specific defect synthesis

半導体・外観検査画像では，以下が特に有効。

```text
- scratch
- spot
- particle
- contamination
- missing pattern
- local brightness change
- blur
- line break
```

実装では，pseudo anomaly generator を dataset wrapper として実装する。

```python
PseudoAnomalyDataset(base_normal_dataset)
```

出力：

```python
{
    "image": pseudo_anomaly_image,
    "pseudo_mask": pseudo_anomaly_mask,
    "source_image": original_normal_image,
    "label": 1,
    "is_pseudo": True
}
```

---

# 14. Training Protocol

## 14.1 Teacher Training

既存 PatchCore の手順で，正常 train 画像のみを使って Teacher を構築する。

```text
train/good
  ↓
PatchCore feature extraction
  ↓
memory bank construction
  ↓
coreset subsampling
  ↓
save PatchCore Teacher
```

## 14.2 Teacher Target Precompute

以下の画像に対して Teacher 出力を保存する。

```text
- train/good
- pseudo/anomaly
- validation/good optional
- test/good optional
- test/anomaly optional for analysis only
```

注意：

最終評価用の test/anomaly に対して Teacher 出力を保存してもよいが，Student の学習には使用しない。

## 14.3 Student Training

Student の学習データ：

```text
train/good + pseudo/anomaly
```

実異常 test は学習に使用しない。

Loss：

```text
L = λ_map L_map + λ_score L_score
```

初期値：

```yaml
loss:
  lambda_map: 1.0
  lambda_score: 1.0
```

異常マップ重視の場合：

```yaml
loss:
  lambda_map: 1.0
  lambda_score: 0.1
```

## 14.4 Validation

validation に実異常を使えない場合，以下で代用する。

```text
validation:
  - normal validation images
  - pseudo anomaly validation images
```

選択基準：

```text
- validation loss
- pseudo anomaly AUROC
- normal score stability
```

ただし，最終評価は必ず実異常 test set で行う。

---

# 15. Evaluation 要件

## 15.1 評価対象

以下を評価する。

1. Teacher PatchCore
2. Student Pattern A: Feature-to-AnomalyMap
3. Student Pattern B: Image-to-AnomalyMap

可能であれば，以下も比較する。

4. Student without pseudo anomalies
5. Student with pseudo anomalies
6. Student max pooling
7. Student top-k pooling

---

## 15.2 Image-level Metrics

画像単位の異常検知性能を評価する。

必須：

```text
Image-level AUROC
Image-level Average Precision
Image-level F1 max
```

推奨：

```text
Accuracy at optimal threshold
Precision
Recall
Specificity
```

しきい値は validation set で決定するか，test set 上で oracle threshold として明記する。

研究評価としては，主指標は AUROC とする。

---

## 15.3 Pixel-level Metrics

異常 mask がある場合，pixel-level 評価を行う。

必須：

```text
Pixel-level AUROC
Pixel-level Average Precision
Pixel-level F1 max
```

推奨：

```text
AUPRO
IoU at optimal threshold
Dice score
```

半導体欠陥のように異常領域が小さい場合，pixel AUROC だけでは過大評価されやすいため，AUPRO または pixel AP を重視する。

---

## 15.4 Localization Metrics

異常マップの局在性能を評価する。

使用可能な指標：

```text
AUPRO
PRO score
IoU
Dice
Pixel AP
```

優先度：

```text
1. AUPRO
2. Pixel AP
3. Pixel AUROC
4. Dice / IoU
```

---

## 15.5 Distillation Fidelity Metrics

Student が Teacher をどれだけ再現できているかを評価する。

対象：

```text
A_S vs A_T
s_S vs s_T
```

指標：

```text
Map MSE
Map MAE
Score MSE
Score MAE
Pearson correlation
Spearman correlation
```

特に，score ranking が重要なため Spearman correlation を推奨する。

---

## 15.6 Efficiency Metrics

蒸留の目的は高速化・軽量化なので，効率指標を必ず評価する。

必須：

```text
Inference time per image
FPS
Model parameters
GPU memory usage
```

推奨：

```text
FLOPs
Peak VRAM
CPU inference time
ONNX export compatibility
```

比較対象：

```text
PatchCore Teacher
ResNet18-FPN Student Pattern A
ResNet18-FPN Student Pattern B
```

PatchCore では memory bank search の時間も含める。

---

# 16. Visualization 要件

評価時に以下を保存する。

```text
input image
ground truth mask if available
Teacher anomaly map
Student anomaly map
absolute difference map |A_T - A_S|
overlay visualization
image-level scores
```

保存先：

```text
results/distillation/{experiment_name}/visualizations/
```

出力例：

```text
sample_0001_input.png
sample_0001_teacher_map.png
sample_0001_student_map.png
sample_0001_diff_map.png
sample_0001_overlay.png
```

---

# 17. Config 要件

## 17.1 共通 config

```yaml
experiment:
  name: resnet18_fpn_patchcore_distillation
  seed: 42
  device: cuda

dataset:
  name: mvtec_or_custom
  category: target_category
  image_size: 224
  train_split: train
  test_split: test
  use_pseudo_anomaly: true

teacher:
  type: patchcore
  checkpoint_path: path/to/patchcore.ckpt
  use_precomputed_targets: true
  target_dir: path/to/precomputed_targets

student:
  pattern: image_to_map
  backbone: resnet18
  pretrained: true
  freeze_backbone: false
  fpn_out_channels: 128
  decoder_channels: 128
  output_activation: none

score_pooling:
  type: topk_mean
  topk_ratio: 0.01

loss:
  map_loss: smooth_l1
  lambda_map: 1.0
  lambda_score: 0.1
  use_weighted_map_loss: true
  anomaly_weight_alpha: 5.0
  normalize_teacher_map: true
  normalize_teacher_score: true
  use_feature_loss: false

training:
  epochs: 100
  batch_size: 32
  optimizer: adamw
  learning_rate: 1.0e-4
  weight_decay: 1.0e-4
  scheduler: cosine
  num_workers: 4
  amp: true
  early_stopping: true
  patience: 20

evaluation:
  image_metrics:
    - auroc
    - average_precision
    - f1_max
  pixel_metrics:
    - auroc
    - average_precision
    - aupro
    - dice
    - iou
  efficiency_metrics:
    - inference_time
    - fps
    - parameters
    - gpu_memory
```

---

# 18. 実装すべき Script

## 18.1 Teacher target precompute

```bash
python scripts/precompute_patchcore_teacher_outputs.py \
  --config configs/distillation/resnet18_fpn_image_to_map.yaml
```

処理：

1. PatchCore checkpoint を load
2. 対象 dataset を iterate
3. `A_T`, `s_T`, optional `F_T`, `D_T` を計算
4. sample ごとに保存

## 18.2 Student training

```bash
python scripts/train_patchcore_distill.py \
  --config configs/distillation/resnet18_fpn_image_to_map.yaml
```

処理：

1. config load
2. dataset load
3. precomputed teacher target load
4. Student model build
5. distillation loss で training
6. checkpoint save

## 18.3 Student evaluation

```bash
python scripts/eval_patchcore_distill.py \
  --config configs/distillation/resnet18_fpn_image_to_map.yaml \
  --checkpoint path/to/student.ckpt
```

処理：

1. Student checkpoint load
2. test set inference
3. image-level metrics
4. pixel-level metrics
5. efficiency metrics
6. visualization save
7. csv / json summary save

---

# 19. 出力ファイル要件

各実験の結果は以下に保存する。

```text
results/distillation/{experiment_name}/
  checkpoints/
    best.ckpt
    last.ckpt

  metrics/
    image_metrics.json
    pixel_metrics.json
    efficiency_metrics.json
    distillation_fidelity.json

  predictions/
    scores.csv
    pixel_maps/

  visualizations/
    *.png

  config.yaml
  train_log.csv
```

`scores.csv` には最低限以下を含める。

```text
sample_id
image_path
label
teacher_score
student_score
anomaly_type
```

可能なら以下も含める。

```text
map_mse
map_mae
inference_time_ms
```

---

# 20. 比較実験設計

最低限，以下の4条件を比較する。

| ID | Pattern           | Training Data           | Purpose    |
| -- | ----------------- | ----------------------- | ---------- |
| T0 | PatchCore Teacher | normal only             | 上限・基準性能    |
| S1 | Feature-to-map    | normal only             | 最近傍探索の近似   |
| S2 | Image-to-map      | normal only             | 完全Student化 |
| S3 | Image-to-map      | normal + pseudo anomaly | 疑似異常の効果確認  |

可能なら追加：

| ID | Pattern        | Training Data                        | Purpose                |
| -- | -------------- | ------------------------------------ | ---------------------- |
| S4 | Feature-to-map | normal + pseudo anomaly              | feature-to-mapでの疑似異常効果 |
| S5 | Image-to-map   | normal + pseudo anomaly, top-k score | score pooling比較        |
| S6 | Image-to-map   | normal + pseudo anomaly, max score   | score pooling比較        |

---

# 21. 重要な実装注意点

## 21.1 Teacher と Student の前処理を揃える

以下は必ず一致させる。

```text
resize
crop
normalization
channel handling
image range
```

前処理がずれると，Teacher target と Student input の対応が崩れる。

## 21.2 Anomaly map の解像度を明示する

Teacher anomaly map と Student anomaly map の解像度が異なる場合，どちらに合わせるか config で明示する。

```yaml
target_map_size: teacher
```

または

```yaml
target_map_size: student
```

## 21.3 Test anomaly は学習に使わない

実異常画像が test set にしかない場合，それを Student training に使ってはいけない。

使用可能なのは evaluation のみ。

## 21.4 疑似異常は training 用

疑似異常画像は training / validation に使用してよい。

ただし，最終評価は実異常 test set で行う。

## 21.5 Teacher target の正規化

Teacher anomaly map / score のスケールは dataset category に依存する可能性がある。

以下を実装する。

```text
- no normalization
- min-max normalization
- z-score normalization
```

初期実験では map は min-max normalization，score は z-score または min-max を使用する。

---

# 22. データに関する最終方針

本研究で必要なデータは以下。

## 必須

```text
1. 正常画像 train set
2. 正常画像 test set
3. 実異常画像 test set
4. 実異常 mask
   ※ pixel-level / localization 評価を行う場合
```

## 強く推奨

```text
5. 疑似異常画像 train set
6. 疑似異常 mask
```

## 不要

```text
実異常画像 train set
```

実異常画像は test set のみでよい。
むしろ，実異常画像を training に使わないことで，異常検知タスクとしての評価が明確になる。

ただし，Student を正常画像のみで蒸留すると，異常領域を強く出力する学習信号が不足する可能性がある。
そのため，疑似異常画像を用いて Teacher PatchCore の anomaly map を蒸留する構成を推奨する。

---

# 23. 完了条件

Coding Agent は，以下を満たした時点で実装完了とする。

1. 既存 PatchCore Teacher を読み込める
2. Teacher anomaly map / score を precompute できる
3. Pattern A: Feature-to-AnomalyMap Student を学習できる
4. Pattern B: Image-to-AnomalyMap Student を学習できる
5. Student の anomaly map / score を推論できる
6. Teacher と Student の結果を同一 test set で比較できる
7. Image-level AUROC / AP / F1 を出力できる
8. Pixel-level AUROC / AP / AUPRO を出力できる
9. 推論時間・FPS・パラメータ数・GPUメモリを出力できる
10. anomaly map visualization を保存できる
11. 実験結果を json / csv で保存できる
12. 既存 PatchCore 実装を壊さず，config で蒸留実験を切り替えられる
