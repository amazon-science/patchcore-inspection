datapath=/path/to/data/from/mvtec
datasets=('bottle'  'cable'  'capsule'  'carpet'  'grid'  'hazelnut' 'leather'  'metal_nut'  'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

############# Detection
### IM224:
# Baseline: Backbone: WR50, Blocks: 2 & 3, Coreset Percentage: 10%, Embedding Dimensionalities: 1024 > 1024, neighbourhood aggr. size: 3, neighbours: 1, seed: 0
# Performance: Instance AUROC: 0.992, Pixelwise AUROC: 0.981, PRO: 0.944
python bin/run_patchcore.py --gpu 0 --seed 0 --save_patchcore_model --log_group IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0 --log_online --log_project MVTecAD_Results results \
patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 3 sampler -p 0.1 approx_greedy_coreset dataset --resize 256 --imagesize 224 "${dataset_flags[@]}" mvtec $datapath

# Ensemble: Backbone: WR101 / ResNext101/ DenseNet201, Blocks: 2 & 3, Coreset Percentage: 1%, Embedding Dimensionalities: 1024 > 384, neighbourhood aggr. size: 3, neighbours: 1, seed: 3
# Performance: Instance AUROC: 0.993, Pixelwise AUROC: 0.981, PRO: 0.942
python bin/run_patchcore.py --gpu 0 --seed 3 --save_patchcore_model --log_group IM224_Ensemble_L2-3_P001_D1024-384_PS-3_AN-1_S3 --log_online --log_project MVTecAD_Results results \
patch_core -b wideresnet101 -b resnext101 -b densenet201 -le 0.layer2 -le 0.layer3 -le 1.layer2 -le 1.layer3 -le 2.features.denseblock2 -le 2.features.denseblock3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 384 --anomaly_scorer_num_nn 1 --patchsize 3 sampler -p 0.01 approx_greedy_coreset dataset --resize 256 --imagesize 224 "${dataset_flags[@]}" mvtec $datapath


### IM320:
# Baseline: Backbone: WR50, Blocks: 2 & 3, Coreset Percentage: 1%, Embedding Dimensionalities: 1024 > 1024, neighbourhood aggr. size: 3, neighbours: 1, seed: 22
# Performance: Instance AUROC: 0.993, Pixelwise AUROC: 0.978, PRO: 0.943
python bin/run_patchcore.py --gpu 0 --seed 22 --save_patchcore_model --log_group IM320_WR50_L2-3_P001_D1024-1024_PS-3_AN-1_S22 --log_online --log_project MVTecAD_Results results \
patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 3 sampler -p 0.01 approx_greedy_coreset dataset --resize 366 --imagesize 320 "${dataset_flags[@]}" mvtec $datapath

# Ensemble: Backbone: WR101 / ResNext101/ DenseNet201, Blocks: 2 & 3, Coreset Percentage: 1%, Embedding Dimensionalities: 1024 > 384, neighbourhood aggr. size: 3, neighbours: 1, seed: 40
# Performance: Instance AUROC: 0.996, Pixelwise AUROC: 0.982, PRO: 0.949
python bin/run_patchcore.py --gpu 0 --seed 40 --save_patchcore_model --log_group IM320_Ensemble_L2-3_P001_D1024-384_PS-3_AN-1_S40 --log_online --log_project MVTecAD_Results results \
patch_core -b wideresnet101 -b resnext101 -b densenet201 -le 0.layer2 -le 0.layer3 -le 1.layer2 -le 1.layer3 -le 2.features.denseblock2 -le 2.features.denseblock3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 384 --anomaly_scorer_num_nn 1 --patchsize 3 sampler -p 0.01 approx_greedy_coreset dataset --resize 366 --imagesize 320 "${dataset_flags[@]}" mvtec $datapath


############# Segmentation
### IM320
# Baseline: Backbone: WR50, Blocks: 2 & 3, Coreset Percentage: 1%, Embedding Dimensionalities: 1024 > 1024, neighbourhood aggr. size: 5, neighbours: 3, seed: 39
# Performance: Instance AUROC: 0.99, Pixelwise AUROC: 0.984, PRO: 0.942
python bin/run_patchcore.py --gpu 0 --seed 22 --save_patchcore_model --log_group IM320_WR50_L2-3_P001_D1024-1024_PS-5_AN-3_S39 --log_online --log_project MVTecAD_Results results \
patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 1024 --anomaly_scorer_num_nn 3 --patchsize 5 sampler -p 0.01 approx_greedy_coreset dataset --resize 366 --imagesize 320 "${dataset_flags[@]}" mvtec $datapath

# Ensemble: Backbone: WR101 / ResNext101/ DenseNet201, Blocks: 2 & 3, Coreset Percentage: 1%, Embedding Dimensionalities: 1024 > 384, neighbourhood aggr. size: 5, neighbours: 5, seed: 88
# Performance: Instance AUROC: 0.996, Pixelwise AUROC: 0.982, PRO: 0.949
python bin/run_patchcore.py --gpu 0 --seed 88 --save_patchcore_model --log_group IM320_Ensemble_L2-3_P001_D1024-384_PS-5_AN-5_S88 --log_online --log_project MVTecAD_Results results \
patch_core -b wideresnet101 -b resnext101 -b densenet201 -le 0.layer2 -le 0.layer3 -le 1.layer2 -le 1.layer3 -le 2.features.denseblock2 -le 2.features.denseblock3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 384 --anomaly_scorer_num_nn 5 --patchsize 5 sampler -p 0.01 approx_greedy_coreset dataset --resize 366 --imagesize 320 "${dataset_flags[@]}" mvtec $datapath