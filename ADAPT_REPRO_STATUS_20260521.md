# ADAPT Reproduction Status - 2026-05-21

## Scope

This folder is for reproducing `jxbbb/ADAPT` on the remote Windows host through WSL/Linux, using the existing single RTX 4090 instead of the paper/repo's 4-GPU setup.

Remote repo:

```text
E:\sbw\ADAPT_repro\ADAPT
```

Preprocessed data:

```text
E:\sbw\ADAPT_repro\ADAPT\ADAPT_PREPROCESSED_DATASET
```

WSL distro and environment:

```text
ADAPT-Ubuntu
conda env: adapt
```

## Official Reference

Official repo: <https://github.com/jxbbb/ADAPT>

The official README states:

- ADAPT supports Linux with NVIDIA GPUs.
- Preprocessed BDDX can be downloaded from the Baidu link, or raw BDD-X can be preprocessed with `src/prepro`.
- Released checkpoints belong under `checkpoints`.
- Base Video-Swin weights belong under `models/video_swin_transformer`.
- Basic training uses `sh scripts/BDDX_multitask.sh`.

The official training script in this repo uses:

```text
CUDA_VISIBLE_DEVICES=4,5,6,7
--nproc_per_node=4
--per_gpu_train_batch_size 4
--gradient_accumulation_steps 4
--num_train_epochs 40
--train_yaml BDDX/training_32frames.yaml
--val_yaml BDDX/testing_32frames.yaml
```

Effective training batch in the official command is:

```text
4 GPUs * 4 per GPU * 4 accumulation = 64
```

## Downloaded Dataset Validation

The downloaded preprocessed package is present and usable.

Important local README content from the downloaded dataset:

```text
datasets is identical to BDDX.
datasets_part excludes data without control signals.
Training uses datasets_part training set.
Evaluation uses datasets full test set for fair comparison with SOTA.
```

Therefore both folders are needed:

- `datasets_part` is used for multitask training because ADAPT predicts caption plus control signals.
- `datasets` is used for full BDDX test evaluation because some full-test samples lack control signals but caption evaluation remains valid.

Validated counts:

| Root | Train | Val | Test |
|---|---:|---:|---:|
| `datasets/BDDX` | 21143 | 2519 | 2859 |
| `datasets_part/BDDX` | 16392 | 1914 | 2123 |
| `datasets/BDDX_des` | 21143 | 2519 | 2859 |
| `datasets/BDDX_exp` | 21143 | 2519 | 2859 |
| `datasets_part/BDDX_des` | 16392 | 1914 | 2123 |
| `datasets_part/BDDX_exp` | 16392 | 1914 | 2123 |

Large frame TSVs are present, including:

```text
datasets/BDDX/frame_tsv/training_32frames_img_size256.img.tsv
datasets/BDDX/frame_tsv/testing_32frames_img_size256.img.tsv
datasets_part/BDDX/frame_tsv_part/training_32frames_img_size256.img.tsv
```

Validation output:

```text
E:\sbw\ADAPT_repro\ADAPT\repro_logs\adapt_preprocessed_dataset_validation.json
```

## Path Layout Fixed

Repo-root links now point to the downloaded data:

```text
datasets\BDDX      -> ADAPT_PREPROCESSED_DATASET\datasets\BDDX
datasets\BDDX_des  -> ADAPT_PREPROCESSED_DATASET\datasets\BDDX_des
datasets\BDDX_exp  -> ADAPT_PREPROCESSED_DATASET\datasets\BDDX_exp
datasets_part      -> ADAPT_PREPROCESSED_DATASET\datasets_part
```

Model/checkpoint links are also present:

```text
models      -> F:\sbw_adapt_assets\models
checkpoints -> F:\sbw_adapt_assets\checkpoints
```

Available assets:

```text
models/captioning/bert-base-uncased/*
models/video_swin_transformer/swin_base_patch244_window877_kinetics600_22k.pth
checkpoints/basemodel/checkpoints/model.bin
checkpoints/basemodel/checkpoints/optmizer_state.bin
```

## Environment Validation

Validated inside WSL:

```text
torch 1.13.1+cu117
CUDA 11.7
GPU: NVIDIA GeForce RTX 4090
apex with legacy amp import available
sklearn 1.3.2
java available at /usr/bin/java
```

Fixes already applied:

- Replaced incompatible PyPI `apex` path with legacy NVIDIA Apex that provides `apex.amp`.
- Added WSL/conda CUDA library path activation hook so `libcuda.so` and cuDNN load.
- Installed Java runtime for COCO caption eval.
- Installed scikit-learn for signal metrics.

Known non-blocking warning:

```text
azureml.core warns about packaging>=26 with packaging<22 requirement.
```

This only disables AzureML online run context and logs offline; it did not block training/evaluation startup.

## Single-GPU Scripts Added

Scripts added under:

```text
E:\sbw\ADAPT_repro\ADAPT\scripts
```

Files:

```text
ADAPT_single_gpu_multitask.sh
ADAPT_single_gpu_eval.sh
ADAPT_single_gpu_smoke.sh
ADAPT_single_gpu_smoke_timeout.sh
ADAPT_single_gpu_eval_smoke_timeout.sh
ADAPT_single_gpu_fullscript_start_smoke.sh
```

Main single-GPU training mapping:

```text
official: 4 GPUs * batch 4 * accum 4 = effective batch 64
single GPU: 1 GPU * batch 2 * accum 32 = effective batch 64
```

The single-GPU command uses `--data_dir datasets_part`, because the downloaded dataset README says training uses `datasets_part`.

The validation YAML still resolves to full `datasets/BDDX/testing_*`, because `datasets_part/BDDX/testing_32frames.yaml` points to `../../datasets/BDDX/...`.

## Smoke Results

Training smoke reached real batch execution.

Evidence from:

```text
repro_logs/single_gpu_smoke.log
```

Observed batch tensors:

```text
input_ids      = torch.Size([1, 30])
attention_mask = torch.Size([1, 814, 814])
token_type_ids = torch.Size([1, 30])
img_feats      = torch.Size([1, 32, 3, 224, 224])
masked_pos     = torch.Size([1, 30])
masked_ids     = torch.Size([1, 45])
car_info       = torch.Size([1, 2, 32])
```

The first smoke continued past 1000 iterations because ADAPT's `--limited_samples` does not reduce `max_iter`. It was stopped intentionally and replaced with timeout smoke wrappers.

Pretrained checkpoint eval smoke:

```text
repro_logs/single_gpu_pretrained_eval_smoke_timeout.log
```

This loaded:

```text
checkpoints/basemodel/checkpoints/model.bin
```

and entered the test loop, reaching eval iteration 11 before the intentional 120-second timeout.

Full single-GPU training-script startup smoke:

```text
repro_logs/single_gpu_fullscript_start_smoke.log
```

This confirmed the main single-GPU script starts with:

```text
Train with 2 images per GPU.
Total training steps 8196 for one epoch.
input_ids      = torch.Size([2, 30])
attention_mask = torch.Size([2, 814, 814])
img_feats      = torch.Size([2, 32, 3, 224, 224])
car_info       = torch.Size([2, 2, 32])
```

No OOM occurred. The run was stopped by timeout, so the final `SIGTERM` in the log is expected.

## Current Reproduction Status

Complete:

- Data is downloaded and validated.
- `datasets` vs `datasets_part` behavior is understood and wired correctly.
- WSL/Linux runtime is configured.
- GPU, Torch, Apex, cuDNN, Java, sklearn are validated.
- Released BERT, Video-Swin and basemodel checkpoint assets are present.
- Single-GPU training and evaluation scripts are added.
- Training dataloader, model forward, first batch, Deepspeed startup, checkpoint loading, and eval loop startup have all been smoke-verified.

Not yet complete:

- Full 40-epoch single-GPU training has not been completed.
- Full official checkpoint evaluation over all 2859 test samples has not been completed in this session.

Reason:

```text
Official reproduction expects 4 V100-class GPUs.
Single RTX 4090 can run it, but full training is long.
The start smoke suggests one epoch has 8196 microsteps and full 40-epoch training will likely take multiple days on one GPU.
```

