# ADAPT Findings and Remaining Gaps - 2026-05-21

## Main Finding

The downloaded `ADAPT_PREPROCESSED_DATASET` is the correct kind of package for reproducing ADAPT, and the repo has now been wired so ADAPT sees the expected official layout.

The important data distinction is:

```text
datasets      = full BDDX-style processed data
datasets_part = subset excluding missing control signals
```

For ADAPT multitask training, both are used:

```text
training: datasets_part/BDDX/training_32frames.yaml
testing:  datasets/BDDX/testing_32frames.yaml
```

This matches the downloaded dataset README, which states that training uses `datasets_part`, while evaluation uses full `datasets` for fair SOTA comparison.

## What Was Fixed

### Path wiring

The repo now has direct links from expected ADAPT paths to the downloaded dataset:

```text
datasets\BDDX
datasets\BDDX_des
datasets\BDDX_exp
datasets_part
```

This avoids editing ADAPT internals for dataset paths.

### Linux runtime

ADAPT is Linux-only in the official README. The current Windows machine can run it through:

```text
WSL distro: ADAPT-Ubuntu
Conda env: adapt
```

The following blockers were fixed:

```text
legacy apex.amp missing
libcuda.so / cuDNN library path in WSL
Java missing for COCO eval
scikit-learn missing for signal metrics
```

### Single-GPU adaptation

Official 4-GPU command was adapted to a single RTX 4090 by:

```text
--nproc_per_node=1
CUDA_VISIBLE_DEVICES=0
per_gpu_train_batch_size=2
gradient_accumulation_steps=32
```

This preserves official effective batch size 64.

## Verified Evidence

### Training dataloader and batch

`single_gpu_fullscript_start_smoke.log` shows:

```text
Train with 2 images per GPU.
Total training steps 8196
input_ids = torch.Size([2, 30])
attention_mask = torch.Size([2, 814, 814])
img_feats = torch.Size([2, 32, 3, 224, 224])
car_info = torch.Size([2, 2, 32])
```

This confirms the repo loads:

```text
datasets_part/BDDX/training_32frames.yaml
datasets_part/BDDX/frame_tsv_part/training_32frames_img_size256.img.tsv
```

### Eval loop and checkpoint load

`single_gpu_pretrained_eval_smoke_timeout.log` shows:

```text
Loading state dict from checkpoint checkpoints/basemodel/checkpoints/model.bin
yaml_file: BDDX/testing_32frames.yaml
loading datasets/BDDX/testing.caption.tsv
loading datasets/BDDX/frame_tsv/testing_32frames_img_size256.img.tsv
eval loop reached iteration 11 before timeout
```

### No active runaway process

`remote_stop_adapt_smoke.ps1` was run after smoke tests. It reported no remaining ADAPT `run_adapt.py` / `torch.distributed.launch` processes.

## Remaining Gaps

### 1. Full 40-epoch training is not completed

The single-GPU setup is runnable, but full reproduction training is not yet done.

Based on smoke logs:

```text
one epoch: 8196 microsteps
official epochs: 40
eval is triggered during training
```

A full 40-epoch run on one RTX 4090 is expected to be very long compared with the official 4-GPU setup.

### 2. Full checkpoint evaluation is not completed

The released checkpoint loads and starts eval, but full test evaluation over all 2859 full test samples was not completed in the smoke phase.

To complete it:

```bash
CUDA_VISIBLE_DEVICES=0 \
EVAL_DIR=checkpoints/basemodel/checkpoints/ \
DATA_DIR=datasets \
VAL_YAML=BDDX/testing_32frames.yaml \
PER_GPU_EVAL_BATCH_SIZE=4 \
bash scripts/ADAPT_single_gpu_eval.sh
```

### 3. Exact paper speed cannot be matched on one GPU

This is expected. The official script uses four GPUs. The single-GPU reproduction preserves effective batch size through gradient accumulation, but wall-clock runtime will be slower.

### 4. Optional batch-size speed test remains

The safe script uses:

```text
batch 2, accumulation 32
```

RTX 4090 may be able to run:

```text
batch 4, accumulation 16
```

but this has not been validated. It could reduce overhead but may OOM because ADAPT uses 32 frames, Video-Swin base, BERT, fp16, and deepspeed.

## Files Added

Scripts:

```text
scripts/ADAPT_single_gpu_multitask.sh
scripts/ADAPT_single_gpu_eval.sh
scripts/ADAPT_single_gpu_smoke.sh
scripts/ADAPT_single_gpu_smoke_timeout.sh
scripts/ADAPT_single_gpu_eval_smoke_timeout.sh
scripts/ADAPT_single_gpu_fullscript_start_smoke.sh
```

Remote tools:

```text
repro_tools/remote_adapt_dataset_validate.py
repro_tools/remote_adapt_status.ps1
repro_tools/remote_adapt_verify_detail.ps1
repro_tools/remote_chmod_adapt_single_gpu.ps1
repro_tools/remote_run_adapt_eval_smoke_timeout.ps1
repro_tools/remote_run_adapt_fullscript_start_smoke.ps1
repro_tools/remote_stop_adapt_smoke.ps1
```

Logs:

```text
repro_logs/adapt_preprocessed_dataset_validation.json
repro_logs/single_gpu_smoke.log
repro_logs/single_gpu_pretrained_eval_smoke_timeout.log
repro_logs/single_gpu_fullscript_start_smoke.log
```

## Practical Next Step

If the goal is "complete reproduction result", run full checkpoint evaluation first because it is much shorter than 40-epoch training:

```bash
bash scripts/ADAPT_single_gpu_eval.sh
```

Then start the full single-GPU training only if you are ready to leave the remote machine running for a long time:

```bash
bash scripts/ADAPT_single_gpu_multitask.sh
```

