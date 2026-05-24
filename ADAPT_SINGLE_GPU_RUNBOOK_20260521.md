# ADAPT Single-GPU Runbook - 2026-05-21

## Enter Remote Repo

From local Windows PowerShell:

```powershell
ssh lenovo@100.75.8.120
cd E:\sbw\ADAPT_repro\ADAPT
```

Run Linux commands through WSL:

```powershell
wsl -d ADAPT-Ubuntu
cd /mnt/e/sbw/ADAPT_repro/ADAPT
source /opt/conda/etc/profile.d/conda.sh
conda activate adapt
```

## Verify Environment

```bash
python - <<'PY'
import torch, apex, sklearn, shutil
print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("apex", apex.__file__)
print("sklearn", sklearn.__version__)
print("java", shutil.which("java"))
PY
```

Expected:

```text
torch 1.13.1+cu117
cuda_available True
gpu NVIDIA GeForce RTX 4090
```

## Official 4-GPU Command

Official repo script:

```bash
bash scripts/BDDX_multitask.sh
```

This uses 4 GPUs and should not be used directly on the single-GPU host.

## Single-GPU Full Training

Recommended single-GPU full training:

```bash
cd /mnt/e/sbw/ADAPT_repro/ADAPT
source /opt/conda/etc/profile.d/conda.sh
conda activate adapt

CUDA_VISIBLE_DEVICES=0 \
OUTPUT_DIR=./output/repro_single_gpu/multitask_sensor_course_speed \
PER_GPU_TRAIN_BATCH_SIZE=2 \
PER_GPU_EVAL_BATCH_SIZE=2 \
GRADIENT_ACCUMULATION_STEPS=32 \
NUM_TRAIN_EPOCHS=40 \
bash scripts/ADAPT_single_gpu_multitask.sh
```

Effective batch:

```text
1 GPU * batch 2 * accumulation 32 = 64
```

This matches the official effective batch:

```text
4 GPUs * batch 4 * accumulation 4 = 64
```

If GPU memory is insufficient, use:

```bash
PER_GPU_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=64
```

If you want to test whether RTX 4090 can run faster:

```bash
PER_GPU_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=16
```

This has not been fully smoke-tested yet.

## Smoke Training

Finite smoke wrapper:

```bash
cd /mnt/e/sbw/ADAPT_repro/ADAPT
SMOKE_TIMEOUT_SECONDS=180 bash scripts/ADAPT_single_gpu_smoke_timeout.sh
```

This is only a startup/batch proof. It intentionally stops by timeout.

The original `--limited_samples` behavior is not sufficient by itself because ADAPT still computes `max_iter` from the full training set.

## Full-Script Startup Smoke

This validates the exact full single-GPU script settings with batch size 2:

```bash
cd /mnt/e/sbw/ADAPT_repro/ADAPT
SMOKE_TIMEOUT_SECONDS=240 bash scripts/ADAPT_single_gpu_fullscript_start_smoke.sh
```

Expected log:

```text
Train with 2 images per GPU.
input_ids = torch.Size([2, 30])
img_feats = torch.Size([2, 32, 3, 224, 224])
car_info = torch.Size([2, 2, 32])
```

The timeout SIGTERM is expected.

## Pretrained Checkpoint Eval

Smoke eval:

```bash
cd /mnt/e/sbw/ADAPT_repro/ADAPT
SMOKE_TIMEOUT_SECONDS=120 bash scripts/ADAPT_single_gpu_eval_smoke_timeout.sh
```

Full eval:

```bash
cd /mnt/e/sbw/ADAPT_repro/ADAPT
CUDA_VISIBLE_DEVICES=0 \
EVAL_DIR=checkpoints/basemodel/checkpoints/ \
DATA_DIR=datasets \
VAL_YAML=BDDX/testing_32frames.yaml \
PER_GPU_EVAL_BATCH_SIZE=4 \
bash scripts/ADAPT_single_gpu_eval.sh
```

This follows the official `BDDX_test.sh` behavior, but uses GPU 0 instead of hard-coded GPU 5.

## Windows Watch Commands

Check active ADAPT processes:

```powershell
ssh lenovo@100.75.8.120 "powershell -NoProfile -ExecutionPolicy Bypass -File E:\sbw\ADAPT_repro\ADAPT\repro_tools\remote_adapt_status.ps1"
```

Tail training smoke:

```powershell
ssh lenovo@100.75.8.120 "powershell -NoProfile -Command Get-Content E:\sbw\ADAPT_repro\ADAPT\repro_logs\single_gpu_fullscript_start_smoke.log -Tail 80"
```

Tail full training if launched:

```powershell
ssh lenovo@100.75.8.120 "powershell -NoProfile -Command Get-ChildItem E:\sbw\ADAPT_repro\ADAPT\output\repro_single_gpu\multitask_sensor_course_speed -Recurse | Sort-Object LastWriteTime -Descending | Select-Object -First 20 FullName,Length,LastWriteTime"
```

Stop ADAPT Python processes:

```powershell
ssh lenovo@100.75.8.120 "powershell -NoProfile -ExecutionPolicy Bypass -File E:\sbw\ADAPT_repro\ADAPT\repro_tools\remote_stop_adapt_smoke.ps1"
```

