param(
  [int]$Batch = 4,
  [int]$Accum = 16,
  [int]$TimeoutSeconds = 180
)
$ErrorActionPreference = "Stop"
wsl -d ADAPT-Ubuntu -- bash -lc "cd /mnt/e/sbw/ADAPT_repro/ADAPT && PER_GPU_TRAIN_BATCH_SIZE=$Batch GRADIENT_ACCUMULATION_STEPS=$Accum SMOKE_TIMEOUT_SECONDS=$TimeoutSeconds bash scripts/ADAPT_single_gpu_mem_probe.sh"
