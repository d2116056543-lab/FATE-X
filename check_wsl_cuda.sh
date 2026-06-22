set -euo pipefail
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}
cd /mnt/e/sbw/FATE_Drive/fate_x_flowtrace_pmt_v1_worktree
printf '%s\n' '---wsl cwd---'
pwd
printf '%s\n' '---python/cuda---'
/opt/conda/envs/adapt/bin/python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda', torch.cuda.is_available())
print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print('cudnn', torch.backends.cudnn.version())
if torch.cuda.is_available():
    x=torch.randn(2,3,16,16,device='cuda')
    m=torch.nn.Conv2d(3,4,3).cuda()
    y=m(x)
    print('conv_ok', tuple(y.shape), float(y.mean().detach().cpu()))
PY
printf '%s\n' '---active python---'
ps -eo pid,ppid,etime,cmd | grep -E 'train_acpr_flowcal_v2|run_acpr_flowcal|python' | grep -v grep || true
printf '%s\n' '---gpu---'
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
printf '%s\n' '---recent acpr v2 active runs---'
ls -dt /mnt/e/sbw/FATE_Drive/active_runs/acpr* 2>/dev/null | head -10 || true
