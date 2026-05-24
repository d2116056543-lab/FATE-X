#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="/mnt/e/sbw/ADAPT_repro/ADAPT/repro_logs/linux_setup"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/setup_adapt_linux_env.log") 2>&1

echo "==== ADAPT Linux environment setup started: $(date -Is) ===="

if [[ "$(id -u)" -ne 0 ]]; then
  echo "This script is expected to run as root inside the imported WSL distro."
  echo "If you later create a normal Linux user, run this via sudo."
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y \
  build-essential git curl wget ca-certificates unzip ffmpeg \
  pkg-config libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
  ninja-build cmake rsync vim less openmpi-bin libopenmpi-dev \
  libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
  libavfilter-dev libswscale-dev libswresample-dev

if [[ ! -x /opt/conda/bin/conda ]]; then
  echo "Installing Miniconda to /opt/conda"
  MINICONDA_URL="${MINICONDA_URL:-https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh}"
  curl -L -C - --retry 20 --retry-delay 5 --retry-all-errors --connect-timeout 30 -o /tmp/miniconda.sh "$MINICONDA_URL"
  test -s /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p /opt/conda
fi

source /opt/conda/etc/profile.d/conda.sh
conda config --set auto_activate_base false
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

if ! conda env list | awk '{print $1}' | grep -qx "adapt"; then
  conda create -y -n adapt python=3.8 pip
fi

conda activate adapt
python -m pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
python -m pip config set global.timeout 120
python -m pip install --retries 20 --timeout 120 -U pip setuptools wheel

echo "Installing official ADAPT PyTorch stack: torch 1.13.1 cu117"
if python - <<'PY'
import torch
assert torch.__version__ == "1.13.1+cu117"
assert torch.cuda.is_available()
PY
then
  echo "Official ADAPT PyTorch stack already installed and CUDA is available."
elif [[ -d /mnt/e/sbw/ADAPT_repro/downloads/torch_cu117 ]]; then
  echo "Installing PyTorch stack from predownloaded local cu117 wheels."
  python -m pip install --retries 20 --timeout 120 \
    /mnt/e/sbw/ADAPT_repro/downloads/torch_cu117/torch-1.13.1+cu117-cp38-cp38-linux_x86_64.whl \
    /mnt/e/sbw/ADAPT_repro/downloads/torch_cu117/torchvision-0.14.1+cu117-cp38-cp38-linux_x86_64.whl \
    /mnt/e/sbw/ADAPT_repro/downloads/torch_cu117/torchaudio-0.13.1+cu117-cp38-cp38-linux_x86_64.whl
else
  python -m pip install --retries 20 --timeout 120 \
    torch==1.13.1+cu117 \
    torchvision==0.14.1+cu117 \
    torchaudio==0.13.1+cu117 \
    -f https://download.pytorch.org/whl/torch_stable.html
fi

echo "Installing MPI packages with Ubuntu OpenMPI + pip mpi4py, matching the ADAPT README intent without slow conda-forge metadata"
mpirun --version | head -n 2 || true
python -m pip install --retries 20 --timeout 120 mpi4py
python - <<'PY'
import mpi4py
print("mpi4py", mpi4py.__version__)
PY

echo "Installing PyAV with Cython<3 for av==10.0.0 compatibility"
python -m pip install --retries 20 --timeout 120 "Cython<3"
python -m pip install --retries 20 --timeout 120 --no-build-isolation av==10.0.0
python -m pip install --retries 20 --timeout 120 "pandas==2.0.3"

REPO="/mnt/e/sbw/ADAPT_repro/ADAPT"
REQ="$REPO/requirements.txt"
FILTERED="$LOG_DIR/requirements_linux_filtered.txt"

if [[ ! -f "$REQ" ]]; then
  echo "Missing requirements.txt at $REQ"
  exit 1
fi

echo "Creating filtered requirements file"
python - "$REQ" "$FILTERED" <<'PY'
from pathlib import Path
import re
src = Path(__import__("sys").argv[1])
dst = Path(__import__("sys").argv[2])
skip_patterns = [
    r"^\s*#",
    r"^\s*$",
    r"^torch==",
    r"^torchvision==",
    r"^torchaudio==",
    r"^av==",
    r"^deepspeed==",
    r"^pandas==",
    r"^openmpi==",
    r"^apex\b",
    r"^tinycudann\b",
]
kept = []
skipped = []
for line in src.read_text(encoding="utf-8", errors="ignore").splitlines():
    if any(re.search(p, line) for p in skip_patterns):
        skipped.append(line)
    else:
        kept.append(line)
dst.write_text("\n".join(kept) + "\n", encoding="utf-8")
(dst.parent / "requirements_linux_skipped.txt").write_text("\n".join(skipped) + "\n", encoding="utf-8")
print(f"kept={len(kept)} skipped={len(skipped)}")
PY

echo "Installing filtered ADAPT requirements"
python -m pip install --retries 20 --timeout 120 -r "$FILTERED"

echo "Checking for nvcc before Apex build"
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version
else
  echo "nvcc is not available. Installing minimal CUDA 11.7 compile packages via conda; full cuda-toolkit is avoided because it pulls multi-GB GUI/profiler packages."
  conda install -y --solver=classic -c nvidia/label/cuda-11.7.0 \
    cuda-nvcc cuda-cudart-dev cuda-cccl
fi

if [[ -d "$CONDA_PREFIX" ]]; then
  export CUDA_HOME="$CONDA_PREFIX"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

echo "Installing deepspeed 0.14.0 explicitly after CUDA toolkit is available"
python -m pip install --retries 20 --timeout 120 "packaging>=23.2"
python -m pip install --retries 20 --timeout 120 deepspeed==0.14.0

echo "Installing NVIDIA Apex with CUDA extensions, matching the ADAPT README"
rm -rf /tmp/adapt_apex
git clone https://github.com/NVIDIA/apex /tmp/adapt_apex
pushd /tmp/adapt_apex
python -m pip install -v --no-cache-dir --no-build-isolation \
  --global-option="--cpp_ext" \
  --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" \
  --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" \
  ./
popd

echo "Installing ADAPT repo in editable mode if packaging metadata exists"
cd "$REPO"
if [[ -f "setup.py" || -f "pyproject.toml" ]]; then
  python -m pip install -e .
else
  echo "No setup.py/pyproject.toml found; ADAPT is used via repo-relative imports."
fi

echo "Writing conda activation helper"
cat > "$REPO/repro_tools/activate_adapt_linux.sh" <<'SH'
#!/usr/bin/env bash
source /opt/conda/etc/profile.d/conda.sh
conda activate adapt
cd /mnt/e/sbw/ADAPT_repro/ADAPT
SH
chmod +x "$REPO/repro_tools/activate_adapt_linux.sh"

echo "==== ADAPT Linux environment setup finished: $(date -Is) ===="
