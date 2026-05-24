import importlib
import subprocess

mods = [
    "torch",
    "torchvision",
    "torchaudio",
    "deepspeed",
    "apex",
    "av",
    "mpi4py",
    "transformers",
    "xformers",
]

for name in mods:
    mod = importlib.import_module(name)
    print(f"{name}={getattr(mod, '__version__', 'ok')}")

import torch

print(f"torch_cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"torch_cuda_version={torch.version.cuda}")
    print(f"torch_device={torch.cuda.get_device_name(0)}")

try:
    nvcc = subprocess.run(["nvcc", "--version"], check=True, capture_output=True, text=True)
    print("nvcc_ok=1")
    print(nvcc.stdout.splitlines()[-1])
except Exception as exc:
    print(f"nvcc_ok=0 error={exc}")
