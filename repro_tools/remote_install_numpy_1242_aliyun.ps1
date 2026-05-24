$ErrorActionPreference = "Stop"
wsl.exe -d ADAPT-Ubuntu -- bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate adapt && python -m pip install --no-cache-dir --retries 20 --timeout 120 -i https://mirrors.aliyun.com/pypi/simple/ numpy==1.24.2"
