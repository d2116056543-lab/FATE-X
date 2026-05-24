$ErrorActionPreference = "Stop"
wsl.exe -d ADAPT-Ubuntu -- bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate adapt && python -m pip install -i https://mirrors.aliyun.com/pypi/simple/ scikit-learn==1.3.2 && python - <<'PY'
import sklearn
print('sklearn', sklearn.__version__)
PY"
