$ErrorActionPreference = "Continue"
wsl.exe -d ADAPT-Ubuntu -- bash -lc @'
set +e
for url in \
  "https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/cu117/torch-1.13.1%2Bcu117-cp38-cp38-linux_x86_64.whl" \
  "https://download.pytorch.org/whl/cu117/torch-1.13.1%2Bcu117-cp38-cp38-linux_x86_64.whl"
do
  echo "--- $url"
  curl -I --connect-timeout 20 "$url" | sed -n '1,12p'
  echo "exit=$?"
done
'@
