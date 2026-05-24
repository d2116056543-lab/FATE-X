$ErrorActionPreference = "Stop"
Set-Location "E:\sbw\ADAPT_repro\ADAPT"
Write-Host "---SET APT MIRROR---"
wsl.exe -d ADAPT-Ubuntu -- bash -lc @'
set -euxo pipefail
cp -a /etc/apt/sources.list /etc/apt/sources.list.bak.$(date +%Y%m%d_%H%M%S) || true
cat >/etc/apt/sources.list <<'EOF'
deb http://mirrors.aliyun.com/ubuntu/ jammy main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ jammy-updates main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ jammy-backports main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ jammy-security main restricted universe multiverse
EOF
apt-get clean
apt-get update
'@
Write-Host "mirror_exit=$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}
