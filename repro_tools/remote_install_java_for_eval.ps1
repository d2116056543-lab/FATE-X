$ErrorActionPreference = "Stop"
wsl.exe -d ADAPT-Ubuntu -- bash -lc "set -e; apt-get update; DEBIAN_FRONTEND=noninteractive apt-get install -y default-jre-headless; java -version"
