$ErrorActionPreference = "Stop"
wsl.exe -d ADAPT-Ubuntu -- bash -lc '/opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main'
wsl.exe -d ADAPT-Ubuntu -- bash -lc '/opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r'
