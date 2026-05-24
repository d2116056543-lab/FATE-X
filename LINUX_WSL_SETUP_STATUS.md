# ADAPT Linux / WSL Setup Status

This repo is prepared for running ADAPT in Linux through WSL2, but the host still needs one more reboot before WSL2 can import and run the Ubuntu distro.

## Current State After First Reboot

- Remote repo: `E:\sbw\ADAPT_repro\ADAPT`
- Target WSL distro name: `ADAPT-Ubuntu`
- Ubuntu 22.04 WSL rootfs: `F:\sbw_adapt_assets\wsl\ubuntu-jammy-wsl-amd64-wsl.rootfs.tar.gz`
- Rootfs tar integrity check passed with `tar_exit=0`.
- Planned WSL install root: `F:\sbw_adapt_assets\wsl\ADAPT-Ubuntu`
- `Microsoft-Windows-Subsystem-Linux`: `Enabled`
- `VirtualMachinePlatform`: `Enabled`
- Component Based Servicing pending reboot: `True`
- Active Baidu Netdisk processes were detected after the first reboot, so I did not reboot again.

## Important Clarification

Data can be reused directly in Linux/WSL:

- Windows `E:\...` is visible in WSL as `/mnt/e/...`.
- Windows `F:\...` is visible in WSL as `/mnt/f/...`.
- The Baidu-downloaded BDDX processed files can stay under the existing ADAPT repo/dataset paths and be read from WSL without copying.

The Windows Python environment cannot be reused directly:

- `E:\Anaconda\envs\sbw39` contains Windows binaries.
- WSL/Linux needs a separate Linux conda environment.
- The setup script creates a Linux conda env named `adapt` inside WSL.

GPU usage:

- WSL uses the Windows NVIDIA driver.
- Do not install a Linux NVIDIA display driver inside WSL.
- After the second reboot and WSL import, `nvidia-smi` and `torch.cuda.is_available()` must work inside WSL.

## Files Prepared

- `repro_tools\setup_wsl_adapt_post_reboot.ps1`
- `repro_tools\setup_adapt_linux_env.sh`
- `repro_tools\verify_adapt_linux_env.sh`

## Next Step After Baidu Download Finishes

Run this once on the remote Windows host:

```powershell
Restart-Computer
```

After reconnecting by SSH:

```powershell
cd E:\sbw\ADAPT_repro\ADAPT
powershell -NoProfile -ExecutionPolicy Bypass -File .\repro_tools\setup_wsl_adapt_post_reboot.ps1
```

This will:

1. Refuse to continue if Windows still has a pending reboot.
2. Check WSL and VirtualMachinePlatform.
3. Set WSL default version to 2.
4. Import Ubuntu 22.04 as `ADAPT-Ubuntu`.
5. Install Linux system packages.
6. Create conda env `adapt`.
7. Install ADAPT's PyTorch 1.13.1 CUDA 11.7 stack.
8. Install MPI, requirements, DeepSpeed, and Apex.
9. Verify CUDA, DeepSpeed, Apex, and ADAPT imports.

If you only want to import Ubuntu first:

```powershell
cd E:\sbw\ADAPT_repro\ADAPT
powershell -NoProfile -ExecutionPolicy Bypass -File .\repro_tools\setup_wsl_adapt_post_reboot.ps1 -SkipLinuxEnvInstall
```

## Known Risk Points

- Apex CUDA-extension build requires `nvcc`; the setup script installs CUDA 11.7 toolkit through conda if `nvcc` is missing.
- ADAPT's `requirements.txt` includes environment-specific or CUDA-build-sensitive entries. The Linux setup script installs official PyTorch/MPI separately and writes skipped requirement lines to `repro_logs/linux_setup/requirements_linux_skipped.txt`.
- Docker is not configured yet. The current route is native WSL2 + conda, not ADAPT's Docker image route.
