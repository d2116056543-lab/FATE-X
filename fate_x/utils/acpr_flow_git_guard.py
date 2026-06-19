from __future__ import annotations

import subprocess
from pathlib import Path


def _windows_gitdir_to_wsl(path: str) -> str:
    normalized = path.strip().replace("\\", "/")
    if len(normalized) >= 3 and normalized[1:3] == ":/":
        drive = normalized[0].lower()
        return f"/mnt/{drive}/{normalized[3:]}"
    return normalized


def _worktree_git_args(cwd: str | Path) -> list[str]:
    git_file = Path(cwd) / ".git"
    if not git_file.is_file():
        return []
    text = git_file.read_text(encoding="utf-8").strip()
    if not text.lower().startswith("gitdir:"):
        return []
    gitdir = _windows_gitdir_to_wsl(text.split(":", 1)[1].strip())
    return ["--git-dir", gitdir, "--work-tree", str(cwd)]


def _git(args: list[str], cwd: str | Path) -> str:
    try:
        return subprocess.check_output(["git", "-c", "safe.directory=*", *args], cwd=str(cwd), text=True, stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError:
        prefix = _worktree_git_args(cwd)
        if not prefix:
            raise
        return subprocess.check_output(["git", *prefix, *args], cwd=str(cwd), text=True, stderr=subprocess.STDOUT).strip()


def git_guard(repo_root: str | Path, expected_branch: str = "flowtrace_pmt_v1", remote: str = "github") -> dict[str, str]:
    branch = _git(["branch", "--show-current"], repo_root)
    status = _git(["status", "--porcelain"], repo_root)
    head = _git(["rev-parse", "HEAD"], repo_root)
    try:
        remote_head = _git(["ls-remote", remote, f"refs/heads/{expected_branch}"], repo_root).split()[0]
    except Exception:
        remote_head = ""
    return {"branch": branch, "dirty_status": status, "git_head": head, "github_remote_head": remote_head}


def require_clean_pushed_head(repo_root: str | Path, expected_branch: str = "flowtrace_pmt_v1", remote: str = "github") -> dict[str, str]:
    info = git_guard(repo_root, expected_branch, remote)
    if info["branch"] != expected_branch:
        raise RuntimeError(f"wrong branch {info['branch']} != {expected_branch}")
    if info["dirty_status"]:
        raise RuntimeError("worktree is dirty")
    if info["github_remote_head"] and info["github_remote_head"] != info["git_head"]:
        raise RuntimeError("local HEAD does not match pushed GitHub HEAD")
    return info
