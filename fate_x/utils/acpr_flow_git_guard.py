from __future__ import annotations

import subprocess
from pathlib import Path


def _git(args: list[str], cwd: str | Path) -> str:
    return subprocess.check_output(["git", "-c", "safe.directory=*", *args], cwd=str(cwd), text=True, stderr=subprocess.STDOUT).strip()


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
