"""Download BDD-X raw videos from URLs recorded in the public annotation CSV."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def _video_name(url: str) -> str:
    return Path(urlparse(url).path).name


def load_unique_urls(csv_path: Path) -> list[str]:
    seen = set()
    urls = []
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            url = (row.get("Input.Video") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            urls.append(url)
    return urls


def probe_url(url: str, timeout: int) -> tuple[int | None, int | None, str | None]:
    req = Request(url, method="HEAD")
    try:
        with urlopen(req, timeout=timeout) as resp:
            length = resp.headers.get("Content-Length")
            return resp.status, int(length) if length else None, None
    except HTTPError as exc:
        return exc.code, None, str(exc)
    except URLError as exc:
        return None, None, str(exc)


def download_one(url: str, output: Path, timeout: int) -> tuple[bool, str | None]:
    tmp = output.with_suffix(output.suffix + ".part")
    req = Request(url)
    try:
        with urlopen(req, timeout=timeout) as resp, tmp.open("wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        tmp.replace(output)
        return True, None
    except HTTPError as exc:
        if tmp.exists():
            tmp.unlink()
        return False, f"HTTP {exc.code}: {exc.reason}"
    except Exception as exc:  # noqa: BLE001 - keep download status auditable.
        if tmp.exists():
            tmp.unlink()
        return False, repr(exc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--status_jsonl", required=True, type=Path)
    parser.add_argument("--max_videos", type=int, default=0)
    parser.add_argument("--stop_after_errors", type=int, default=25)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--probe_only", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.status_jsonl.parent.mkdir(parents=True, exist_ok=True)
    urls = load_unique_urls(args.csv)
    if args.max_videos > 0:
        urls = urls[: args.max_videos]

    summary = {
        "csv": str(args.csv),
        "output_dir": str(args.output_dir),
        "requested": len(urls),
        "downloaded": 0,
        "existing": 0,
        "failed": 0,
        "probe_only": args.probe_only,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    consecutive_errors = 0
    with args.status_jsonl.open("a", encoding="utf-8") as status_f:
        for idx, url in enumerate(urls, start=1):
            name = _video_name(url)
            out = args.output_dir / name
            record = {"index": idx, "url": url, "file": str(out)}
            if out.exists() and out.stat().st_size > 0:
                record.update({"status": "existing", "bytes": out.stat().st_size})
                summary["existing"] += 1
                consecutive_errors = 0
            else:
                status, length, probe_error = probe_url(url, args.timeout)
                record.update({"head_status": status, "head_bytes": length})
                if probe_error:
                    record["head_error"] = probe_error
                if args.probe_only:
                    record["status"] = "probed"
                elif status and 200 <= status < 300:
                    ok, error = download_one(url, out, args.timeout)
                    if ok:
                        record.update({"status": "downloaded", "bytes": out.stat().st_size})
                        summary["downloaded"] += 1
                        consecutive_errors = 0
                    else:
                        record.update({"status": "failed", "error": error})
                        summary["failed"] += 1
                        consecutive_errors += 1
                else:
                    record.update({"status": "failed", "error": probe_error or f"HEAD status {status}"})
                    summary["failed"] += 1
                    consecutive_errors += 1
            status_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            status_f.flush()
            print(json.dumps(record, ensure_ascii=False))
            if args.stop_after_errors > 0 and consecutive_errors >= args.stop_after_errors:
                summary["stopped_reason"] = f"consecutive_errors>={args.stop_after_errors}"
                break

    summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
