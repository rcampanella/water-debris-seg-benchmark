#!/usr/bin/env python3
"""Download ORCA Synology public file shares (FloW-RI / FloW-Img zips)."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE = "https://nas.orca-tech.cn:5000"

FLOW_RI_SHARES = [
    ("FloW-RI-1-FloW_RI_Sequence_1_10", "CRFscZuGI"),
    ("FloW-RI-2-FloW_RI_Sequence_11_21", "CKxN4FSZg"),
    ("Flow-RI_ReadMe_and_Development_Tool", "NKDJd8tIx"),
]

FLOW_IMG_SHARES = [
    ("FloW-Img Image Dataset", "tCGbUO7r6"),
    ("FloW-Img Video Sequence 1-50", "zZh5iZQKC"),
    ("FloW-Img Video Sequence 51-100", "Lk9dzWJ0O"),
    ("FloW-Img Video Sequence 101-150", "Z2ZSo6Tc5"),
    ("FloW-Img Video Sequence 151-200", "qNIb4uifd"),
    ("FloW-Img Video Sequence 201-244", "rqnVBSx0W"),
]

BUNDLES = {
    "flow-ri": FLOW_RI_SHARES,
    "flow-img": FLOW_IMG_SHARES,
    "all": FLOW_RI_SHARES + FLOW_IMG_SHARES,
}


def _sharing_session(link_id: str) -> tuple[requests.Session, str, dict]:
    session = requests.Session()
    session.headers["User-Agent"] = "water-debris-seg/flow-download/1.0"
    session.get(f"{BASE}/sharing/{link_id}", timeout=120)
    entry = f"{BASE}/sharing/webapi/entry.cgi"
    headers = {"X-SYNO-SHARING": link_id}
    r = session.post(
        entry,
        data={
            "api": "SYNO.Core.Sharing.Login",
            "version": 1,
            "method": "login",
            "sharing_id": link_id,
            "password": "",
        },
        headers=headers,
        timeout=60,
    )
    r.raise_for_status()
    body = r.json()
    if not body.get("success"):
        raise RuntimeError(f"Sharing login failed for {link_id}: {body}")

    r = session.post(
        entry,
        data={
            "api": "SYNO.Core.Sharing.Session",
            "version": 1,
            "method": "get",
            "sharing_id": link_id,
            "sharing_status": "none",
        },
        headers=headers,
        timeout=60,
    )
    r.raise_for_status()
    m = re.search(r'"filename"\s*:\s*"([^"]+)"', r.text)
    if not m:
        raise RuntimeError(f"Could not read filename for share {link_id}")
    return session, entry, headers


def download_share(name: str, link_id: str, dest_dir: Path) -> Path:
    session, entry, headers = _sharing_session(link_id)
    r = session.post(
        entry,
        data={"api": "SYNO.Core.Sharing.Session", "version": 1, "method": "get",
              "sharing_id": link_id, "sharing_status": "none"},
        headers=headers,
        timeout=60,
    )
    m = re.search(r'"filename"\s*:\s*"([^"]+)"', r.text)
    filename = m.group(1)
    out = dest_dir / filename
    part = out.with_suffix(out.suffix + ".part")

    params = {
        "api": "SYNO.FolderSharing.Download",
        "version": 2,
        "method": "download",
        "mode": '"download"',
        "_sharing_id": json.dumps(link_id),
        "path": json.dumps([f"/{filename}"]),
    }

    have = part.stat().st_size if part.exists() else 0
    if out.exists() and not part.exists():
        print(f"[skip] {name}: {out} already exists", flush=True)
        return out

    req_headers = dict(headers)
    if have:
        req_headers["Range"] = f"bytes={have}-"

    with session.get(
        entry, params=params, headers=req_headers, stream=True, timeout=120
    ) as resp:
        resp.raise_for_status()
        if "application/json" in resp.headers.get("Content-Type", ""):
            raise RuntimeError(f"Download refused for {name}: {resp.text[:500]}")
        mode = "ab" if have and resp.status_code == 206 else "wb"
        if mode == "wb" and part.exists():
            part.unlink()
            have = 0
        total = resp.headers.get("Content-Length")
        total_i = int(total) + have if total else None
        done = have
        chunk = 1024 * 1024
        with open(part, mode) as fh:
            for block in resp.iter_content(chunk_size=chunk):
                if not block:
                    continue
                fh.write(block)
                done += len(block)
                if total_i:
                    pct = 100.0 * done / total_i
                    print(
                        f"\r[{name}] {done/1e9:.2f}/{total_i/1e9:.2f} GB ({pct:.1f}%)",
                        end="",
                        flush=True,
                    )
                else:
                    print(f"\r[{name}] {done/1e9:.2f} GB", end="", flush=True)

    part.replace(out)
    print(f"\n[done] {name} -> {out}", flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle",
        choices=sorted(BUNDLES),
        default="flow-ri",
        help="Which ORCA share set to download (default: flow-ri)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("/scratch-shared") / __import__("os").environ.get("USER", "user")
        / "water-debris-seg/data/flow/sources",
    )
    args = parser.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)

    shares = BUNDLES[args.bundle]
    for name, link_id in shares:
        try:
            download_share(name, link_id, args.dest)
        except Exception as exc:
            print(f"[error] {name}: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
