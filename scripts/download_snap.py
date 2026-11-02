"""Fetch the public SNAP networks and record their provenance in manifest.json.

Usage:
    python3 scripts/download_snap.py --tier scale
    python3 scripts/download_snap.py --tier all --out data/raw/snap
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

BASE = "https://snap.stanford.edu/data/"

# For the scalability experiment, chosen to spread sum_u D_u^2.
SCALE = {
    "wiki-Vote": "wiki-Vote.txt.gz",
    "soc-Epinions1": "soc-Epinions1.txt.gz",
    "web-NotreDame": "web-NotreDame.txt.gz",
    "web-Google": "web-Google.txt.gz",
    "wiki-Talk": "wiki-Talk.txt.gz",
    "web-BerkStan": "web-BerkStan.txt.gz",
    "cit-Patents": "cit-Patents.txt.gz",
    "soc-LiveJournal1": "soc-LiveJournal1.txt.gz",
}

# For q_e estimated from data as a temporal activity probability.
TEMPORAL = {
    "email-Eu-core-temporal": "email-Eu-core-temporal.txt.gz",
    "CollegeMsg": "CollegeMsg.txt.gz",
    "sx-mathoverflow": "sx-mathoverflow.txt.gz",
    "sx-askubuntu": "sx-askubuntu.txt.gz",
    "wiki-talk-temporal": "wiki-talk-temporal.txt.gz",
}

TIERS = {"scale": SCALE, "temporal": TEMPORAL, "all": {**SCALE, **TEMPORAL}}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, filename: str, out_dir: Path) -> dict:
    """Fetch one dataset, skipping the download if the file already exists."""
    url = BASE + filename
    dest = out_dir / filename
    if dest.exists():
        print(f"[skip] {name}: already present ({dest.stat().st_size:,} bytes)")
    else:
        print(f"[get ] {name}: {url}", flush=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as r, tmp.open("wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.rename(dest)
        print(f"[done] {name}: {dest.stat().st_size:,} bytes", flush=True)
    return {
        "name": name,
        "url": url,
        "file": filename,
        "bytes": dest.stat().st_size,
        "sha256": sha256(dest),
        "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=sorted(TIERS), default="all")
    parser.add_argument("--out", type=Path, default=Path("data/raw/snap"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    failed = []
    for name, filename in TIERS[args.tier].items():
        try:
            manifest[name] = fetch(name, filename, args.out)
        except Exception as exc:  # one failure must not stop the rest
            print(f"[FAIL] {name}: {exc}", file=sys.stderr, flush=True)
            failed.append(name)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    total = sum(e["bytes"] for e in manifest.values())
    print(f"\n{len(manifest)} datasets / {total / 1e9:.2f} GB total -> {args.out}")
    if failed:
        print(f"failed: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
