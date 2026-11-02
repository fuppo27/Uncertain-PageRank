"""Record the information needed to reproduce a run."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy


def git_state(repo: Path) -> dict[str, str]:
    """The git commit and working-tree state at run time."""

    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": "yes" if run("status", "--porcelain") else "no",
    }


def provenance(config: dict, repo: Path) -> dict:
    """Collect the configuration, the environment and the git state."""
    return {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git": git_state(repo),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "config": config,
    }


def save_run(out_dir: Path, config: dict, rows: list[dict], repo: Path) -> Path:
    """Write the result CSV and the reproduction metadata JSON.

    Returns:
        The path of the CSV written.
    """
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    (out_dir / "meta.json").write_text(
        json.dumps(provenance(config, repo), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return csv_path
