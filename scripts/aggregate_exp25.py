"""Aggregate experiment 25: the cost of the attribution on real networks.

Prints the cost table of the paper, and the three figures its text quotes for
the largest network.

The experiment runs each network under more than one contrast family, and the
costs differ between them: the attribution is a solve per contrast, so
`node_top` and `boundary` do not take the same time. The table reports the
`boundary` contrast, the one the per-source validation uses, and this script
prints the others alongside so that the table and the text cannot drift apart
from each other again.

Usage:
    python scripts/aggregate_exp25.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]

# The order of the table: by node count, the smallest first.
ORDER = ["Trackback NW", "wiki-Vote", "soc-Epinions1", "Follow NW"]
TABLE_CONTRAST = "boundary"


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        REPO / "outputs/exp25_real_networks/results.csv")
    d = pd.read_csv(path)
    d = d[d.method == "phi_hat"]
    print(f"read: {path} ({len(d):,} rows, method=phi_hat)")
    missing = set(ORDER) - set(d.network)
    assert not missing, f"network missing from the results: {sorted(missing)}"

    print("\n" + "=" * 72)
    print(f"[1] Cost table, {TABLE_CONTRAST} contrast")
    print("=" * 72)
    print(f"{'Network':14s} {'n':>7s} {'m':>8s} {'|U|':>7s} "
          f"{'P_bar':>7s} {'Phi_hat':>8s} {'RSS':>5s}")
    print(f"{'':14s} {'':>7s} {'':>8s} {'(sources)':>7s}")
    for net in ORDER:
        r = d[(d.network == net) & (d.contrast == TABLE_CONTRAST)]
        assert len(r) == 1, f"{net}: {len(r)} rows for contrast {TABLE_CONTRAST}"
        r = r.iloc[0]
        print(f"{net:14s} {r.n:7,d} {r.m:8,d} {r.n_sources:7,d} "
              f"{r.expected_transition_seconds:7.2f} "
              f"{r.attribution_seconds:8.2f} {r.peak_rss_mb:5.0f}")

    print("\n" + "=" * 72)
    print("[2] The same cost under every contrast family measured")
    print("=" * 72)
    print(f"{'Network':14s} {'contrast':10s} {'P_bar':>7s} {'Phi_hat':>8s} {'RSS':>7s}")
    for net in ORDER:
        for _, r in d[d.network == net].sort_values("contrast").iterrows():
            mark = "  <- table" if r.contrast == TABLE_CONTRAST else ""
            print(f"{net:14s} {r.contrast:10s} "
                  f"{r.expected_transition_seconds:7.2f} "
                  f"{r.attribution_seconds:8.2f} {r.peak_rss_mb:7.2f}{mark}")

    print("\n" + "=" * 72)
    print("[3] The figures the text of the section quotes")
    print("=" * 72)
    big = ORDER[-1]
    r = d[(d.network == big) & (d.contrast == TABLE_CONTRAST)].iloc[0]
    print(f"{big}: {r.n:,} nodes, {r.m:,} links, {r.n_sources:,} uncertain sources"
          f" over {r.n_uncertain:,} uncertain links")
    print(f"  attribution        {r.attribution_seconds:.1f} s")
    print(f"  peak resident size {r.peak_rss_mb:.0f} MB")
    print(f"  build of P_bar     {r.expected_transition_seconds:.1f} s")
    print(f"  ratio build:attribution  1 : "
          f"{r.attribution_seconds / r.expected_transition_seconds:.1f}")


if __name__ == "__main__":
    main()
