"""Aggregate the per-source validation on the real networks.

Reads `outputs/real_per_source_<net>.csv`, written by `real_per_source.py`, and
prints the three things the paper reports from it:

  1. the same-footing comparison of Table VII: one half of the Owen reference is
     the ground truth, and the competitor is either the other half (an
     independent run of equal budget) or the analytic `phi_hat`
  2. the paired bootstrap over the candidate sources for each of those four
     comparisons, which is what decides whether the analytic route is ahead or
     merely equivalent
  3. the overall and within-stratum correlations of the cheap heuristics, whose
     overall values are dominated by the stratified design

The bootstrap resamples the sources, recomputes both Spearman correlations on
each resample and takes their difference, so the two correlations stay paired.
The candidates are a stratified sample rather than a random one, so the interval
describes this candidate set, not the whole network.

Usage:
    python scripts/aggregate_real_per_source.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
NETS = [("trackback", "Trackback NW"), ("wikivote", "wiki-Vote")]
HEURISTICS = ["pagerank_src", "y0sq_x_trcov", "y0sq_x_qvar", "phi_no_y0sq",
              "phi_randdir", "out_degree"]


def paired_boot(truth: np.ndarray, a: np.ndarray, b: np.ndarray,
                n_boot: int = 20000, seed: int = 0) -> tuple[float, float, float]:
    """95% interval for rho(truth, a) - rho(truth, b), resampling the sources."""
    rng = np.random.default_rng(seed)
    n = truth.size
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        t = truth[idx]
        # A resample can repeat a source enough times to make a vector constant,
        # which leaves Spearman undefined; those draws are dropped.
        ra = spearmanr(t, a[idx]).statistic
        rb = spearmanr(t, b[idx]).statistic
        diffs[i] = ra - rb
    diffs = diffs[~np.isnan(diffs)]
    point = spearmanr(truth, a).statistic - spearmanr(truth, b).statistic
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=20000)
    a = ap.parse_args()

    print("=" * 78)
    print("[1] Table VII: same footing, both directions")
    print("=" * 78)
    print(f"{'network':14s} {'ground truth':13s} {'Owen 40k':>9s} {'phi_hat':>9s}")
    rows = {}
    for key, title in NETS:
        d = pd.read_csv(REPO / f"outputs/real_per_source_{key}.csv")
        rows[key] = d
        for truth_name, truth, comp in [("half B", d.psi_b, d.psi_a),
                                        ("half A", d.psi_a, d.psi_b)]:
            r_mc = spearmanr(truth, comp).statistic
            r_phi = spearmanr(truth, d.phi_hat).statistic
            lead = "*" if r_phi > r_mc else " "
            print(f"{title:14s} {truth_name:13s} {r_mc:9.3f} {r_phi:8.3f}{lead}")

    print("\n" + "=" * 78)
    print(f"[2] Paired bootstrap of phi_hat - Owen, over the candidate sources "
          f"({a.n_boot:,} resamples)")
    print("=" * 78)
    print(f"{'network':14s} {'ground truth':13s} {'diff':>8s} {'95% CI':>20s}  "
          f"separates from 0")
    for key, title in NETS:
        d = rows[key]
        for truth_name, truth, comp in [("half B", d.psi_b, d.psi_a),
                                        ("half A", d.psi_a, d.psi_b)]:
            pt, lo, hi = paired_boot(truth.to_numpy(), d.phi_hat.to_numpy(),
                                     comp.to_numpy(), n_boot=a.n_boot)
            sig = "YES" if lo > 0 else "no"
            print(f"{title:14s} {truth_name:13s} {pt:+8.3f}   "
                  f"[{lo:+.3f}, {hi:+.3f}]  {sig}")

    print("\n" + "=" * 78)
    print("[3] Cheap heuristics: overall against within-stratum")
    print("=" * 78)
    for key, title in NETS:
        d = rows[key]
        print(f"\n{title}  ({len(d)} candidates, "
              f"{d.stratum.nunique()} strata)")
        print(f"  {'method':16s} {'overall':>8s}   per stratum")
        for m in HEURISTICS:
            if m not in d:
                continue
            overall = spearmanr(d.psi_ref, d[m]).statistic
            per = []
            for s in sorted(d.stratum.unique()):
                t = d[d.stratum == s]
                r = spearmanr(t.psi_ref, t[m]).statistic if len(t) > 2 else np.nan
                per.append(f"{s}={r:+.2f}" if np.isfinite(r) else f"{s}=n/a")
            print(f"  {m:16s} {overall:+8.3f}   {'  '.join(per)}")


if __name__ == "__main__":
    main()
