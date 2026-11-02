"""Aggregate experiment 23: does the analytic route hold up at medium scale?

The reference carries sampling error of its own, so no figure here can be read
without the **noise ceiling** alongside it: the rank correlation between two
independent half-samples.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ORDER = ["phi_hat", "mc_4000", "mc_1000", "mc_250", "y0sq_x_qvar",
         "pagerank_src", "q_variance", "entropy", "n_uncertain", "out_degree"]


def cluster_boot(d: pd.Series, n_boot=20000, seed=0):
    """Bootstrap with a (setting, seed) cluster and the setting as stratum."""
    d = d.dropna()
    if d.size < 2:
        return (float(d.mean()) if d.size else np.nan, np.nan, np.nan)
    cl = {k: v.to_numpy() for k, v in d.groupby(level=[0, 1])}
    st: dict = {}
    for (lab, _s), a in cl.items():
        st.setdefault(lab, []).append(a)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        parts = []
        for arrs in st.values():
            idx = rng.integers(0, len(arrs), size=len(arrs))
            parts.extend(arrs[j] for j in idx)
        out[b] = np.concatenate(parts).mean()
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        REPO / "outputs/exp23_medium_scale/results.csv")
    df = pd.read_csv(path)
    have = [m for m in ORDER if m in set(df["method"])]
    u = df.drop_duplicates(["label", "seed", "contrast"])
    print(f"read: {path}  ({len(df):,} rows)")
    print(f"{df['label'].nunique()} settings  seeds {sorted(df['seed'].unique())}  "
          f"contrast {sorted(df['contrast'].unique())}")
    for lab, g in df.groupby("label"):
        gu = g.drop_duplicates(["seed", "contrast"])
        print(f"  {lab:16s} n={g.n.iloc[0]:4d} m={g.m.iloc[0]:5d} "
              f"uncertain={g.n_uncertain.iloc[0]:5d} "
              f"sources={g.n_sources_total.iloc[0]:4d} "
              f"candidates={g.n_candidates.iloc[0]:3d}")

    print("\n" + "=" * 92)
    print("[0] Quality of the reference (the ceiling on every correlation below)")
    print("=" * 92)
    print(u.groupby("label")[["reference_split_half", "reference_ci_width_rel",
                             "reference_seconds"]].mean().round(4).to_string())
    ceil = u["reference_split_half"].mean()
    print(f"\nnoise ceiling (rank correlation of two independent halves), "
          f"mean: {ceil:.3f}")
    print("The reference averages the two halves, so it is somewhat more "
          "reliable than\nthis figure, but not perfectly so: any method's "
          "correlation can top out here.")

    print("\n" + "=" * 92)
    print("[1] Rank correlation with the reference, by contrast")
    print("=" * 92)
    print(df.pivot_table(index="contrast", columns="method", values="spearman")
          .reindex(columns=have).round(3).to_string())
    print("\nmean over contrasts:")
    print(df.groupby("method")["spearman"].mean().reindex(have).round(3).to_string())

    piv = df.pivot_table(index=["label", "seed", "contrast"],
                         columns="method", values="spearman")
    print("\n" + "=" * 92)
    print("[2] Paired difference against phi_hat (positive favours phi_hat)")
    print("=" * 92)
    print(f"{'competitor':16s} {'mean diff':>10s} {'95% CI':>22s} "
          f"{'win/tie/loss':>14s}  sig")
    for b in have[1:]:
        if b not in piv:
            continue
        d = (piv["phi_hat"] - piv[b]).dropna()
        m, lo, hi = cluster_boot(d)
        print(f"{b:14s} {m:+8.3f}   [{lo:+.3f}, {hi:+.3f}]   "
              f"{int((d>0).sum()):3d}/{int((d==0).sum()):3d}/{int((d<0).sum()):3d}  "
              f"{'YES' if lo > 0 else 'no'}")

    print("\n" + "=" * 92)
    print("[3] Cost: what Monte Carlo spends to reach the same agreement")
    print("=" * 92)
    t = df.groupby("method").agg(rho=("spearman", "mean"), sec=("seconds", "mean"),
                                 solves=("solves", "mean"),
                                 top5=("top5_overlap", "mean"))
    print(t.reindex(have).round(4).to_string())
    ph = df[df.method == "phi_hat"]["seconds"].mean()
    ref_s = u["reference_seconds"].mean()
    print(f"\nphi_hat takes {ph:.4f}s against {ref_s:.1f}s for the reference, "
          f"a factor of {ref_s / ph:,.0f}")
    print("Monte Carlo needs N*(1 + candidates) model evaluations, growing with "
          "the\nnumber of candidates; the analytic route solves one resolvent "
          "per contrast.")


if __name__ == "__main__":
    main()
