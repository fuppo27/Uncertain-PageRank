"""Aggregate experiment 23 with the Owen reference.

An earlier version kept only the conditions whose split-half agreement reached
0.7, a workaround for the naive reference estimator being broken. With the Owen
reference, **every condition is used**.

Four quantities are reported:
  1. the Spearman correlation of the source ordering
  2. whether the top source agrees
  3. the error in the effect sizes, normalised by the total variance
  4. accuracy against running time
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ORDER = ["phi_hat", "mc_2000", "mc_500", "mc_125", "y0sq_x_trcov", "y0sq_x_qvar",
         "pagerank_src", "q_variance", "entropy", "n_uncertain", "out_degree"]


def cluster_boot(d: pd.Series, n_boot=20000, seed=0):
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
        REPO / "outputs/exp23c_owen/results.csv")
    df = pd.read_csv(path)
    have = [m for m in ORDER if m in set(df["method"])]
    u = df.drop_duplicates(["label", "seed", "contrast"])
    print(f"read: {path} ({len(df):,} rows)")
    print(f"{df.label.nunique()} settings  seeds {sorted(df.seed.unique())}  "
          f"contrasts {sorted(df.contrast.unique())}  estimator "
          f"{sorted(df.get('reference_estimator', pd.Series(['?'])).unique())}")
    print(f"**all {len(u)} conditions used** (no split-half >= 0.7 exclusion)")

    print("\n" + "=" * 86)
    print("[0] Reproducibility of the reference (split-half)")
    print("=" * 86)
    print(u.groupby(["label", "contrast"])["reference_split_half"].mean()
          .unstack().round(4).to_string())
    print(f"\nmean over all conditions {u.reference_split_half.mean():.4f}  "
          f"minimum {u.reference_split_half.min():.4f}")

    print("\n" + "=" * 86)
    print("[1] Spearman correlation of the source ordering")
    print("=" * 86)
    print(df.pivot_table(index="contrast", columns="method", values="spearman")
          .reindex(columns=have).round(4).to_string())
    print("\nmean over contrasts:")
    print(df.groupby("method")["spearman"].mean().reindex(have).round(4).to_string())

    piv = df.pivot_table(index=["label", "seed", "contrast"], columns="method",
                         values="spearman")
    print("\n-- paired difference against phi_hat, stratified cluster bootstrap --")
    print(f"{'competitor':16s} {'mean diff':>10s} {'95% CI':>22s}  sig")
    for b in have[1:]:
        if b not in piv:
            continue
        m, lo, hi = cluster_boot((piv["phi_hat"] - piv[b]).dropna())
        print(f"{b:16s} {m:+8.4f}   [{lo:+.4f}, {hi:+.4f}]  {'YES' if lo > 0 else 'no'}")

    print("\n" + "=" * 86)
    print("[2] Agreement on the top source, and top-set overlap")
    print("=" * 86)
    cols = {"top1_hit": "top1 hit", "top5_overlap": "top5 overlap"}
    if "ref_top1_rank" in df:
        cols["ref_top1_rank"] = "rank of reference top1"
    print(df.assign(top1_hit=df["top1_hit"].astype(float)).groupby("method")[list(cols)].mean().rename(columns=cols)
          .reindex(have).round(4).to_string())

    print("\n" + "=" * 86)
    print("[3] Error in the effect sizes, normalised by the total variance\n"
          "    (both numerator and denominator are estimates)")
    print("=" * 86)
    if "abs_err_over_varz" in df:
        e = df.groupby("method")[["abs_err_over_varz", "max_err_over_varz"]].mean()
        print(e.reindex([m for m in have if m in e.index]).dropna(how="all").round(6).to_string())

    print("\n" + "=" * 86)
    print("[4] Accuracy against running time")
    print("=" * 86)
    t = df.groupby("method").agg(rho=("spearman", "mean"),
                                 sec=("seconds", "mean"),
                                 solves=("solves", "mean"))
    print(t.reindex(have).round(4).to_string())
    print(f"\nreference (two Owen runs): {u.reference_seconds.mean():.0f}s, "
          f"{u.reference_solves.mean():.0f} solves")


if __name__ == "__main__":
    main()
