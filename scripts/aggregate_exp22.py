"""Aggregate experiment 22: does the attribution predict the true source effect?

Resampling uses the same unit as experiment 21: the **cluster is a
(setting, seed) pair** and the strata are the settings.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ORDER = ["phi_hat", "phi_no_y0sq", "y0sq_x_trcov", "y0sq_x_qvar", "y0sq_x_entropy", "pagerank_src",
         "phi_randdir", "entropy", "q_variance", "n_uncertain", "out_degree", "random"]


def cluster_boot(d: pd.Series, n_boot=20000, seed=0, stratify=True):
    d = d.dropna()
    if d.empty:
        return (np.nan, np.nan, np.nan)
    cl = {k: v.to_numpy() for k, v in d.groupby(level=[0, 1])}
    st: dict = {}
    for (lab, _s), a in cl.items():
        st.setdefault(lab if stratify else "_all", []).append(a)
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
        REPO / "outputs/exp22_source_attribution/results.csv")
    df = pd.read_csv(path)
    have = [m for m in ORDER if m in set(df["method"])]
    print(f"read: {path}  ({len(df):,} rows)")
    print(f"{df['label'].nunique()} settings  seeds {sorted(df['seed'].unique())}  "
          f"contrast {sorted(df['contrast'].unique())}")
    u = df.drop_duplicates(["label", "seed", "contrast"])
    print("\nthe settings:")
    for lab, g in df.groupby("label"):
        print(f"  {lab:30s} n={g.n.iloc[0]:3d} uncertain={g.n_uncertain.iloc[0]:3d} "
              f"worlds={g.n_worlds.iloc[0]:7d} "
              f"active sources={g.n_active_sources.mean():.1f}")

    print("\n" + "=" * 94)
    print("[1] Rank correlation with the true Psi_u, by contrast")
    print("=" * 94)
    print(df.pivot_table(index="contrast", columns="method", values="spearman")
          .reindex(columns=have).round(3).to_string())
    print("\nmean over contrasts:")
    print(df.groupby("method")["spearman"].mean().reindex(have).round(3).to_string())

    piv = df.pivot_table(index=["label", "seed", "contrast"],
                         columns="method", values="spearman")
    print("\n" + "=" * 94)
    print("[2] Paired difference against phi_hat (stratified cluster bootstrap;\n"
          "    positive favours phi_hat)")
    print("=" * 94)
    print(f"{'baseline':16s} {'mean diff':>10s} {'95% CI':>22s} "
          f"{'win/tie/loss':>14s}  sig")
    for b in have[1:]:
        d = (piv["phi_hat"] - piv[b]).dropna()
        m, lo, hi = cluster_boot(d)
        print(f"{b:16s} {m:+8.3f}   [{lo:+.3f}, {hi:+.3f}]   "
              f"{int((d>0).sum()):3d}/{int((d==0).sum()):3d}/{int((d<0).sum()):3d}  "
              f"{'YES' if lo > 0 else 'no'}")

    print("\n" + "=" * 94)
    print("[3] Top-source agreement and set-selection regret, in true variance\n"
          "    reduction")
    print("=" * 94)
    agg = df.groupby("method").agg(
        top1=("top1_hit", "mean"), top3=("top3_overlap", "mean"),
        regret_b1=("regret_b1", "mean"), regret_b2=("regret_b2", "mean"),
        regret_b3=("regret_b3", "mean"),
        judgekai_b2=("revealed_edges_b2", "mean"))
    agg = agg.rename(columns={"judgekai_b2": "revealed_edges_b2"})
    print(agg.reindex(have).round(5).to_string())
    print("\nedges revealed by the oracle: " + "  ".join(
        f"b{b}={df[f'oracle_revealed_b{b}'].mean():.2f}" for b in (1, 2, 3)))

    print("\n" + "=" * 94)
    print("[4] Interaction I_inter and linearization error (the true nonlinearity)")
    print("=" * 94)
    print(u.groupby("contrast")[["interaction_share", "lin_rel_err_median"]]
          .agg(["mean", "max"]).round(4).to_string())
    print(f"\nI_inter/Var overall: mean {u['interaction_share'].mean():.4f} "
          f"max {u['interaction_share'].max():.4f}")
    print(f"median relative linearization error: "
          f"mean {u['lin_rel_err_median'].mean():.4f} "
          f"max {u['lin_rel_err_median'].max():.4f}")
    print(f"\nenumeration {u['enumerate_seconds'].mean():.2f}s vs analytic "
          f"{u['analytic_seconds'].mean():.4f}s (ratio "
          f"{u['enumerate_seconds'].mean() / u['analytic_seconds'].mean():.0f}x)."
          "\nEnumeration is 2^m_u, so this ratio grows exponentially with the "
          "number of uncertain links.")


if __name__ == "__main__":
    main()
