"""Aggregate experiment 21: does the dV advantage survive a finite budget?

The gate:
  1. dV must beat the best cheap baseline in a paired comparison, consistently
  2. that advantage must survive a practical Monte Carlo budget
  3. it must show a better time-loss curve than evaluating the value of
     information per candidate by sampling

Point 3 is not judged here, because select_seconds is currently the total over
all six methods.

On the resampling unit:
  Repetitions of the decoder share the selection and the evaluation worlds, so
  they are not independent as far as selection quality goes, and the budgets are
  close to nested on one graph. The **cluster is therefore a (setting, seed)
  pair**, and resampling is **stratified by setting** so that all seven settings
  are represented every time. Resampling rows, or (setting, seed, budget)
  triples, independently would narrow the intervals spuriously.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
BASELINES = ["pagerank_src", "q_variance", "inv_outdeg", "bridge", "random"]
N_BOOT = 20000


def stratified_cluster_boot(
    d: pd.Series, n_boot: int = N_BOOT, seed: int = 0, stratify: bool = True
) -> tuple[float, float, float, int]:
    """Stratified cluster bootstrap.

    Args:
        d: a series of differences indexed by (label, seed, ...).
    Returns:
        (mean difference, CI lower, CI upper, number of clusters).
    """
    d = d.dropna()
    if d.empty:
        return (np.nan, np.nan, np.nan, 0)
    # (label, seed) is the cluster and label the stratum
    clusters = {key: sub.to_numpy() for key, sub in d.groupby(level=[0, 1])}
    strata: dict[str, list[np.ndarray]] = {}
    for (label, _seed), arr in clusters.items():
        # Without stratification every cluster goes into one stratum, which is
        # the plain 35-cluster bootstrap
        strata.setdefault(label if stratify else "_all", []).append(arr)

    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        parts = []
        for arrs in strata.values():  # resample clusters within each stratum
            idx = rng.integers(0, len(arrs), size=len(arrs))
            parts.extend(arrs[j] for j in idx)
        means[b] = np.concatenate(parts).mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(d.mean()), float(lo), float(hi), len(clusters))


def compare(piv: pd.DataFrame, baselines=BASELINES, stratify: bool = True) -> pd.DataFrame:
    """The paired difference between dV_edge and each baseline. Negative favours dV."""
    rows = []
    for b in baselines:
        if b not in piv.columns:
            continue
        d = (piv["dV_edge"] - piv[b]).dropna()
        mean, lo, hi, ncl = stratified_cluster_boot(d, stratify=stratify)
        wins, ties, losses = int((d < 0).sum()), int((d == 0).sum()), int((d > 0).sum())
        nontied = wins + losses
        rows.append({
            "baseline": b, "n": d.size, "clusters": ncl,
            "mean_diff": mean, "ci_lo": lo, "ci_hi": hi,
            "win": wins, "tie": ties, "loss": losses,
            "strict_win_rate": wins / d.size if d.size else np.nan,
            # The win rate excluding ties; not to be confused with the strict rate
            "win_rate_ex_ties": wins / nontied if nontied else np.nan,
            "significant": bool(hi < 0),
        })
    return pd.DataFrame(rows).sort_values("mean_diff")


def show(df: pd.DataFrame) -> str:
    cols = ["baseline", "mean_diff", "ci_lo", "ci_hi", "win", "tie", "loss",
            "strict_win_rate", "win_rate_ex_ties", "significant"]
    return df[cols].round(5).to_string(index=False)


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        REPO / "outputs/exp21_mc_decoder/results.csv"
    )
    df = pd.read_csv(path)
    print(f"read: {path}  ({len(df):,} rows)")
    print(f"{df['label'].nunique()} settings  seeds {sorted(df['seed'].unique())}  "
          f"methods {sorted(df['method'].unique())}")
    if "decode_cached" in df:
        print(f"share of decode_cached: {df['decode_cached'].mean():.1%}")
    print("\nthe settings (the generator overrides n from the config):")
    for lab, g in df.groupby("label"):
        print(f"  {lab:24s} n={g.n.iloc[0]:3d} m={g.m.iloc[0]:3d} "
              f"uncertain={g.n_uncertain.iloc[0]:3d} "
              f"tied mass={g.tied_mass.mean():.4f}")

    # R_star does not depend on the decoder, so drop the duplicates
    sel = df.drop_duplicates(["label", "seed", "budget", "method"])
    piv_star = sel.pivot_table(
        index=["label", "seed", "budget"], columns="method", values="regret_star"
    )

    print("\n" + "=" * 96)
    print("[1] Regret at R_star: the infinite-sample limit, i.e. selection quality")
    print("=" * 96)
    print(sel.groupby("method")["regret_star"].agg(["mean", "std", "count"])
          .round(5).sort_values("mean").to_string())
    print("\n-- paired difference against dV_edge (negative favours dV; cluster\n"
          "   bootstrap stratified over the seven settings) --")
    print(show(compare(piv_star)))
    print("\n-- for reference: the unstratified 35-cluster bootstrap. Stratifying\n"
          "   removes the between-setting variation and narrows the interval --")
    print(show(compare(piv_star, stratify=False)))

    print("\n" + "=" * 96)
    print("[2] Total regret at a finite Monte Carlo budget, decoder error included")
    print("=" * 96)
    for n_mc, sub in df.groupby("decoder_samples"):
        agg = sub.groupby(["label", "seed", "budget", "method"])["total_regret"].mean()
        piv = agg.reset_index().pivot_table(
            index=["label", "seed", "budget"], columns="method", values="total_regret"
        )
        print(f"\n--- N={n_mc}, averaging repetitions first; {len(piv)} conditions ---")
        print(show(compare(piv)))

    print("\n" + "=" * 96)
    print("[3] Decoder error R_mc - R_star, which should fall with the sample size")
    print("=" * 96)
    print(df.groupby("decoder_samples")["decoder_gap"].agg(["mean", "std"])
          .round(5).to_string())

    print("\n" + "=" * 96)
    print("[4] Tied mass against the disadvantage of dV, shown on R_star, which\n"
          "    reflects selection alone")
    print("=" * 96)
    star = sel[sel["label"].str.startswith("Star")]
    for lab, sub in star.groupby("label"):
        p = sub.pivot_table(index=["label", "seed", "budget"],
                            columns="method", values="regret_star")
        print(f"\n-- {lab}  (n={sub.n.iloc[0]}, "
              f"tied mass {sub['tied_mass'].mean():.4f}) --")
        print(show(compare(p, ["pagerank_src"])))

    print("\n-- correlation over all settings, as evidence of little confounding --")
    tm = sel.groupby(["label", "seed", "budget"])["tied_mass"].first()
    d = (piv_star["dV_edge"] - piv_star["pagerank_src"]).reindex(tm.index)
    ok = d.notna()
    from scipy.stats import pearsonr, spearmanr
    print(f"  tied mass vs (dV - pagerank_src)  "
          f"Spearman {spearmanr(tm[ok], d[ok]).statistic:+.3f}"
          f"  Pearson {pearsonr(tm[ok], d[ok])[0]:+.3f}")
    q = pd.qcut(tm[ok], 4, duplicates="drop")
    print(d[ok].groupby(q, observed=True).agg(["mean", "count"]).round(5).to_string())

    print("\n" + "=" * 96)
    print("[5] dV against pagerank_src, the best cheap baseline, by budget")
    print("=" * 96)
    for bud, sub in sel.groupby("budget"):
        p = sub.pivot_table(index=["label", "seed"], columns="method", values="regret_star")
        r = compare(p, ["pagerank_src"]).iloc[0]
        print(f"budget {bud}: mean diff {r['mean_diff']:+.5f}  "
              f"CI[{r['ci_lo']:+.5f},{r['ci_hi']:+.5f}]  "
              f"win {r['win']}/tie {r['tie']}/loss {r['loss']}  "
              f"{'significant' if r['significant'] else 'not significant'}")


if __name__ == "__main__":
    main()
