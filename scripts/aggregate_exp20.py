"""Aggregate experiment 20: the Gaussian decoder against the Bayes-optimal report.

Prints the table of the supplemental material on the two negative results, plus
the three checks that rule out numerical explanations for the failure.

The excess loss is the exact loss of the set the decoder reports, minus the loss
of the Bayes-optimal report. It is not `base_loss_gap`, which is a different
quantity: how badly the decoder predicts its own risk, signed in both
directions.

Usage:
    python scripts/aggregate_exp20.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]

# The order the table uses: by excess, cheapest first.
LABELS = {
    "ER n=16 Beta": "ER, Beta q",
    "ER n=16 mixed certain": "ER, mixed certain",
    "ER n=16 uniform q=0.5": "ER, uniform q=0.5",
    "Cycle with 2 chords": "Cycle with chords",
    "Preferential attachment n=16": "Preferential attach.",
    "Star (6 bidirectional spokes)": "Star",
}


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        REPO / "outputs/exp20_analytic/results.csv")
    d = pd.read_csv(path)
    d["excess"] = d.loss_of_reported_set_exact - d.base_loss_exact
    print(f"read: {path} ({len(d):,} rows)")
    print(f"{d.label.nunique()} generators  seeds {sorted(d.seed.unique())}  "
          f"tie orders {sorted(d.tie_variant.unique())}  "
          f"budgets {sorted(d.budget.unique())}")
    unknown = set(d.label) - set(LABELS)
    assert not unknown, f"generator not in the table: {sorted(unknown)}"

    print("\n" + "=" * 70)
    print("[1] Excess symmetric-difference loss over the Bayes-optimal report")
    print("=" * 70)
    g = d.groupby("label")[["base_loss_exact", "loss_of_reported_set_exact",
                            "excess"]].mean()
    print(f"{'Generator':22s} {'Bayes':>7s} {'Gaussian':>9s} {'excess':>8s}")
    for key in sorted(LABELS, key=lambda k: g.loc[k, "excess"]):
        r = g.loc[key]
        print(f"{LABELS[key]:22s} {r.base_loss_exact:7.3f} "
              f"{r.loss_of_reported_set_exact:9.3f} {r.excess:+8.3f}")
    print(f"\nexcess, mean over rows            {d.excess.mean():.3f}")
    print(f"excess, mean over generator means {g.excess.mean():.3f}")
    star = [k for k in LABELS if k.startswith("Star")][0]
    print(f"on the star the reported set costs "
          f"{100 * g.loc[star, 'excess'] / g.loc[star, 'base_loss_exact']:.0f}% "
          f"more than the optimum")

    print("\n" + "=" * 70)
    print("[2] Does the decoder misjudge its own risk?")
    print("    predicted minus true, so negative means understating")
    print("=" * 70)
    p = d.groupby("label")[["base_loss_exact", "base_loss_analytic",
                            "base_loss_gap"]].mean()
    for key in LABELS:
        r = p.loc[key]
        way = "understates" if r.base_loss_gap < 0 else "overstates "
        print(f"{LABELS[key]:22s} true {r.base_loss_exact:6.3f}  "
              f"predicted {r.base_loss_analytic:6.3f}  {way} "
              f"({r.base_loss_gap:+.3f})")

    print("\n" + "=" * 70)
    print("[3] Three explanations the numbers rule out")
    print("=" * 70)
    print(f"covariance conditioning: clipped eigenvalue mass never exceeds "
          f"{d.clipped_eigenvalue_mass.max():.1e}")
    print(f"candidate truncation:    mean coverage {d.mean_coverage.mean():.3f}, "
          f"full coverage {d.full_coverage.mean():.3f}")
    full = d[d.full_coverage == 1.0]
    if not full.empty:
        fg = full.groupby("label").excess.mean()
        print("                         on the conditions with full coverage the "
              "pattern is unchanged:")
        for key in sorted(LABELS, key=lambda k: fg.get(k, np.nan)):
            if key in fg:
                print(f"                           {LABELS[key]:22s} {fg[key]:+.3f}")


if __name__ == "__main__":
    main()
