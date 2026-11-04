"""Aggregate the budget curve by within-stratum correlation.

The overall Spearman correlation saturates at a small budget, because the strata
span more than ten orders of magnitude. The informative quantity is the
within-stratum correlation, so the mean over the strata is the headline figure
and the overall correlation is reported alongside it. Both are averaged over the
three repetitions, with the range.

The time comes from `real_budget_curve_<net>.csv`, not from the per-repetition
files. The three repetitions were run concurrently, so the `seconds` they record
is inflated by contention (Trackback at 6,250 solves: 722 s there against 415 s
measured on its own) and is not a cost. The paper quotes the sequential figure.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
NETS = {"trackback": "Trackback NW", "wikivote": "wiki-Vote"}
REPS = (1, 2, 3)


def rho(ref, est):
    if np.ptp(est) == 0.0 or np.ptp(ref) == 0.0:
        return np.nan
    return float(spearmanr(ref, est).statistic)


def main() -> None:
    for net, name in NETS.items():
        # per[(method, rep)] = {stratum: ([ref], [est])}
        per = defaultdict(lambda: defaultdict(lambda: ([], [])))
        solves: dict[str, int] = {}
        for rep in REPS:
            f = REPO / f"outputs/real_budget_psi_{net}_r{rep}.csv"
            if not f.exists():
                print(f"[skip] {f.name} is missing", file=sys.stderr)
                continue
            for r in csv.DictReader(f.open(encoding="utf-8")):
                key = (r["method"], rep)
                a, b = per[key][r["stratum"]]
                a.append(float(r["psi_ref"]))
                b.append(float(r["psi"]))
                per[key]["__all__"][0].append(float(r["psi_ref"]))
                per[key]["__all__"][1].append(float(r["psi"]))
                solves[r["method"]] = int(r["solves"])

        # The timed run, measured one network at a time with nothing else going.
        timed = REPO / f"outputs/real_budget_curve_{net}.csv"
        secs: dict[str, float] = {}
        if timed.exists():
            for r in csv.DictReader(timed.open(encoding="utf-8")):
                secs[r["method"]] = float(r["seconds"])
        else:
            print(f"[warn] {timed.name} is missing; no time column",
                  file=sys.stderr)

        methods = sorted({m for m, _ in per},
                         key=lambda m: (0, 0) if m == "phi_hat" else (1, int(m.split("_")[1])))
        print(f"\n=== {name} ===")
        print(f"{'method':>10} {'solves':>8} {'sec':>8}  "
              f"{'within-stratum':>18}  {'overall':>18}")
        print(f"{'':>10} {'':>8} {'(timed':>8}  {'mean over 3 reps':>18}  "
              f"{'[min,max]':>18}")
        print(f"{'':>10} {'':>8} {'alone)':>8}")
        for m in methods:
            wi, ov = [], []
            for rep in REPS:
                if (m, rep) not in per:
                    continue
                d = per[(m, rep)]
                strata = [s for s in d if s != "__all__"]
                vals = [rho(np.array(d[s][0]), np.array(d[s][1])) for s in sorted(strata)]
                wi.append(float(np.nanmean(vals)))
                ov.append(rho(np.array(d["__all__"][0]), np.array(d["__all__"][1])))
            sv, sec = solves[m], secs.get(m)
            def fmt(x):
                x = np.array(x)
                return (f"{x.mean():.3f} [{x.min():.3f},{x.max():.3f}]" if x.size > 1
                        else f"{x.mean():.3f}")
            st = f"{sec:8.1f}" if sec is not None else f"{'--':>8}"
            print(f"{m:>10} {sv:8d} {st}  {fmt(wi):>18}  {fmt(ov):>18}")


if __name__ == "__main__":
    main()
