"""The budget curve on real networks.

Quoting a speed-up against the 40,000 solves we chose for the reference compares
against our own choice. The comparison that matters is against the **smallest**
budget at which sampling reaches a usable level of agreement. Against the same 24
candidates and the same reference (the mean of the saved halves A and B), this
runs the Owen estimator at small budgets to trace the accuracy-time curve. The
random draws are independent of the reference and of each other.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from uncertain_pagerank.attribution import (
    analytic_attribution, boundary_contrasts, pickfreeze_attribution_owen)
from uncertain_pagerank.datasets import load_dlbl, load_edgelist
from uncertain_pagerank.expected_transition import expected_transition
from uncertain_pagerank.pagerank import pagerank, uniform_vector

NETS = {"trackback": ("Trackback NW", "data/raw/original/trackback/blog34.dlbl", load_dlbl),
        "wikivote": ("wiki-Vote", "data/raw/snap/wiki-Vote.txt.gz", load_edgelist)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True, choices=sorted(NETS))
    ap.add_argument("--budgets", default="125,250,500")
    a = ap.parse_args()

    name, path, loader = NETS[a.net]
    saved = REPO / f"outputs/real_per_source_{a.net}.csv"
    d = {int(r["source"]): r for r in csv.DictReader(saved.open(encoding="utf-8"))}
    cand = np.array(sorted(d), dtype=np.int64)
    ref = np.array([float(d[int(u)]["psi_ref"]) for u in cand])

    g0 = loader(REPO / path).graph
    g = g0.with_probs(np.random.default_rng(20260923).beta(7, 3, size=g0.m))
    z = uniform_vector(g.n)
    et = expected_transition(g, method="dp")
    y0 = pagerank(et, alpha=0.85, z=z, tol=1e-12).scores
    cs = boundary_contrasts(y0, [10, 100, 1000])

    t0 = time.perf_counter()
    phi = analytic_attribution(g, et, y0, cs, alpha=0.85, z=z)
    t_phi = time.perf_counter() - t0
    print(f"=== {name}: {cand.size} candidates, phi_hat {t_phi:.2f}s "
          f"(rho={spearmanr(ref, phi[cand]).statistic:.4f}) ===", flush=True)

    rows = [("phi_hat", 0, t_phi, 0, float(spearmanr(ref, phi[cand]).statistic))]
    for N in (int(x) for x in a.budgets.split(",")):
        r = pickfreeze_attribution_owen(g, cs, sources=cand, n_samples=N, seed=9000 + N,
                                        alpha=0.85, z=z, tol=1e-8, chunk=256)
        rho = float(spearmanr(ref, r.psi).statistic)
        rows.append((f"owen_{N}", N, r.seconds, r.n_solves, rho))
        print(f"  Owen N={N:5d}  {r.n_solves:7d} solves  {r.seconds:8.0f}s  rho={rho:.4f}",
              flush=True)

    # Keep the points already computed, so the curve lines up when the budgets
    # are run in separate invocations
    out = REPO / f"outputs/real_budget_curve_{a.net}.csv"
    fresh = {r[0] for r in rows}
    if out.exists():
        for r in csv.DictReader(out.open(encoding="utf-8")):
            if r["method"] not in fresh:
                rows.append((r["method"], int(r["n_samples"]), float(r["seconds"]),
                             int(r["solves"]), float(r["spearman_vs_ref"])))
    rows.sort(key=lambda r: r[1])
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "n_samples", "seconds", "solves", "spearman_vs_ref"])
        w.writerows(rows)
    print(f"  -> {out}", flush=True)


if __name__ == "__main__":
    main()
