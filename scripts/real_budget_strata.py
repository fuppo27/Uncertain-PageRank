"""The budget curve on real networks, within stratum.

The overall Spearman correlation saturates, because the strata span ten to
eleven orders of magnitude: on wiki-Vote it reaches 0.96 already at N=10 and
then moves with no relation to the budget. The informative quantity is the
within-stratum correlation, so the curve is traced there. The per-source
estimates are saved, which lets the per-stratum correlations be computed later.

Seeds are independent of the reference, across budgets and across repetitions.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

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
    ap.add_argument("--rep", type=int, required=True,
                    help="repetition index; separates the seeds")
    ap.add_argument("--budgets", default="10,25,50,125")
    a = ap.parse_args()

    name, path, loader = NETS[a.net]
    saved = REPO / f"outputs/real_per_source_{a.net}.csv"
    d = {int(r["source"]): r for r in csv.DictReader(saved.open(encoding="utf-8"))}
    cand = np.array(sorted(d), dtype=np.int64)
    stratum = [d[int(u)]["stratum"] for u in cand]
    ref = np.array([float(d[int(u)]["psi_ref"]) for u in cand])
    phi_saved = np.array([float(d[int(u)]["phi_hat"]) for u in cand])

    g0 = loader(REPO / path).graph
    g = g0.with_probs(np.random.default_rng(20260923).beta(7, 3, size=g0.m))
    z = uniform_vector(g.n)
    et = expected_transition(g, method="dp")
    y0 = pagerank(et, alpha=0.85, z=z, tol=1e-12).scores
    cs = boundary_contrasts(y0, [10, 100, 1000])

    t0 = time.perf_counter()
    phi = analytic_attribution(g, et, y0, cs, alpha=0.85, z=z)
    t_phi = time.perf_counter() - t0
    assert np.allclose(phi[cand], phi_saved, rtol=1e-9), "phi_hat differs from the saved values"
    print(f"=== {name} rep{a.rep}: {cand.size} candidates, phi_hat {t_phi:.2f}s ===",
          flush=True)

    rows = [(u, s, "phi_hat", 0, 0, t_phi, float(pv), float(rv))
            for u, s, pv, rv in zip(cand, stratum, phi[cand], ref)]
    for N in (int(x) for x in a.budgets.split(",")):
        r = pickfreeze_attribution_owen(g, cs, sources=cand, n_samples=N,
                                        seed=700000 + 1000 * a.rep + N,
                                        alpha=0.85, z=z, tol=1e-8, chunk=256)
        rows += [(u, s, f"owen_{N}", N, r.n_solves, r.seconds, float(pv), float(rv))
                 for u, s, pv, rv in zip(cand, stratum, r.psi, ref)]
        print(f"  Owen N={N:5d}  {r.n_solves:7d} solves  {r.seconds:8.0f}s", flush=True)

    out = REPO / f"outputs/real_budget_psi_{a.net}_r{a.rep}.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "stratum", "method", "n_samples", "solves", "seconds",
                    "psi", "psi_ref"])
        w.writerows(rows)
    print(f"  -> {out}", flush=True)


if __name__ == "__main__":
    main()
