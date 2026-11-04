"""Check sum_u phi_hat_u against the total variance tr Cov(Z) of the contrast.

Phi_hat decomposes the variance of the first-order surrogate over sources, so the
shares sum to the surrogate variance by construction. The open question is how
close that surrogate is to the true tr Cov(Z). Being an aggregate rather than a
per-source quantity, tr Cov(Z) can be estimated by drawing N worlds, which needs
only N solves.

Usage:
    python scripts/total_variance_check.py --budgets 500,1500
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from uncertain_pagerank.attribution import analytic_attribution, boundary_contrasts
from uncertain_pagerank.datasets import load_dlbl, load_edgelist
from uncertain_pagerank.expected_transition import expected_transition
from uncertain_pagerank.pagerank import pagerank, uniform_vector
from uncertain_pagerank.world_batch import batch_world_scores, sample_masks

NETS = {"trackback": ("Trackback NW", "data/raw/original/trackback/blog34.dlbl", load_dlbl),
        "wikivote": ("wiki-Vote", "data/raw/snap/wiki-Vote.txt.gz", load_edgelist)}


def main() -> None:
    rows: list[dict] = []
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", choices=sorted(NETS), help="both networks if omitted")
    ap.add_argument("--budgets", default="500,1500")
    ap.add_argument("--seed", type=int, default=20260923)
    a = ap.parse_args()

    for key in ([a.net] if a.net else sorted(NETS)):
        name, path, loader = NETS[key]
        g0 = loader(REPO / path).graph
        g = g0.with_probs(np.random.default_rng(20260923).beta(7, 3, size=g0.m))
        z = uniform_vector(g.n)
        et = expected_transition(g, method="dp")
        y0 = pagerank(et, alpha=0.85, z=z, tol=1e-12).scores
        cs = boundary_contrasts(y0, [10, 100, 1000])
        sum_phi = float(analytic_attribution(g, et, y0, cs, alpha=0.85, z=z).sum())

        for n in (int(x) for x in a.budgets.split(",")):
            rng = np.random.default_rng(a.seed + n)
            t = time.perf_counter()
            masks = sample_masks(g, n, rng)
            scores = batch_world_scores(g, masks, alpha=0.85, z=z, tol=1e-10)
            secs = time.perf_counter() - t
            # tr Cov(Z) = sum_k lam_k Var(w_k^T y_X)
            vals = scores @ cs.W.T                      # (n, r)
            est = lambda v: float((cs.lam * v.var(axis=0, ddof=1)).sum())
            var_mc = est(vals)
            # The variance estimate carries sampling error of its own; the
            # question is whether sum_phi falls inside that width
            boot = np.array([est(vals[rng.integers(0, n, n)]) for _ in range(2000)])
            lo, hi = np.quantile(boot, [0.025, 0.975])
            inside = "inside" if lo <= sum_phi <= hi else "outside"
            print(f"{name:14s} N={n:5d}  trCov(Z)_MC={var_mc:.6e} "
                  f"CI[{lo:.4e},{hi:.4e}] (relative width {(hi-lo)/var_mc:.3f})  "
                  f"sum_phi={sum_phi:.6e} ratio={sum_phi/var_mc:.4f} {inside} CI  "
                  f"({secs:.0f}s, {n} solves)", flush=True)
            rows.append({"network": name, "n_samples": n, "trcov_mc": var_mc,
                         "ci_lo": lo, "ci_hi": hi, "sum_phi": sum_phi,
                         "ratio": sum_phi / var_mc, "inside_ci": lo <= sum_phi <= hi,
                         "seconds": secs})

    out = REPO / "outputs/total_variance_check.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
