"""Sensitivity to the model for q.

Checks whether the conclusions --- the ordering of the attribution and how
concentrated the shares are --- depend on how q was assigned.

The contrast is built from each assignment's own y0, as it would be in use. A
second comparison holds the contrast at the baseline. The first includes the
effect that changing q changes the ranking problem itself; the second isolates
how much the attribution for the same ranking problem moves with q.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from uncertain_pagerank.attribution import analytic_attribution, boundary_contrasts
from uncertain_pagerank.datasets import load_dlbl, load_edgelist
from uncertain_pagerank.expected_transition import expected_transition
from uncertain_pagerank.pagerank import pagerank, uniform_vector

QS = [
    ("Beta(7,3) mean 0.7", lambda r, m: r.beta(7, 3, size=m)),
    ("Beta(5,5) mean 0.5", lambda r, m: r.beta(5, 5, size=m)),
    ("Beta(18,2) mean 0.9", lambda r, m: r.beta(18, 2, size=m)),
    ("uniform q=0.5", lambda r, m: np.full(m, 0.5)),
    ("random U(0.1,0.9)", lambda r, m: r.uniform(0.1, 0.9, size=m)),
]
NETS = [
    ("Trackback NW", "data/raw/original/trackback/blog34.dlbl", load_dlbl),
    ("wiki-Vote", "data/raw/snap/wiki-Vote.txt.gz", load_edgelist),
    ("Follow NW", "data/raw/original/follow/ameblo.dlbl", load_dlbl),
]


def main() -> None:
    rows: list[dict] = []
    for name, p, loader in NETS:
        g0 = loader(REPO / p).graph
        z = uniform_vector(g0.n)
        res = {}
        for qname, qf in QS:
            g = g0.with_probs(qf(np.random.default_rng(20260923), g0.m))
            et = expected_transition(g, method="dp")
            y0 = pagerank(et, alpha=0.85, z=z, tol=1e-12).scores
            cs = boundary_contrasts(y0, [10, 100, 1000])
            res[qname] = (analytic_attribution(g, et, y0, cs, alpha=0.85, z=z),
                          cs, g, et, y0)
        base_cs = res[QS[0][0]][1]
        ref = res[QS[0][0]][0]
        act = ref > 0
        d1 = g0.out_degrees() == 1
        print(f"=== {name} (n={g0.n}, m={g0.m}) ===", flush=True)
        print(f"{'q model':22s} {'top1':>8s} {'top10':>8s} {'90% src':>9s} "
              f"{'D=1 share':>10s} {'rho (own)':>10s} {'rho (fixed)':>12s}")
        for qname, _ in QS:
            phi, cs, g, et, y0 = res[qname]
            s = np.sort(phi)[::-1] / phi.sum()
            phi_fix = analytic_attribution(g, et, y0, base_cs, alpha=0.85, z=z)
            print(f"{qname:20s} {s[0]:8.4f} {s[:10].sum():8.4f} "
                  f"{int(np.searchsorted(np.cumsum(s), 0.90)) + 1:10d} "
                  f"{phi[d1].sum() / phi.sum():8.1%} "
                  f"{spearmanr(ref[act], phi[act]).statistic:11.4f} "
                  f"{spearmanr(ref[act], phi_fix[act]).statistic:11.4f}", flush=True)
            rows.append({"network": name, "q_model": qname,
                         "top1_share": float(s[0]), "top10_share": float(s[:10].sum()),
                         "n_sources_for_90pct": int(np.searchsorted(np.cumsum(s), 0.90)) + 1,
                         "share_outdeg_one": float(phi[d1].sum() / phi.sum()),
                         "rho_contrast_rebuilt": float(spearmanr(ref[act], phi[act]).statistic),
                         "rho_contrast_fixed": float(spearmanr(ref[act], phi_fix[act]).statistic)})
        print(flush=True)

    out = REPO / "outputs/q_sensitivity.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
