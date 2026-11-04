"""Sensitivity to the dangling convention.

  Convention A (used elsewhere): a row with K_u=0 becomes z, so
                                 Pbar = B + delta z^T.
  Convention B:                  a row with K_u=0 becomes e_u, the node stays
                                 put, so Pbar' = B + diag(delta).

**Every transition with K_u>0 is identical under both.** An earlier version
compared by adding a certain self-loop at every node, which turned xi/K_u into
xi/(K_u+1) and so moved transitions that are not dangling at all. That is a
different question, about self-loop augmentation.

Changing the convention moves y0, and hence the rank boundaries, and hence the
contrast itself. Both comparisons are therefore reported: one with the contrast
held fixed, isolating the effect of the transition model, and one where each
convention builds the contrast from its own y0, as it would be in use.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from uncertain_pagerank.attribution import (
    analytic_attribution, attribution_selfloop_dangling, boundary_contrasts,
    selfloop_dangling_transition)
from uncertain_pagerank.datasets import load_dlbl, load_edgelist
from uncertain_pagerank.expected_transition import expected_transition
from uncertain_pagerank.pagerank import pagerank, uniform_vector

NETS = [("Trackback NW", "data/raw/original/trackback/blog34.dlbl", load_dlbl),
        ("wiki-Vote", "data/raw/snap/wiki-Vote.txt.gz", load_edgelist),
        ("soc-Epinions1", "data/raw/snap/soc-Epinions1.txt.gz", load_edgelist),
        ("Follow NW", "data/raw/original/follow/ameblo.dlbl", load_dlbl)]


def main() -> None:
    rows: list[dict] = []
    print(f"{'network':14s} {'rho (fixed)':>12s} {'top1 same':>10s} "
          f"{'rho (own)':>10s} {'top1 same':>10s} {'top1 share A':>13s} "
          f"{'top1 share B':>13s} {'dangling':>9s}")
    for name, p, loader in NETS:
        g0 = loader(REPO / p).graph
        g = g0.with_probs(np.random.default_rng(20260923).beta(7, 3, size=g0.m))
        z = uniform_vector(g.n)
        et = expected_transition(g, method="dp")
        y0A = pagerank(et, alpha=0.85, z=z, tol=1e-12).scores
        et_mod = selfloop_dangling_transition(et)
        y0B = pagerank(et_mod, alpha=0.85, z=z, tol=1e-12).scores
        csA, csB = boundary_contrasts(y0A, [10, 100, 1000]), boundary_contrasts(y0B, [10, 100, 1000])
        phiA = analytic_attribution(g, et, y0A, csA, alpha=0.85, z=z)
        phiB_same = attribution_selfloop_dangling(g, et, y0B, csA, alpha=0.85, z=z)
        phiB_own = attribution_selfloop_dangling(g, et, y0B, csB, alpha=0.85, z=z)
        act = phiA > 0
        sA = np.sort(phiA)[::-1] / phiA.sum()
        sB = np.sort(phiB_own)[::-1] / phiB_own.sum()
        same_top1 = int(np.argmax(phiA)) == int(np.argmax(phiB_same))
        own_top1 = int(np.argmax(phiA)) == int(np.argmax(phiB_own))
        print(f"{name:14s} {spearmanr(phiA[act], phiB_same[act]).statistic:11.4f} "
              f"{str(same_top1):>14s} "
              f"{spearmanr(phiA[act], phiB_own[act]).statistic:11.4f} "
              f"{str(own_top1):>14s} {sA[0]:11.4f} {sB[0]:11.4f} "
              f"{et.delta.mean():10.4f}", flush=True)
        rows.append({"network": name,
                     "rho_contrast_fixed": float(spearmanr(phiA[act], phiB_same[act]).statistic),
                     "top1_same_fixed": same_top1,
                     "rho_contrast_rebuilt": float(spearmanr(phiA[act], phiB_own[act]).statistic),
                     "top1_same_rebuilt": own_top1,
                     "top1_share_A": float(sA[0]), "top1_share_B": float(sB[0]),
                     "dangling_rate": float(et.delta.mean())})

    out = REPO / "outputs/dangling_sensitivity.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
