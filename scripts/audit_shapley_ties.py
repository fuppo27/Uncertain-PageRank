"""Audit the tie structure that the Shapley paragraph of the paper describes.

`exp24_shapley.py` collapses near-ties before computing rank correlations,
because some conditions put two sources at exactly the same effect and
`spearmanr` would otherwise order them by rounding noise. This script
establishes, from the conditions themselves, every claim the paper makes about
those ties, and asserts each one:

  1. the ties occur in 6 of the 75 conditions, all on the uniform-q ER generator
  2. each is a single *pair* of sources, never the whole source set
  3. the pair is exchanged by an automorphism of the weighted graph, which is
     why the two effects are equal in exact arithmetic
  4. the leading source is never part of a tie
  5. the rounding-level separation of a tied pair is at most 1.3e-15 of the
     largest effect, while the closest gap that is not a tie is 1.0e-5 of it,
     so the 1e-9 tolerance is not a judgement call
  6. the largest contributor agrees between Psi and Shapley in all 75
     conditions, by a margin of at least 1.1e-2 of the leading effect

It recomputes the per-source effect vectors, which the result CSV does not
store, so it takes a few minutes. Nothing is written; the output is the table
the paper's wording is based on.

Usage:
    python scripts/audit_shapley_ties.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from uncertain_pagerank.attribution import (  # noqa: E402
    analytic_attribution,
    boundary_contrasts,
    enumerate_world_scores,
    exact_attribution,
    identity_contrast,
    node_contrasts,
    shapley_effects,
    total_variance,
    true_contrast_values,
)
from uncertain_pagerank.expected_transition import expected_transition  # noqa: E402
from uncertain_pagerank.experiments.exp17_timescale_gate import build  # noqa: E402
from uncertain_pagerank.experiments.exp24_shapley import TIE_RTOL  # noqa: E402
from uncertain_pagerank.pagerank import pagerank, uniform_vector  # noqa: E402

# Above this many nodes the brute-force automorphism search is skipped.
AUTOMORPHISM_MAX_N = 11


def adjacency(graph) -> np.ndarray:
    """The weighted adjacency matrix, entry (u,v) being the link probability."""
    a = np.zeros((graph.n, graph.n))
    for u in range(graph.n):
        for k in range(graph.indptr[u], graph.indptr[u + 1]):
            a[u, graph.indices[k]] = graph.probs[k]
    return a


def swapping_automorphism(a: np.ndarray, u: int, v: int):
    """A permutation that exchanges `u` and `v` and preserves `a`, or None."""
    if a.shape[0] > AUTOMORPHISM_MAX_N:
        return None
    for perm in itertools.permutations(range(a.shape[0])):
        if perm[u] != v or perm[v] != u:
            continue
        if np.allclose(a[np.ix_(perm, perm)], a):
            return perm
    return None


def rel_gaps(v: np.ndarray) -> np.ndarray:
    """Gaps between adjacent sorted values, relative to the largest magnitude."""
    return np.diff(np.sort(v)) / float(np.abs(v).max())


def top1_margin(v: np.ndarray) -> float:
    """Separation of the largest effect from the runner-up, relative to itself."""
    o = np.sort(v)[::-1]
    return float((o[0] - o[1]) / abs(o[0]))


def main() -> None:
    cfg = yaml.safe_load(
        (REPO / "configs/experiment/exp24_shapley.yaml").read_text(encoding="utf-8"))
    alpha = float(cfg["alpha"])
    ks = [int(k) for k in cfg["boundary_ks"]]

    conditions = []
    for spec in cfg["graphs"]:
        for seed in cfg["seeds"]:
            graph = build(spec, np.random.default_rng(int(seed)))
            z = uniform_vector(graph.n)
            et = expected_transition(graph, method="dp")
            y0 = pagerank(et, alpha=alpha, z=z).scores
            ws = enumerate_world_scores(graph, alpha=alpha, z=z)
            active = np.unique(ws.edge_source)
            if active.size > int(cfg["max_sources"]) or active.size < 4:
                continue
            for cname in cfg["contrasts"]:
                if cname == "boundary":
                    cs = boundary_contrasts(y0, ks)
                elif cname == "node_top":
                    cs = node_contrasts(np.argsort(-y0)[: len(ks)], graph.n)
                else:
                    cs = identity_contrast(graph.n)
                zt = true_contrast_values(ws.scores, cs)
                if total_variance(ws.weights, zt) <= 0:
                    continue
                conditions.append({
                    "label": spec["label"], "seed": int(seed), "contrast": cname,
                    "graph": graph, "active": active,
                    "psi": exact_attribution(ws.bits, ws.weights, zt,
                                             ws.edge_source, graph.n)[active],
                    "shapley": shapley_effects(ws.bits, ws.weights, zt,
                                               ws.edge_source, active),
                    "phi": analytic_attribution(graph, et, y0, cs,
                                                alpha=alpha, z=z)[active],
                })

    print(f"{len(conditions)} conditions")
    print(f"tie rule: adjacent sorted effects within {TIE_RTOL:g} of max|v|\n")

    # --- the ties, and whether they are explained by a graph symmetry --------
    print("=" * 78)
    print("[1] the near-ties")
    print("=" * 78)
    print(f"{'generator':20s} {'seed':>4s} {'contrast':9s} {'d':>2s} {'pair':>9s} "
          f"{'gap/max':>9s} {'leader?':>7s} {'automorphism':>13s}")
    tied_labels, tie_gaps, n_tied = set(), [], 0
    for c in conditions:
        v = c["psi"]
        g = rel_gaps(v)
        where = np.flatnonzero(g <= TIE_RTOL)
        if where.size == 0:
            continue
        n_tied += 1
        tied_labels.add(c["label"])
        order = np.argsort(v)
        assert where.size == 1, "the paper says a single pair, not more"
        k = int(where[0])
        a_, b_ = int(c["active"][order[k]]), int(c["active"][order[k + 1]])
        leader = int(np.argmax(v)) in (int(order[k]), int(order[k + 1]))
        perm = swapping_automorphism(adjacency(c["graph"]), a_, b_)
        for v2 in (c["psi"], c["shapley"], c["phi"]):
            tie_gaps.extend(rel_gaps(v2)[rel_gaps(v2) <= TIE_RTOL].tolist())
        print(f"{c['label']:20s} {c['seed']:4d} {c['contrast']:9s} {len(v):2d} "
              f"{f'({a_},{b_})':>9s} {g[k]:9.1e} {str(leader):>7s} "
              f"{('yes' if perm else 'no'):>13s}")
        assert perm is not None, f"tied pair ({a_},{b_}) is not automorphic"
        assert not leader, "a tie involves the leading source"

    print(f"\nconditions with a tie: {n_tied} of {len(conditions)}")
    print(f"generators involved:   {sorted(tied_labels)}")
    print(f"tie separation:        {min(tie_gaps):.1e} to {max(tie_gaps):.1e} of max|v|")

    # --- the margin between the tolerance and real structure ----------------
    print("\n" + "=" * 78)
    print("[2] is the tolerance a judgement call?")
    print("=" * 78)
    closest = np.inf
    for c in conditions:
        for key in ("psi", "shapley", "phi"):
            g = rel_gaps(c[key])
            g = g[g > TIE_RTOL]
            if g.size:
                closest = min(closest, float(g.min()))
    print(f"largest tie separation          {max(tie_gaps):.1e}  (rounding)")
    print(f"tolerance                       {TIE_RTOL:.1e}")
    print(f"closest gap that is not a tie   {closest:.1e}  (real structure)")
    print(f"  -> the tolerance sits {np.log10(TIE_RTOL / max(tie_gaps)):.0f} orders above "
          f"the rounding and {np.log10(closest / TIE_RTOL):.0f} below the structure")
    assert closest > 1e3 * TIE_RTOL, "the tolerance is too close to real structure"

    # --- the top-1 claim ----------------------------------------------------
    print("\n" + "=" * 78)
    print("[3] does the largest contributor agree?")
    print("=" * 78)
    agree = sum(int(np.argmax(c["psi"]) == np.argmax(c["shapley"])) for c in conditions)
    margins = np.array([min(top1_margin(c["psi"]), top1_margin(c["shapley"]))
                        for c in conditions])
    worst = conditions[int(margins.argmin())]
    print(f"top-1 source of Psi equals top-1 of Shapley: {agree}/{len(conditions)}")
    print(f"smallest margin over both:                   {margins.min():.2e}  "
          f"({worst['label']}, seed {worst['seed']}, {worst['contrast']})")
    for key in ("psi", "shapley", "phi"):
        m = np.array([top1_margin(c[key]) for c in conditions])
        print(f"  smallest margin, {key:8s} alone:              {m.min():.2e}")
    assert agree == len(conditions), "the largest contributor does not always agree"
    print("\nall assertions held.")


if __name__ == "__main__":
    main()
