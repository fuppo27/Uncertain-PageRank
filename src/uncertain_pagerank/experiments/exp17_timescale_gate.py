"""Experiment 17: a robustness gate on the time scale (short, deliberately limited).

This exists only to fix the limits of what we claim. The time scale itself is not
a main result.

What it checks:
1. Whether "tau = L(1-alpha) is not a good normalisation" reproduces across
   non-uniform q_e, degree heterogeneity and several seeds, by comparing the
   coefficient of variation of L_half and tau_half over alpha.
2. The monotonicity and nonlinearity of rho(L), and the behaviour of the top-k
   and rank correlations.
3. The distance from d_L to both p-avg and s-avg; one side alone hides the shape.

**We do not claim that L itself is universal.** On the star topology L_half
already moves from 1.68 to 4.82 with alpha. What can be said is only that
normalising by tau does not work across this set of settings.

Note also that L is the number of random-walk transitions for which a world stays
fixed, which is not the same thing as a time window in real data. Placing it
against real data would need a separate argument relating link autocorrelation
and persistence to the time scale of the walk.

Usage:
    python -m uncertain_pagerank.experiments.exp17_timescale_gate \\
        --config configs/experiment/exp17_timescale.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

from ..expected_transition import expected_transition
from ..generators import erdos_renyi_digraph, scale_free_digraph
from ..graph import UncertainGraph
from ..metrics import topk_overlap
from ..pagerank import pagerank, uniform_vector
from ..possible_worlds import exact_savg
from ..timescale import visit_distribution
from .runtime import save_run

REPO = Path(__file__).resolve().parents[3]


def build(spec: dict, rng: np.random.Generator) -> UncertainGraph:
    """Build a topology together with its connection probabilities."""
    if spec["topology"] == "er":
        g = erdos_renyi_digraph(int(spec["n"]), int(spec["m"]), rng)
    elif spec["topology"] == "pa":
        g = scale_free_digraph(int(spec["n"]), int(spec["m_out"]), rng)
    elif spec["topology"] == "star":
        # Here n is the number of nodes in the star. An isolated node left over
        # would make the label disagree with the instance, so the spoke count is
        # taken from m and n is matched to it.
        n, m = int(spec["n"]), int(spec["m"])
        spokes = min(m // 2, n - 1)
        edges = [e for i in range(1, spokes + 1) for e in ((0, i), (i, 0))]
        g = UncertainGraph.from_edges(spokes + 1, edges, 1.0)
    elif spec["topology"] == "cycle":
        # Actually add chords to the cycle. With m exactly the cycle length no
        # chord would fit, so the cycle length is derived from m to leave room
        # for them, and no isolated node is created.
        n, m = int(spec["n"]), int(spec["m"])
        n_chords = min(2, max(0, m - 3))
        length = max(3, min(m - n_chords, n))
        edges = [(i, (i + 1) % length) for i in range(length)]
        for cand in [(0, length // 2), (length // 3, (2 * length) // 3)]:
            if len(edges) < m and cand[0] != cand[1] and cand not in edges:
                edges.append(cand)
        g = UncertainGraph.from_edges(length, edges, 1.0)
    else:
        raise ValueError(f"unknown topology: {spec['topology']}")
    if spec["q"] == "uniform":
        return g.with_probs(np.full(g.m, float(spec["q_value"])))
    if spec["q"] == "mixed":
        # Make some links certain (q=1) and draw the rest from a Beta. This
        # catches confusion between the two index spaces: link ids in the
        # original graph and positions in the uncertain-link array.
        a, c = float(spec["q_mean"]), float(spec["q_concentration"])
        probs = np.clip(rng.beta(a * c, (1 - a) * c, size=g.m), 1e-6, 1 - 1e-6)
        n_certain = int(round(float(spec.get("certain_fraction", 0.25)) * g.m))
        probs[rng.choice(g.m, size=n_certain, replace=False)] = 1.0
        return g.with_probs(probs)
    if spec["q"] == "beta":
        a, c = float(spec["q_mean"]), float(spec["q_concentration"])
        return g.with_probs(np.clip(rng.beta(a * c, (1 - a) * c, size=g.m), 1e-6, 1 - 1e-6))
    raise ValueError(f"unknown q specification: {spec['q']}")


def sweep(graph: UncertainGraph, z: np.ndarray, alpha: float, config: dict) -> list[dict]:
    """Sweep L for one alpha."""
    p = pagerank(expected_transition(graph), alpha=alpha, z=z).scores
    s = exact_savg(graph, alpha=alpha, z=z).scores
    gap = float(np.abs(p - s).sum())
    ks = [k for k in config["topk"] if k < graph.n]
    rows = []
    for L in config["intervals"]:
        L = int(L)
        r = visit_distribution(graph, L, alpha=alpha, z=z)
        d = r.visit
        row = {
            "alpha": alpha, "L": L, "tau": L * (1 - alpha),
            "dist_to_pavg": float(np.abs(d - p).sum()),
            "dist_to_savg": float(np.abs(d - s).sum()),
            "pavg_savg_gap": gap,
            "rho": 1.0 - float(np.abs(d - s).sum()) / gap if gap > 0 else float("nan"),
            "kendall_vs_pavg": float(stats.kendalltau(d, p).statistic),
            "kendall_vs_savg": float(stats.kendalltau(d, s).statistic),
            "iterations": r.iterations,
        }
        for k in ks:
            row[f"overlap_vs_pavg@{k}"] = topk_overlap(p, d, k)
            row[f"overlap_vs_savg@{k}"] = topk_overlap(s, d, k)
        rows.append(row)
    return rows


def half_point(rows: list[dict]) -> tuple[float, float]:
    """The L and tau at which rho reaches 0.5, by linear interpolation."""
    prev = (1.0, 0.0)
    for r in rows:
        if r["rho"] >= 0.5:
            L0, r0 = prev
            L_half = L0 + (0.5 - r0) * (r["L"] - L0) / (r["rho"] - r0) if r["rho"] > r0 else r["L"]
            return L_half, L_half * (1.0 - r["alpha"])
        prev = (float(r["L"]), r["rho"])
    return float("nan"), float("nan")


def run(config: dict) -> list[dict]:
    rows: list[dict] = []
    for spec in config["graphs"]:
        for seed in config["seeds"]:
            graph = build(spec, np.random.default_rng(int(seed)))
            z = uniform_vector(graph.n)
            halves = {}
            for alpha in config["alpha"]:
                alpha = float(alpha)
                sub = sweep(graph, z, alpha, config)
                L_half, tau_half = half_point(sub)
                halves[alpha] = (L_half, tau_half)
                for r in sub:
                    rows.append({"label": spec["label"], "seed": int(seed),
                                 "n": graph.n, "m": graph.m,
                                 "L_half": L_half, "tau_half": tau_half, **r})
            ls = np.array([v[0] for v in halves.values()])
            ts = np.array([v[1] for v in halves.values()])
            cv_l = float(np.std(ls) / np.mean(ls)) if np.isfinite(ls).all() else float("nan")
            cv_t = float(np.std(ts) / np.mean(ts)) if np.isfinite(ts).all() else float("nan")
            print(f"  [{spec['label']:24s} seed={seed}] "
                  f"L_half={ls.min():.2f}-{ls.max():.2f} (CV={cv_l:.3f})  "
                  f"tau_half={ts.min():.3f}-{ts.max():.3f} (CV={cv_t:.3f})", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    out_dir = args.out or (REPO / config["output_dir"])
    t0 = time.perf_counter()
    rows = run(config)
    csv_path = save_run(out_dir, config, rows, REPO)
    print(f"\n{len(rows)} rows written to {csv_path} "
          f"({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
