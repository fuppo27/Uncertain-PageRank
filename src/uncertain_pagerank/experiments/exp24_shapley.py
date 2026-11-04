"""Experiment 24: Shapley effects against the first-order attribution.

The first-order `Psi_u`, the numerator of the group Sobol index, does not sum to
`tr Cov(Z)`. The gap `I_inter` is the interaction between sources, and a fully
sum-preserving allocation requires
variance-based Shapley effect

    Sh_u = sum_{S subset D\\{u}} |S|!(d-|S|-1)!/d! ( c(S+u) - c(S) ),
    c(S) = tr Cov( E[Z | X_S] )

which needs all 2^d subsets, so this is a **side evaluation** limited to small
scale. It asks:

  1. how much reallocating the interaction changes the ordering (Psi vs Shapley)
  2. whether the analytic `Phi_hat` predicts Shapley too, or only Psi
  3. whether the two diverge more as the share of `I_inter` grows

Usage:
    python -m uncertain_pagerank.experiments.exp24_shapley \\
        --config configs/experiment/exp24_shapley.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import spearmanr

from ..attribution import (
    analytic_attribution,
    boundary_contrasts,
    enumerate_world_scores,
    exact_attribution,
    identity_contrast,
    interaction_term,
    node_contrasts,
    shapley_effects,
    total_variance,
    true_contrast_values,
)
from ..expected_transition import expected_transition
from ..pagerank import pagerank, uniform_vector
from .exp17_timescale_gate import build

REPO = Path(__file__).resolve().parents[3]

# Relative tolerance below which two effects count as tied. On a generator with
# uniform `q` the sources are exchangeable, so their effects are *exactly* equal
# in exact arithmetic and differ by 1e-18 to 1e-15 once computed in floating
# point. Six of the 75 conditions sit on such a tie, and the closest genuine gap
# anywhere is 9e-5, four orders of magnitude up, so the threshold is not a
# judgement call. Without it `spearmanr` ranks a tie by rounding noise, and the
# reported correlation changes with the summation order.
TIE_RTOL = 1e-9


def tied_ranks(v: np.ndarray, rtol: float = TIE_RTOL) -> np.ndarray:
    """`v` with values that are tied to within `rtol` of its largest magnitude
    collapsed onto one representative, so `spearmanr` assigns them equal ranks.
    """
    v = np.asarray(v, dtype=float)
    scale = max(float(np.abs(v).max()), np.finfo(float).tiny)
    order = np.argsort(v, kind="stable")
    out = v.copy()
    rep = v[order[0]]
    for j in order[1:]:
        if v[j] - rep <= rtol * scale:
            out[j] = rep
        else:
            rep = v[j]
    return out


def tied_spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation that treats near-ties in either vector as ties."""
    return float(spearmanr(tied_ranks(a), tied_ranks(b)).statistic)


def run(config: dict) -> list[dict]:
    rows: list[dict] = []
    alpha = float(config["alpha"])
    ks = [int(k) for k in config["boundary_ks"]]
    max_sources = int(config["max_sources"])

    for spec in config["graphs"]:
        for seed in config["seeds"]:
            graph = build(spec, np.random.default_rng(int(seed)))
            z = uniform_vector(graph.n)
            et = expected_transition(graph, method="dp")
            y0 = pagerank(et, alpha=alpha, z=z).scores
            ws = enumerate_world_scores(graph, alpha=alpha, z=z)
            active = np.unique(ws.edge_source)
            if active.size > max_sources or active.size < 4:
                print(f"  [{spec['label']}, seed={seed}] {active.size} sources: out of range",
                      flush=True)
                continue

            for cname in config["contrasts"]:
                if cname == "boundary":
                    cs = boundary_contrasts(y0, ks)
                elif cname == "node_top":
                    cs = node_contrasts(np.argsort(-y0)[: len(ks)], graph.n)
                elif cname == "identity":
                    cs = identity_contrast(graph.n)
                else:
                    raise ValueError(cname)

                zt = true_contrast_values(ws.scores, cs)
                var = total_variance(ws.weights, zt)
                if var <= 0:
                    continue
                psi = exact_attribution(ws.bits, ws.weights, zt, ws.edge_source, graph.n)
                phi = analytic_attribution(graph, et, y0, cs, alpha=alpha, z=z)
                t0 = time.perf_counter()
                sh = shapley_effects(ws.bits, ws.weights, zt, ws.edge_source, active)
                t_sh = time.perf_counter() - t0
                inter = interaction_term(ws.weights, zt, psi)

                p, s_, f = psi[active], sh, phi[active]
                rows.append({
                    "label": spec["label"], "seed": int(seed), "contrast": cname,
                    "n": graph.n, "n_uncertain": int(ws.bits.shape[1]),
                    "n_worlds": int(ws.bits.shape[0]), "d_sources": int(active.size),
                    "total_variance": var,
                    "interaction_share": inter / var,
                    "shapley_sum_rel_err": abs(s_.sum() - var) / var,
                    "psi_sum_share": float(p.sum() / var),
                    # how much the ordering moves when the interaction is
                    # reallocated
                    "spearman_psi_shapley": tied_spearman(p, s_),
                    "top1_same_psi_shapley": bool(np.argmax(p) == np.argmax(s_)),
                    # which of the two the analytic route predicts
                    "spearman_phi_psi": tied_spearman(f, p),
                    "spearman_phi_shapley": tied_spearman(f, s_),
                    # how much value is reallocated
                    "max_abs_reallocation_share": float(np.abs(s_ - p).max() / var),
                    "shapley_seconds": t_sh,
                })
            print(f"  [{spec['label']:16s} seed={seed}] d={active.size} "
                  f"worlds={ws.bits.shape[0]} -> {len(rows)} rows", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    out_dir = args.out or (REPO / config["output_dir"])
    from .runtime import save_run

    t0 = time.perf_counter()
    rows = run(config)
    csv_path = save_run(out_dir, config, rows, REPO)
    print(f"\n{len(rows)} rows written to {csv_path} "
          f"({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
