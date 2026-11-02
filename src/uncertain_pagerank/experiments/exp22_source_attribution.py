"""Experiment 22: does the attribution predict the true source effect?

With a contrast matrix C fixed in advance, the ground truth is

    Psi_u(C) = tr Cov( E[Z | X_u] ),   Z = C y_X

the true expected reduction in variance from observing source u, computed
exactly by enumerating every world.

What is judged:

  1. whether the first-order surrogate Phi_hat_u predicts Psi_u better than
     PageRank, degree, entropy and random, by rank correlation and top-set
     agreement
  2. how close the source set chosen by Phi_hat comes to the global oracle in
     true variance reduction, with the cost normalised both by source count and
     by the number of links revealed
  3. how large the interaction I_inter = Var(Z) - sum_u Psi_u is

**The decisive ablation**: Phi_hat is alpha^2 y0(u)^2 (x^T Cov_u x), so unless it
beats a baseline built from the product of y0(u)^2 and sum_e q(1-q) alone, the
directional information from the resolvent is not doing any work. It is always
included, because the same y0(u)^2 confound caught an earlier approach.

The contrast is fixed from the p-average before any world is examined, and
boundaries that worked are never selected afterwards.

Usage:
    python -m uncertain_pagerank.experiments.exp22_source_attribution \\
        --config configs/experiment/exp22_source_attribution.yaml
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import spearmanr

from ..attribution import (
    ContrastSet,
    analytic_attribution,
    boundary_contrasts,
    enumerate_world_scores,
    exact_attribution,
    exact_set_attribution,
    gap_stratified_contrasts,
    identity_contrast,
    interaction_term,
    node_contrasts,
    resolvent_columns,
    source_positions,
    total_variance,
    trace_row_covariance,
    true_contrast_values,
)
from ..expected_transition import expected_transition
from ..pagerank import pagerank, uniform_vector
from ..variance import variance_attribution
from .exp17_timescale_gate import build

REPO = Path(__file__).resolve().parents[3]


def make_contrasts(name: str, y0: np.ndarray, ks, rng) -> "tuple":
    """Build the contrast set fixed in advance. No world is examined."""
    if name == "boundary":
        return boundary_contrasts(y0, ks)
    if name == "node_top":
        return node_contrasts(np.argsort(-y0)[: len(ks)], y0.size)
    if name == "identity":
        return identity_contrast(y0.size)
    if name == "gap_stratified":
        return gap_stratified_contrasts(y0, per_stratum=2, rng=rng)
    raise ValueError(f"unknown contrast: {name}")


def selection_scores(graph, y0, ws, rng) -> dict[str, np.ndarray]:
    """Scores for choosing sources. No world is examined."""
    q = graph.probs[ws.uncertain_edges]
    qv = np.zeros(graph.n)
    ent = np.zeros(graph.n)
    cnt = np.zeros(graph.n)
    for u, qq in zip(ws.edge_source, q):
        qv[u] += qq * (1.0 - qq)
        ent[u] += -(qq * np.log2(qq) + (1 - qq) * np.log2(1 - qq))
        cnt[u] += 1.0
    return {
        "pagerank_src": y0,
        "out_degree": graph.out_degrees().astype(float),
        "n_uncertain": cnt,
        "entropy": ent,
        "q_variance": qv,
        # The decisive ablation: the product of Phi_hat's two factors alone
        "y0sq_x_qvar": y0**2 * qv,
        "y0sq_x_entropy": y0**2 * ent,
        "random": rng.random(graph.n),
    }


def global_oracle_sources(ws, values, active, budget, max_edges) -> tuple[tuple, float]:
    """The source set of largest true variance reduction, over all subsets."""
    best, best_val = (), -np.inf
    for combo in itertools.combinations(active.tolist(), budget):
        pos = source_positions(ws.edge_source, combo)
        if pos.size > max_edges:
            continue
        val = exact_set_attribution(ws.bits, ws.weights, values, pos)
        if val > best_val:
            best, best_val = combo, val
    return best, best_val


def run(config: dict) -> list[dict]:
    rows: list[dict] = []
    alpha = float(config["alpha"])
    ks = [int(k) for k in config["boundary_ks"]]
    budgets = [int(b) for b in config["budgets"]]
    max_edges = int(config.get("max_observed_edges", 20))

    for spec in config["graphs"]:
        for seed in config["seeds"]:
            graph = build(spec, np.random.default_rng(int(seed)))
            z = uniform_vector(graph.n)
            et = expected_transition(graph, method="dp")
            y0 = pagerank(et, alpha=alpha, z=z).scores

            t0 = time.perf_counter()
            ws = enumerate_world_scores(graph, alpha=alpha, z=z)
            t_enum = time.perf_counter() - t0
            active = np.unique(ws.edge_source)
            if active.size < max(budgets) + 1:
                continue
            scores = selection_scores(graph, y0, ws, np.random.default_rng(int(seed) + 7))

            for cname in config["contrasts"]:
                cs = make_contrasts(cname, y0, ks, np.random.default_rng(int(seed) + 11))
                zt = true_contrast_values(ws.scores, cs)
                var = total_variance(ws.weights, zt)
                if var <= 0:
                    continue
                psi = exact_attribution(ws.bits, ws.weights, zt, ws.edge_source, graph.n)
                t1 = time.perf_counter()
                phi = analytic_attribution(graph, et, y0, cs, alpha=alpha, z=z)
                t_phi = time.perf_counter() - t1
                inter = interaction_term(ws.weights, zt, psi)

                # Contrast-specific ablations; each drops one component of
                # phi_hat.
                #   phi_randdir : replaces the direction x by a random contrast
                #   phi_no_y0sq : drops the y0(u)^2 weight, keeping direction
                rng_a = np.random.default_rng(int(seed) + 123)
                wr = rng_a.normal(size=cs.W.shape)
                wr /= np.linalg.norm(wr, axis=1, keepdims=True)
                phi_rand = analytic_attribution(
                    graph, et, y0, ContrastSet(W=wr, lam=cs.lam, labels=["rand"] * cs.r),
                    alpha=alpha, z=z,
                )
                xk = resolvent_columns(et, cs.W.T, alpha=alpha, z=z)
                phi_nov = np.zeros(graph.n)
                for kk in range(cs.r):
                    phi_nov += cs.lam[kk] * variance_attribution(
                        graph, et, np.ones(graph.n), xk[:, kk], alpha=alpha, z=z
                    ).per_source

                # The oracle does not depend on the method, so compute it once
                # outside the method loop
                oracles = {
                    b: global_oracle_sources(ws, zt, active, b, max_edges)
                    for b in budgets
                }
                # A control with only the direction removed, keeping the row
                # normalisation and the within-row covariance
                phi_trc = y0**2 * trace_row_covariance(graph, et, z)
                allsc = {"phi_hat": phi, "phi_randdir": phi_rand,
                         "phi_no_y0sq": phi_nov, "y0sq_x_trcov": phi_trc, **scores}
                nz = psi[active] > 1e-14
                common = {
                    "label": spec["label"], "seed": int(seed), "contrast": cname,
                    "n": graph.n, "m": graph.m, "n_uncertain": int(ws.bits.shape[1]),
                    "n_worlds": int(ws.bits.shape[0]), "n_active_sources": int(active.size),
                    "r_contrasts": cs.r,
                    "total_variance": var,
                    "interaction": inter, "interaction_share": inter / var,
                    "enumerate_seconds": t_enum, "analytic_seconds": t_phi,
                    # The error in magnitude from linearizing, not in rank
                    "lin_rel_err_median": float(np.median(
                        np.abs(phi[active][nz] - psi[active][nz]) / psi[active][nz]
                    )) if nz.any() else np.nan,
                }

                for mname, sc in allsc.items():
                    rho = spearmanr(psi[active], sc[active]).statistic
                    order = active[np.argsort(-sc[active], kind="stable")]
                    true_order = active[np.argsort(-psi[active], kind="stable")]
                    row = {
                        **common, "method": mname,
                        "spearman": float(rho),
                        "top1_hit": bool(order[0] == true_order[0]),
                        "top3_overlap": float(
                            len(set(order[:3].tolist()) & set(true_order[:3].tolist())) / 3.0
                        ),
                    }
                    for budget in budgets:
                        pick = tuple(order[:budget].tolist())
                        pos = source_positions(ws.edge_source, pick)
                        got = (exact_set_attribution(ws.bits, ws.weights, zt, pos)
                               if pos.size <= max_edges else np.nan)
                        oset, oval = oracles[budget]
                        row[f"psi_set_b{budget}"] = got
                        row[f"oracle_b{budget}"] = oval
                        row[f"regret_b{budget}"] = oval - got
                        # Keep the number of links revealed, to normalise cost
                        row[f"revealed_edges_b{budget}"] = int(pos.size)
                        row[f"oracle_revealed_b{budget}"] = int(
                            source_positions(ws.edge_source, oset).size
                        )
                    rows.append(row)
            print(f"  [{spec['label']:30s} seed={seed}] "
                  f"worlds={ws.bits.shape[0]:7d} "
                  f"source={active.size:3d} -> {len(rows)} rows", flush=True)
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
