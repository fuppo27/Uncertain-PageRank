"""Experiment 25: source-level attribution on real networks.

Experiments 22 and 23 compared against exhaustive enumeration or a Monte Carlo
reference on synthetic graphs. This one does two things on real networks:

  A. the cost of computing `Phi_hat` for every source, and how concentrated the
     shares are (all networks)
  B. where the size allows it, comparison against a Monte Carlo reference over
     40 stratified candidates

Note that `Psi_u` is about **observing** `X_u`, which leaves the network
unchanged. That is a different problem from repairing a source by setting
`q -> 1`, which changes the network and the correct ordering at the same time.

Part B costs `N * (1 + candidates)` PageRank solves, which is out of reach at the
largest sizes. For those networks only Part A is reported, and the fact that the
result has **not** been checked against a reference is stated.

Usage:
    python -m uncertain_pagerank.experiments.exp25_real_networks \\
        --config configs/experiment/exp25_real_networks.yaml
"""

from __future__ import annotations

import argparse
import resource
import time
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

from ..attribution import (
    ContrastSet,
    analytic_attribution,
    boundary_contrasts,
    node_contrasts,
    pickfreeze_attribution,
    pickfreeze_groups,
    resolvent_columns,
)
from ..datasets import load_dlbl, load_edgelist
from ..expected_transition import expected_transition
from ..pagerank import pagerank, uniform_vector
from ..variance import variance_attribution

REPO = Path(__file__).resolve().parents[3]


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def stratified_candidates(y0, sources, n_take, n_strata, rng) -> np.ndarray:
    """Draw candidates stratified by quantile of y0, never by the phi_hat score."""
    n_take = min(n_take, sources.size)
    order = sources[np.argsort(-y0[sources], kind="stable")]
    strata = np.array_split(order, min(n_strata, n_take))
    per = max(1, n_take // len(strata))
    picked: list[int] = []
    for st in strata:
        picked.extend(rng.choice(st, size=min(per, st.size), replace=False).tolist())
    rest = np.setdiff1d(order, np.array(picked, dtype=np.int64))
    if len(picked) < n_take and rest.size:
        picked.extend(rng.choice(rest, size=min(n_take - len(picked), rest.size),
                                 replace=False).tolist())
    return np.sort(np.array(picked[:n_take], dtype=np.int64))


def cheap_scores(graph, y0) -> dict[str, np.ndarray]:
    """The cheap baselines, for every source."""
    n = graph.n
    src = np.repeat(np.arange(n), graph.out_degrees())
    unc = (graph.probs > 0.0) & (graph.probs < 1.0)
    qv = np.bincount(src[unc], weights=(graph.probs * (1 - graph.probs))[unc], minlength=n)
    q = np.clip(graph.probs[unc], 1e-12, 1 - 1e-12)
    h = -(q * np.log2(q) + (1 - q) * np.log2(1 - q))
    ent = np.bincount(src[unc], weights=h, minlength=n)
    cnt = np.bincount(src[unc], minlength=n).astype(float)
    return {
        "pagerank_src": y0, "q_variance": qv, "entropy": ent,
        "n_uncertain": cnt, "out_degree": graph.out_degrees().astype(float),
        "y0sq_x_qvar": y0**2 * qv,
    }


def run_network(spec: dict, config: dict) -> list[dict]:
    path = REPO / spec["path"]
    loader = load_dlbl if path.suffix == ".dlbl" else load_edgelist
    t0 = time.perf_counter()
    graph0 = loader(path).graph
    t_load = time.perf_counter() - t0

    alpha, tol = float(config["alpha"]), float(config["tol"])
    rng = np.random.default_rng(int(config["seed"]))
    a, c = float(config["q_mean"]), float(config["q_concentration"])
    graph = graph0.with_probs(rng.beta(a * c, (1.0 - a) * c, size=graph0.m))
    z = uniform_vector(graph.n)

    t0 = time.perf_counter()
    et = expected_transition(graph, method="dp")
    t_et = time.perf_counter() - t0
    t0 = time.perf_counter()
    y0 = pagerank(et, alpha=alpha, z=z, tol=tol).scores
    t_pr = time.perf_counter() - t0

    src = np.repeat(np.arange(graph.n), graph.out_degrees())
    unc_src = np.unique(src[(graph.probs > 0.0) & (graph.probs < 1.0)])
    base_scores = cheap_scores(graph, y0)
    rows: list[dict] = []

    for cname in config["contrasts"]:
        ks = [int(k) for k in config["boundary_ks"]]
        if cname == "boundary":
            cs = boundary_contrasts(y0, ks)
        elif cname == "node_top":
            cs = node_contrasts(np.argsort(-y0)[: len(ks)], graph.n)
        else:
            raise ValueError(cname)

        t0 = time.perf_counter()
        phi = analytic_attribution(graph, et, y0, cs, alpha=alpha, z=z)
        t_phi = time.perf_counter() - t0

        # Ablations: which of the direction x and y0^2 is doing the work,
        # measured on real data too
        arng = np.random.default_rng(int(config["seed"]) + 123)
        wr = arng.normal(size=cs.W.shape)
        wr /= np.linalg.norm(wr, axis=1, keepdims=True)
        phi_rand = analytic_attribution(
            graph, et, y0, ContrastSet(W=wr, lam=cs.lam, labels=["rand"] * cs.r),
            alpha=alpha, z=z)
        xk = resolvent_columns(et, cs.W.T, alpha=alpha, z=z)
        phi_nov = np.zeros(graph.n)
        for kk in range(cs.r):
            phi_nov += cs.lam[kk] * variance_attribution(
                graph, et, np.ones(graph.n), xk[:, kk], alpha=alpha, z=z).per_source

        share = phi / phi.sum() if phi.sum() > 0 else phi
        srt = np.sort(share)[::-1]
        act = phi > 0
        common = {
            "network": spec["name"], "n": graph.n, "m": graph.m,
            "n_uncertain": int(((graph.probs > 0) & (graph.probs < 1)).sum()),
            "n_sources": int(unc_src.size), "contrast": cname, "r_contrasts": cs.r,
            "load_seconds": t_load, "expected_transition_seconds": t_et,
            "pagerank_seconds": t_pr, "attribution_seconds": t_phi,
            "peak_rss_mb": peak_rss_mb(),
            "total_variance_surrogate": float(phi.sum()),
            "top1_share": float(srt[0]), "top10_share": float(srt[:10].sum()),
            "top100_share": float(srt[:100].sum()),
            "n_sources_for_90pct": int(np.searchsorted(np.cumsum(srt), 0.90) + 1),
            "corr_phi_vs_pagerank": float(
                stats.spearmanr(y0[act], phi[act]).statistic) if act.sum() > 2 else np.nan,
            "corr_phi_vs_outdeg": float(
                stats.spearmanr(graph.out_degrees()[act], phi[act]).statistic
            ) if act.sum() > 2 else np.nan,
            "mc_validated": False,
        }
        rows.append({**common, "method": "phi_hat", "spearman_vs_reference": np.nan,
                     "seconds": t_phi, "solves": 0})

        if not spec.get("mc_validation", False):
            continue

        # ---- B: comparison against the Monte Carlo reference, where feasible
        n_ref = int(config["reference_samples"])
        mc_tol = float(config.get("mc_tol", 1e-8))
        chunk = int(config.get("mc_chunk", 256))
        half = max(1, n_ref // 2)
        mode = config.get("validation_mode", "blocks")

        if mode == "blocks":
            # Per-source pick-freeze does not work at real-network scale.
            # psi_u is of order Var(Z) / (number of sources) while the standard
            # error of the estimator is of order Var(Z) / sqrt(N), so balancing
            # them needs N ~ (number of sources)^2, which is 10^7 to 10^9
            # samples here. Grouping by quantile of y0 makes psi_G of order
            # Var(Z) / (number of groups), for which N >> (groups)^2 suffices.
            nb = int(config.get("n_blocks", 12))
            order = unc_src[np.argsort(-y0[unc_src], kind="stable")]
            blocks = [b for b in np.array_split(order, nb) if b.size]
            cand = np.concatenate(blocks)
            ref_a = pickfreeze_groups(graph, cs, blocks, n_samples=half, seed=1,
                                      alpha=alpha, z=z, tol=mc_tol, chunk=chunk,
                                      n_boot=400)
            ref_b = pickfreeze_groups(graph, cs, blocks, n_samples=half, seed=2,
                                      alpha=alpha, z=z, tol=mc_tol, chunk=chunk)
            # The analytic side sums within a group: the first-order surrogate
            # is additive over sources
            def agg(v):
                return np.array([v[b].sum() for b in blocks])
        else:
            cand = stratified_candidates(
                y0, unc_src, int(config["n_candidate_sources"]),
                int(config.get("n_strata", 8)),
                np.random.default_rng(int(config["seed"]) + 42))
            blocks = None
            ref_a = pickfreeze_attribution(graph, cs, sources=cand, n_samples=half,
                                           seed=1, alpha=alpha, z=z, tol=mc_tol,
                                           chunk=chunk, n_boot=400)
            ref_b = pickfreeze_attribution(graph, cs, sources=cand, n_samples=half,
                                           seed=2, alpha=alpha, z=z, tol=mc_tol,
                                           chunk=chunk)

            def agg(v):
                return v[cand]
        ref = (ref_a.psi + ref_b.psi) / 2.0
        reliability = float(stats.spearmanr(ref_a.psi, ref_b.psi).statistic)
        scale = float(np.abs(ref).max())
        common.update({
            "mc_validated": True, "validation_mode": mode,
            "n_blocks": len(blocks) if blocks is not None else 0,
            "n_candidates": int(cand.size),
            "reference_samples": n_ref,
            "reference_seconds": ref_a.seconds + ref_b.seconds,
            "reference_solves": ref_a.n_solves + ref_b.n_solves,
            "reference_split_half": reliability,
            "reference_ci_width_rel": float(
                np.nanmean(ref_a.psi_hi - ref_a.psi_lo)) / scale if scale > 0 else np.nan,
        })
        phi_agg = agg(phi)
        phi_rho = float(stats.spearmanr(ref, phi_agg).statistic)
        # With groups the values themselves are comparable, not just the ranks
        common["phi_sum_over_ref_sum"] = float(phi_agg.sum() / ref.sum()) if ref.sum() else np.nan
        rows[-1].update(common)
        rows[-1]["spearman_vs_reference"] = phi_rho

        entries = [("phi_randdir", agg(phi_rand), 0.0, 0),
                   ("phi_no_y0sq", agg(phi_nov), 0.0, 0)]
        for name, sc in base_scores.items():
            entries.append((name, agg(sc), 0.0, 0))
        for b in config.get("mc_budgets", []):
            if blocks is not None:
                mc = pickfreeze_groups(graph, cs, blocks, n_samples=int(b), seed=777,
                                       alpha=alpha, z=z, tol=mc_tol, chunk=chunk)
            else:
                mc = pickfreeze_attribution(graph, cs, sources=cand, n_samples=int(b),
                                            seed=777, alpha=alpha, z=z, tol=mc_tol,
                                            chunk=chunk)
            entries.append((f"mc_{b}", mc.psi, mc.seconds, mc.n_solves))
        for name, sc, secs, nsolve in entries:
            rows.append({**common, "method": name,
                         "spearman_vs_reference": float(stats.spearmanr(ref, sc).statistic),
                         "seconds": secs, "solves": nsolve})
        print(f"    {cname}: reference split-half={reliability:.3f}  "
              f"phi_hat={phi_rho:.3f}", flush=True)
    return rows


def run(config: dict) -> list[dict]:
    rows: list[dict] = []
    for spec in config["networks"]:
        print(f"  [{spec['name']}] start", flush=True)
        rows.extend(run_network(spec, config))
        print(f"  [{spec['name']}] done -> {len(rows)} rows", flush=True)
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
