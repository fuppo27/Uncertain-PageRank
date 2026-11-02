"""Experiment 23: does the analytic Phi_hat hold up where enumeration cannot run?

At small scale (experiment 22) the true Psi_u came from exhaustive enumeration,
but 2^{m_u} puts that out of reach at medium scale. Here a Monte Carlo estimator
serves as the **reference**, and we measure:

  - how closely Phi_hat agrees with it, solving one resolvent per contrast
  - how many samples and seconds Monte Carlo needs to reach that agreement
  - whether the gap to the cheap baselines survives at this scale

The reference carries sampling error of its own, so:
  (a) the reference and the budgeted competitor use **independent draws**;
      correlated errors would inflate the agreement
  (b) the bootstrap width of the reference is reported alongside
  (c) agreement is measured by rank correlation, since agreement in value is
      buried in the reference's own error

Candidate sources are fixed in advance by **random draw from a fixed seed**, so
that no method is favoured. They are never chosen by score.

Usage:
    python -m uncertain_pagerank.experiments.exp23_medium_scale \\
        --config configs/experiment/exp23_medium_scale.yaml
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
    identity_contrast,
    node_contrasts,
    pickfreeze_attribution,
    pickfreeze_attribution_owen,
    trace_row_covariance,
)
from ..expected_transition import expected_transition
from ..pagerank import pagerank, uniform_vector
from .exp17_timescale_gate import build

REPO = Path(__file__).resolve().parents[3]


def uncertain_sources(graph) -> np.ndarray:
    """The source nodes with an uncertain out-link."""
    src = np.repeat(np.arange(graph.n), graph.out_degrees())
    return np.unique(src[(graph.probs > 0.0) & (graph.probs < 1.0)])


def cheap_scores(graph, y0, cand, et=None, z=None) -> dict[str, np.ndarray]:
    """The cheap baselines, valued on the candidate sources.

    `y0sq_x_trcov` is phi_hat with **only the direction x** removed.
    `y0sq_x_qvar` also discards the row normalisation and the within-row
    covariance, so the former is the better control for what the direction
    contributes.
    """
    src = np.repeat(np.arange(graph.n), graph.out_degrees())
    unc = (graph.probs > 0.0) & (graph.probs < 1.0)
    qv = np.zeros(graph.n)
    ent = np.zeros(graph.n)
    cnt = np.zeros(graph.n)
    for u, qq in zip(src[unc], graph.probs[unc]):
        qv[u] += qq * (1.0 - qq)
        ent[u] += -(qq * np.log2(qq) + (1 - qq) * np.log2(1 - qq))
        cnt[u] += 1.0
    return {
        "pagerank_src": y0[cand],
        "q_variance": qv[cand],
        "entropy": ent[cand],
        "n_uncertain": cnt[cand],
        "out_degree": graph.out_degrees()[cand].astype(float),
        "y0sq_x_qvar": (y0**2 * qv)[cand],
        **({"y0sq_x_trcov": (y0**2 * trace_row_covariance(graph, et, z))[cand]}
           if et is not None else {}),
    }


def run(config: dict) -> list[dict]:
    rows: list[dict] = []
    alpha = float(config["alpha"])
    ks = [int(k) for k in config["boundary_ks"]]
    n_ref = int(config["reference_samples"])
    budgets = [int(b) for b in config["mc_budgets"]]
    n_cand = int(config["n_candidate_sources"])

    for spec in config["graphs"]:
        for seed in config["seeds"]:
            graph = build(spec, np.random.default_rng(int(seed)))
            z = uniform_vector(graph.n)
            et = expected_transition(graph, method="dp")
            y0 = pagerank(et, alpha=alpha, z=z).scores
            src_all = uncertain_sources(graph)
            if src_all.size < 5:
                continue
            # How candidates are chosen. Neither route uses the phi_hat score.
            pick_rng = np.random.default_rng(int(seed) + 4242)
            mode = config.get("candidate_selection", "random")
            if mode == "random":
                cand = np.sort(pick_rng.choice(
                    src_all, size=min(n_cand, src_all.size), replace=False))
            elif mode == "stratified_y0":
                # Plain random sampling is neutral between methods, but on a
                # heavy-tailed graph where PageRank concentrates on hubs every
                # candidate turns out to be a source with Psi_u near zero, and
                # the reference cannot estimate a near-zero quantity to good
                # relative accuracy. Stratifying by quantile of y0 fixes that.
                # Stratification raises the diversity of y0 among the
                # candidates, which favours pagerank_src and is conservative
                # for phi_hat.
                n_take = min(n_cand, src_all.size)
                order = src_all[np.argsort(-y0[src_all], kind="stable")]
                strata = np.array_split(order, min(int(config.get("n_strata", 8)), n_take))
                per = max(1, n_take // len(strata))
                picked: list[int] = []
                for st in strata:
                    take = min(per, st.size)
                    picked.extend(pick_rng.choice(st, size=take, replace=False).tolist())
                rest = np.setdiff1d(order, np.array(picked, dtype=np.int64))
                if len(picked) < n_take and rest.size:
                    picked.extend(pick_rng.choice(
                        rest, size=min(n_take - len(picked), rest.size),
                        replace=False).tolist())
                cand = np.sort(np.array(picked[:n_take], dtype=np.int64))
            else:
                raise ValueError(f"unknown candidate_selection: {mode}")

            for cname in config["contrasts"]:
                if cname == "boundary":
                    cs = boundary_contrasts(y0, ks)
                elif cname == "node_top":
                    cs = node_contrasts(np.argsort(-y0)[: len(ks)], graph.n)
                elif cname == "identity":
                    cs = identity_contrast(graph.n)
                else:
                    raise ValueError(cname)

                # The reference is split into two independent halves: their
                # mean is the reference, and their mutual correlation is
                # reported as the **noise ceiling**. The reference is itself
                # noisy, so without this no method's correlation can be read.
                #
                # The estimator defaults to Owen's Correlation 2. The naive
                # pick-freeze design has a large variance where
                # Psi_u << mu^2, which leaves the reference pure noise here:
                # at equal cost its split-half agreement is 0.002 against
                # 0.988.
                est = (pickfreeze_attribution_owen
                       if config.get("reference_estimator", "owen") == "owen"
                       else pickfreeze_attribution)
                half = max(1, n_ref // 2)
                ref_a = est(graph, cs, sources=cand, n_samples=half,
                            seed=int(seed) * 1000 + 1, alpha=alpha, z=z, n_boot=400)
                ref_b = est(graph, cs, sources=cand, n_samples=half,
                            seed=int(seed) * 1000 + 2, alpha=alpha, z=z)
                ref_psi = (ref_a.psi + ref_b.psi) / 2.0
                ref_reliability = float(spearmanr(ref_a.psi, ref_b.psi).statistic)
                ref_seconds = ref_a.seconds + ref_b.seconds
                ref_solves = ref_a.n_solves + ref_b.n_solves
                ref_width = float(np.nanmean(ref_a.psi_hi - ref_a.psi_lo))
                scale = float(np.abs(ref_psi).max())

                t0 = time.perf_counter()
                phi = analytic_attribution(graph, et, y0, cs, alpha=alpha, z=z)
                t_phi = time.perf_counter() - t0

                common = {
                    "label": spec["label"], "seed": int(seed), "contrast": cname,
                    "n": graph.n, "m": graph.m,
                    "n_uncertain": int(((graph.probs > 0) & (graph.probs < 1)).sum()),
                    "n_sources_total": int(src_all.size),
                    "n_candidates": int(cand.size), "r_contrasts": cs.r,
                    "candidate_selection": mode,
                    "reference_samples": n_ref,
                    "reference_estimator": config.get("reference_estimator", "owen"),
                    "reference_solves": ref_solves,
                    "reference_seconds": ref_seconds,
                    "reference_ci_width_rel": ref_width / scale if scale > 0 else np.nan,
                    # The correlation of two independent halves. No method can
                    # exceed it by much.
                    "reference_split_half": ref_reliability,
                    "var_z": ref_a.var_z,
                }

                entries = [("phi_hat", phi[cand], t_phi, 0, 0)]
                for name, sc in cheap_scores(graph, y0, cand, et, z).items():
                    entries.append((name, sc, 0.0, 0, 0))
                # Run the budgeted Monte Carlo with draws independent of the
                # reference
                for b in budgets:
                    mc = est(graph, cs, sources=cand, n_samples=b,
                             seed=int(seed) * 1000 + 777, alpha=alpha, z=z)
                    entries.append((f"mc_{b}", mc.psi, mc.seconds, b, mc.n_solves))

                order_ref = np.argsort(-ref_psi)
                for name, sc, secs, nb, nsolve in entries:
                    sc = np.asarray(sc, dtype=np.float64)
                    order = np.argsort(-sc)
                    # Error in the effect sizes. The relative error diverges on
                    # a source whose true value is near zero, so the error is
                    # normalised by the total variance; both numerator and
                    # denominator are estimates. The heuristics are in different
                    # units and are not comparable, so this covers phi_hat and
                    # the Monte Carlo competitors only.
                    same_unit = name == "phi_hat" or name.startswith("mc_")
                    vz = float(ref_a.var_z)
                    rows.append({
                        **common, "method": name,
                        "spearman": float(spearmanr(ref_psi, sc).statistic),
                        "top1_hit": bool(order[0] == order_ref[0]),
                        "top5_overlap": float(
                            len(set(order[:5].tolist()) & set(order_ref[:5].tolist())) / 5.0
                        ),
                        # Where this method ranks the reference's top source
                        "ref_top1_rank": int(np.where(order == order_ref[0])[0][0]) + 1,
                        "abs_err_over_varz": (float(np.abs(sc - ref_psi).mean() / vz)
                                              if same_unit and vz > 0 else np.nan),
                        "max_err_over_varz": (float(np.abs(sc - ref_psi).max() / vz)
                                              if same_unit and vz > 0 else np.nan),
                        "seconds": secs, "mc_samples": nb, "solves": nsolve,
                    })
            print(f"  [{spec['label']:18s} seed={seed}] n={graph.n} "
                  f"sources={src_all.size} candidates={cand.size} "
                  f"-> {len(rows)} rows", flush=True)
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
