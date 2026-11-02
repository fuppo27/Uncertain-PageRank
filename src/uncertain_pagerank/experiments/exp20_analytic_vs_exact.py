"""Experiment 20: how closely analytic decoding and selection match enumeration.

Scaling the acquisition approach means replacing the enumeration of all worlds by
an analytic P(v in TopK), which introduces two approximations:

  1. a normal approximation to the joint law of the scores
  2. ignoring everything outside the candidate set of size L

This experiment compares the two at a size where enumeration is still possible,
to judge **whether the approximation costs any selection quality**. If it does,
the approach does not scale.

What is measured:
  decoding quality   whether the analytic report matches the Bayes-optimal report
                     from enumeration
  selection quality  the regret of the links each score selects, evaluated by the
                     **exact loss from enumeration**. Selection and evaluation
                     never share an approximation, which would be circular.

The selection scores compared:
  dV_edge        the closed-form surrogate, O(sum D^2); the one meant to scale
  analytic_voi   the analytic R(empty) - R({e}); costly, but it shows the quality
                 of the approximation directly
  exact_voi      the true value of information from enumeration; the oracle, and
                 not computable at scale
  baselines      q_variance / entropy / pagerank_src / random

Usage:
    python -m uncertain_pagerank.experiments.exp20_analytic_vs_exact \\
        --config configs/experiment/exp20_analytic.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

from ..acquisition import _strict_topk, enumerate_worlds, expected_loss, true_voi_edge
from ..acquisition_scalable import (
    expected_loss_analytic,
    operational_loss,
    positions_to_edge_ids,
    uncertain_edge_ids,
)
from ..expected_transition import expected_transition
from ..pagerank import pagerank, uniform_vector
from ..variance import contribution_vectors, edge_acquisition_scores
from .exp17_timescale_gate import build
from .runtime import save_run

REPO = Path(__file__).resolve().parents[3]


def run(config: dict) -> list[dict]:
    rows: list[dict] = []
    alpha = float(config["alpha"])
    budgets = [int(b) for b in config["budgets"]]
    draws = int(config["normal_draws"])

    for spec in config["graphs"]:
        for seed in config["seeds"]:
            graph = build(spec, np.random.default_rng(int(seed)))
            z = uniform_vector(graph.n)
            unc = np.flatnonzero((graph.probs > 0.0) & (graph.probs < 1.0))
            if unc.size < 4:
                continue
            for k in config["topk"]:
                k = int(k)
                if k + 1 >= graph.n:
                    continue
                # Always include L=n, with no truncation, so that the effect of
                # candidate truncation separates from the remaining analytic
                # approximation
                l_values = sorted({
                    max(k + 1, min(int(round(float(r) * k)), graph.n))
                    for r in config["L_over_k"]
                } | {graph.n})

                for L in l_values:
                  for tie_seed in config["tie_permutations"]:
                      prio = (None if tie_seed is None
                              else np.random.default_rng(int(tie_seed))
                              .permutation(graph.n).astype(float))
                      ens = enumerate_worlds(graph, k, alpha=alpha, z=z, tie_priority=prio)
                      base_exact = expected_loss(ens, [])
                      voi_exact = true_voi_edge(ens)

                      t0 = time.perf_counter()
                      base_an, dec0 = expected_loss_analytic(
                          graph, k, [], alpha=alpha, z=z, n_candidates=L,
                          n_draws=draws, seed=int(seed), tie_priority=prio,
                      )
                      t_decode = time.perf_counter() - t0
                      cand = dec0[0].candidates

                      # The candidate-set gate: both mean and full coverage
                      in_cand = np.zeros(graph.n, dtype=bool)
                      in_cand[cand] = True
                      hits = ens.topk[:, in_cand].sum(axis=1)
                      mean_cover = float(ens.weights @ (hits / k))
                      full_cover = float(ens.weights @ (hits >= k - 1e-9))

                      # Score the set the analytic decoder actually reported,
                      # on the exact worlds
                      report_ind = np.zeros(graph.n)
                      report_ind[dec0[0].report] = 1.0
                      loss_reported = float(
                          ens.weights @ (1.0 - (ens.topk @ report_ind) / k)
                      )
                      bayes = np.flatnonzero(
                          _strict_topk(ens.marginal[None, :], k, prio)[0][0] > 0
                      )
                      report_match = len(set(bayes.tolist()) & set(dec0[0].report.tolist())) / k

                      et = expected_transition(graph, method="dp")
                      y0 = pagerank(et, alpha=alpha, z=z).scores
                      order = np.argsort(-y0, kind="stable")
                      c = contribution_vectors(
                          et, np.array([order[k - 1], order[k]]), alpha=alpha, z=z
                      )
                      x = c[:, 0] - c[:, 1]
                      t0 = time.perf_counter()
                      dv = edge_acquisition_scores(graph, et, y0, x, alpha=alpha, z=z)[unc]
                      t_dv = time.perf_counter() - t0

                      t0 = time.perf_counter()
                      voi_an = np.array([
                          base_an - expected_loss_analytic(
                              graph, k, positions_to_edge_ids(graph, [i]),
                              alpha=alpha, z=z, n_candidates=L, n_draws=draws,
                              seed=int(seed), tie_priority=prio,
                          )[0]
                          for i in range(unc.size)
                      ])
                      t_voi_an = time.perf_counter() - t0

                      rng = np.random.default_rng(int(seed) + 5)
                      q = graph.probs[unc]
                      src = np.repeat(np.arange(graph.n), graph.out_degrees())[unc]
                      scores = {
                          "dV_edge": dv,
                          "analytic_voi": voi_an,
                          "exact_voi": voi_exact,
                          "q_variance": q * (1 - q),
                          "pagerank_src": y0[src],
                          "random": rng.random(unc.size),
                      }
                      chosen, oracle_losses = [], [base_exact]
                      remaining = list(range(unc.size))
                      for _ in range(max(budgets)):
                          best, bl = None, np.inf
                          for e in remaining:
                              v = expected_loss(ens, edges=chosen + [e])
                              if v < bl:
                                  best, bl = e, v
                          if best is None:
                              break
                          chosen.append(best); remaining.remove(best); oracle_losses.append(bl)

                      common = {
                          "label": spec["label"], "seed": int(seed), "k": k, "L": L,
                          "L_over_k": L / k,
                          "tie_variant": "identity" if tie_seed is None else f"perm{tie_seed}",
                          "n": graph.n, "m": graph.m, "n_uncertain": int(unc.size),
                          "n_certain": int(graph.m - unc.size),
                          "base_loss_exact": base_exact, "base_loss_analytic": base_an,
                          "base_loss_gap": base_an - base_exact,
                          "loss_of_reported_set_exact": loss_reported,
                          "report_match": report_match,
                          "mean_coverage": mean_cover, "full_coverage": full_cover,
                          "clipped_eigenvalue_mass": max(
                              d.clipped_eigenvalue_mass for d in dec0),
                          "decode_seconds": t_decode, "dv_seconds": t_dv,
                          "analytic_voi_seconds": t_voi_an,
                          "spearman_dv_vs_exact": float(
                              stats.spearmanr(dv, voi_exact).statistic),
                          "spearman_analytic_vs_exact": float(
                              stats.spearmanr(voi_an, voi_exact).statistic),
                          # The analytic value of information is a difference
                          # of approximate distributions, so non-negativity is
                          # not guaranteed. Record how many are negative and the
                          # minimum, to see whether it is the size of the
                          # approximation error.
                          "analytic_voi_negative_count": int((voi_an < -1e-12).sum()),
                          "analytic_voi_min": float(voi_an.min()),
                          "exact_voi_min": float(voi_exact.min()),
                      }
                      for name, sc in scores.items():
                          ranking = np.argsort(-sc, kind="stable")
                          for b in budgets:
                            pick = ranking[:b]
                            # (1) selection quality: the loss after re-decoding
                            #     with the exact Bayes decoder
                            loss = expected_loss(ens, edges=pick)
                            # (2) performance in use: the exact loss of the set
                            #     the analytic decoder reported. Case (1) cannot
                            #     see a good selection with a wrong report.
                            ope = operational_loss(
                                ens, graph, k, positions_to_edge_ids(graph, pick),
                                alpha=alpha, z=z, n_candidates=L, n_draws=draws,
                                seed=int(seed), tie_priority=prio,
                            )
                            ref = oracle_losses[min(b, len(oracle_losses) - 1)]
                            rows.append({**common, "method": name, "budget": b,
                                         "loss_exact": loss, "regret": loss - ref,
                                         "loss_operational": ope,
                                         "operational_gap": ope - loss,
                                         "oracle_loss": ref})
            print(f"  [{spec['label']:20s} seed={seed}] -> {len(rows)} rows", flush=True)
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
