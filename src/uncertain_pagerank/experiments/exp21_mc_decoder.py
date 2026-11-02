"""Experiment 21: does the selection advantage of dV survive a finite budget?

This is not a test of whether Monte Carlo repairs the analytic decoder. The
"selection quality" measured by exact Bayes decoding is **what infinite-sample
Monte Carlo would reach** on the same observation set. The question is therefore
whether the gap survives a finite budget rather than vanishing into sampling
error.

Episodes are kept separate: estimation and evaluation share neither worlds nor
random draws.

  1. each method chooses an observation set A without seeing the true world
  2. a shared evaluation world X is drawn and only X_A is revealed
  3. membership probabilities are estimated from conditional Monte Carlo samples
     independent of X
  4. the top k of the estimated probabilities is reported, with a deterministic
     tie-break
  5. the symmetric-difference loss against the true T(X) is measured

What is reported:
  R_star          the limiting performance of the chosen set, by exact Bayes
                  decoding: what infinite-sample Monte Carlo would reach
  R_mc            the loss in use, decoding from a finite sample
  R_mc - R_star   the extra loss the Monte Carlo decoder adds
  plus the total regret against the global oracle, wall time and the number of
  PageRank solves

The oracle is the **true global optimum over all subsets**, not greedy.

Usage:
    python -m uncertain_pagerank.experiments.exp21_mc_decoder \\
        --config configs/experiment/exp21_mc_decoder.yaml
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import yaml

from ..acquisition import _strict_topk, enumerate_worlds, expected_loss
from ..acquisition_scalable import positions_to_edge_ids, uncertain_edge_ids
from ..expected_transition import expected_transition
from ..pagerank import pagerank, uniform_vector
from ..variance import contribution_vectors, edge_acquisition_scores
from .exp17_timescale_gate import build

REPO = Path(__file__).resolve().parents[3]


def global_oracle(ens, n_unc: int, budget: int) -> tuple[tuple[int, ...], float]:
    """The true global oracle for the fixed-set problem, over all subsets."""
    best, best_loss = (), np.inf
    for combo in itertools.combinations(range(n_unc), budget):
        loss = expected_loss(ens, edges=list(combo))
        if loss < best_loss:
            best, best_loss = combo, loss
    return best, best_loss


def _batch_scores(graph, probs, alpha, z, tol, n_samples, seed, chunk=1024):
    """Solve the PageRank of n_samples worlds at once.

    Uses the **same random stream and the same iteration** as the sequential
    route (`WorldSampler` plus `pagerank`): `rng.random((S, m))` returns the same
    sequence as calling `rng.random(m)` S times, so the masks drawn match.

    Only the stopping rule differs. The sequential route stops per world, while
    this one iterates until the residual of every world is below tol. Each
    iteration is a contraction towards its own fixed point, so a world that ran
    on only converged further; the difference is below tol in L1.

    At around n=16, a dense tensor is more than an order of magnitude faster.
    """
    n, m = graph.n, graph.m
    rng = np.random.default_rng(seed)
    src = np.repeat(np.arange(n), graph.out_degrees())
    cell = src * n + graph.indices  # the flat position of each link in (n,n)
    has_dup = np.unique(cell).size != cell.size
    indptr = graph.indptr
    out = np.empty((n_samples, n))

    done = 0
    while done < n_samples:
        s_n = min(chunk, n_samples - done)
        mask = (rng.random((s_n, m)) < probs[None, :]).astype(np.float64)

        csum = np.zeros((s_n, m + 1))
        np.cumsum(mask, axis=1, out=csum[:, 1:])
        deg = csum[:, indptr[1:]] - csum[:, indptr[:-1]]  # (s_n, n)
        dpe = deg[:, src]
        data = np.divide(mask, dpe, out=np.zeros_like(mask), where=dpe > 0)

        b = np.zeros((s_n, n * n))
        if has_dup:  # with parallel edges the scatter has to accumulate
            np.add.at(b, (np.arange(s_n)[:, None], cell[None, :]), data)
        else:
            b[:, cell] = data
        b = b.reshape(s_n, n, n)
        delta = (deg == 0).astype(np.float64)

        y = np.broadcast_to(z, (s_n, n)).copy()
        for _ in range(10_000):
            # y^T B as a batched matrix product: this reaches BLAS, einsum does not
            yb = np.matmul(y[:, None, :], b)[:, 0, :]
            dang = (delta * y).sum(axis=1)[:, None]
            y_next = alpha * (yb + z[None, :] * dang) + (1.0 - alpha) * z[None, :]
            resid = np.abs(y_next - y).sum(axis=1)
            y = y_next
            if resid.max() < tol:
                break
        out[done : done + s_n] = y
        done += s_n
    return out


def mc_decode(
    graph,
    k: int,
    observe_ids: np.ndarray,
    observed_states: np.ndarray,
    alpha: float,
    z: np.ndarray,
    n_samples: int,
    seed: int,
    tol: float,
    tie_priority: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Estimate the membership probabilities from worlds consistent with the
    observation.

    The observed links are pinned to their states and only the rest are drawn,
    with random numbers independent of the evaluation world.

    **The top-k set of each sample is cut by the same deterministic rule as the
    exact T(X).** Using metrics.topk_indicator, which splits ties evenly, would
    have the Monte Carlo and exact routes learn different targets, and then the
    estimate does not converge to the exact Bayes decoder as M grows. On the star
    that is fatal: worlds tied at the top-k boundary carry 0.70 to 0.95 of the
    probability mass.

    Returns:
        (the reported top-k set, the number of PageRank solves).
    """
    probs = graph.probs.copy()
    if observe_ids.size:
        probs[observe_ids] = observed_states.astype(np.float64)
    scores = _batch_scores(graph, probs, alpha, z, tol, n_samples, seed)
    memb = _strict_topk(scores, k, tie_priority)[0].mean(axis=0)
    report = np.flatnonzero(_strict_topk(memb[None, :], k, tie_priority)[0][0] > 0)
    return report, n_samples


def selection_scores(graph, k, alpha, z, unc, rng) -> dict[str, np.ndarray]:
    """Scores for choosing an observation set. The true world is never seen.

    At a budget of two or more, every one of these is a **fixed prefix of the
    per-link scores**, a one-shot top-B heuristic. Since dV_e is not additive,
    this is not greedy value of information.

    Note: the six methods are computed together, so the time returned is the
    total over all of them, and cannot be used for a per-method time-loss
    comparison.
    """
    et = expected_transition(graph, method="dp")
    y0 = pagerank(et, alpha=alpha, z=z).scores
    order = np.argsort(-y0, kind="stable")
    c = contribution_vectors(et, np.array([order[k - 1], order[k]]), alpha=alpha, z=z)
    x = c[:, 0] - c[:, 1]
    dv = edge_acquisition_scores(graph, et, y0, x, alpha=alpha, z=z)[unc]
    q = graph.probs[unc]
    src = np.repeat(np.arange(graph.n), graph.out_degrees())[unc]
    tgt = graph.indices[unc]
    deg = graph.out_degrees()
    nbr = [set(graph.neighbors(u)[0].tolist()) for u in range(graph.n)]
    overlap = np.array([
        len(nbr[int(s)] & nbr[int(t)]) / max(len(nbr[int(s)] | nbr[int(t)]), 1)
        for s, t in zip(src, tgt)
    ])
    return {
        "dV_edge": dv,
        "pagerank_src": y0[src],
        "q_variance": q * (1 - q),
        "inv_outdeg": 1.0 / deg[src].astype(float),
        "bridge": q * (1 - q) * (1.0 - overlap),
        "random": rng.random(unc.size),
    }


def mc_operational_loss(
    ens, graph, k, pick, alpha, z, n_mc, seed, tol, tie_priority=None
) -> tuple[float, int]:
    """The expected loss of using the Monte Carlo decoder, scored **exactly by
    enumeration**.

    There are only 2^|A| outcomes, so each branch is decoded once and the
    evaluation is a weighted sum over the worlds. No episode sampling error
    enters, which makes small differences detectable.

    Returns:
        (the expected loss, the number of PageRank solves).
    """
    # Sort, so that the outcome code and the random draws match when the same
    # set is chosen in a different order
    pick = np.sort(np.asarray(pick, dtype=np.int64))
    ids = positions_to_edge_ids(graph, pick) if pick.size else np.empty(0, dtype=np.int64)
    total, solves = 0.0, 0
    for code in range(1 << pick.size):
        states = (code >> np.arange(pick.size)) & 1 if pick.size else np.empty(0, dtype=np.int64)
        if pick.size:
            match = np.all(ens.bits[:, pick] == states[None, :], axis=1)
        else:
            match = np.ones(ens.bits.shape[0], dtype=bool)
        mass = float(ens.weights[match].sum())
        if mass <= 0.0:
            continue
        report, used = mc_decode(graph, k, ids, states, alpha, z, n_mc,
                                 seed=seed + code, tol=tol, tie_priority=tie_priority)
        solves += used
        ind = np.zeros(graph.n)
        ind[report] = 1.0
        total += float(ens.weights[match] @ (1.0 - (ens.topk[match] @ ind) / k))
    return total, solves


def run(config: dict) -> list[dict]:
    rows: list[dict] = []
    alpha, tol = float(config["alpha"]), float(config["tol"])
    k = int(config["topk"])
    budgets = [int(b) for b in config["budgets"]]
    mc_sizes = [int(m) for m in config["decoder_samples"]]
    n_repeats = int(config["decoder_repeats"])

    for spec in config["graphs"]:
        for seed in config["seeds"]:
            graph = build(spec, np.random.default_rng(int(seed)))
            if spec.get("personalization") == "skewed":
                # A uniform z produces a great many ties on the star. This
                # control breaks them with an asymmetric z, to separate the
                # Gaussian failure from the sheer density of ties.
                w = np.random.default_rng(int(seed) + 41).uniform(0.5, 1.5, size=graph.n)
                z = w / w.sum()
            else:
                z = uniform_vector(graph.n)
            unc = uncertain_edge_ids(graph)
            if unc.size < max(budgets) + 1 or k + 1 >= graph.n:
                continue
            ens = enumerate_worlds(graph, k, alpha=alpha, z=z)
            # The decoding depends only on (observation set, sample count,
            # seed), not on which method chose that set. The methods often
            # agree, so the result is memoised rather than recomputed. The
            # measured time from the first computation is carried along, and
            # reuse is marked in a column.
            decode_cache: dict = {}

            t0 = time.perf_counter()
            scores = selection_scores(
                graph, k, alpha, z, unc, np.random.default_rng(int(seed) + 3)
            )
            t_select = time.perf_counter() - t0

            for budget in budgets:
                oracle_set, oracle_loss = global_oracle(ens, unc.size, budget)
                for name, sc in scores.items():
                    pick = np.argsort(-sc, kind="stable")[:budget]
                    r_star = expected_loss(ens, edges=pick)
                    common = {
                        "label": spec["label"], "seed": int(seed), "k": k,
                        "n": graph.n, "m": graph.m, "n_uncertain": int(unc.size),
                        "budget": budget, "method": name,
                        "selected": ",".join(map(str, sorted(pick.tolist()))),
                        "select_seconds": t_select,
                        "R_star": r_star,
                        "oracle_loss": oracle_loss,
                        "regret_star": r_star - oracle_loss,
                        "oracle_set": ",".join(map(str, sorted(oracle_set))),
                        # The tied mass; needed to compare against the
                        # asymmetric-z control, so it is kept in the row
                        "tied_mass": ens.tied_mass,
                    }
                    for n_mc in mc_sizes:
                        # One row per decoder seed; aggregation comes later
                        for rep in range(n_repeats):
                            dseed = int(seed) * 10_000 + rep * 101 + 1
                            key = (tuple(sorted(pick.tolist())), n_mc, dseed)
                            hit = key in decode_cache
                            if hit:
                                loss, used, dt = decode_cache[key]
                            else:
                                t1 = time.perf_counter()
                                loss, used = mc_operational_loss(
                                    ens, graph, k, pick, alpha, z, n_mc,
                                    seed=dseed, tol=tol,
                                )
                                dt = time.perf_counter() - t1
                                decode_cache[key] = (loss, used, dt)
                            rows.append({
                                **common, "decoder_samples": n_mc,
                                "decoder_repeat": rep,
                                "R_mc": loss,
                                "decoder_gap": loss - r_star,
                                "total_regret": loss - oracle_loss,
                                "decode_seconds": dt,
                                "decode_cached": hit,
                                "pagerank_solves": used,
                            })
            print(f"  [{spec['label']:22s} seed={seed}] -> {len(rows)} rows", flush=True)
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
