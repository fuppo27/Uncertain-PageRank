"""Scaling the acquisition problem: enumeration replaced by an analytic
P(v in TopK).

Conditioning on an observation is trivial in this model: learning the state of
link e simply yields **another uncertain graph**, with q_e pinned to 0 or 1, so
the conditional quantities come from the same machinery applied again.

The expected loss

    R(S) = E_o[ 1 - (1/k) * sum_{v in That(o)} Pr(v in T(X) | X_S = o) ]

is then computed by enumerating the 2^|S| outcomes o and, for each, getting the
membership probabilities from the p-average and covariance of the conditional
graph under a normal approximation.

**The cost is 2^|S| in the observation budget, not 2^m in the worlds.** The size
of the graph enters only the cost of the covariance, so the question "which five
links should be inspected in a network of 100,000 nodes" becomes answerable.

Limitations:
- Nodes outside the candidate set (the top L by y0) cannot enter the top k, so
  L >> k is needed and the coverage is reported alongside.
- The normal approximation adds error of its own, measured against exhaustive
  enumeration at small scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .expected_transition import expected_transition
from .graph import UncertainGraph
from .pagerank import _resolve_z, pagerank
from .acquisition import _strict_topk
from .variance import score_covariance


def _membership_strict(
    mean: np.ndarray,
    cov: np.ndarray,
    k: int,
    n_draws: int,
    seed: int,
    tie_priority: np.ndarray | None,
    zero_tol: float = 1e-14,
) -> tuple[np.ndarray, float]:
    """Pr(v in TopK) from samples of a multivariate normal.

    Ties are broken by the same total order as the exact route. Using
    metrics.topk_indicator, which splits ties evenly, would leave the membership
    fractional even under full observation with zero variance, so the risk would
    not fall to zero; the decision problem needs the top-k set to be a set.

    Rounding error in the covariance needs care too. Once every link has been
    observed the uncertainty is gone and the covariance is zero in theory, but
    numerically a residual of order 1e-17 remains. That tiny jitter **breaks ties
    at random** and defeats tie_priority: in the world where every node is
    dangling, PageRank is uniform, and the effect leaves memberships like 0.46,
    0.45 and so on. Covariances that are effectively zero are therefore handled
    without drawing samples.

    The real fix is not the threshold but **skipping the covariance entirely when
    no uncertain link remains**. The threshold is only a safety net for the
    partially observed case, and the amount clipped off is returned so that it
    can be recorded, in case a small but meaningful variance is being erased.

    Args:
        zero_tol: treat a covariance whose diagonal falls below this as zero.
            A safety net.
    Returns:
        (the membership probabilities, the total magnitude clipped from
        negative eigenvalues).
    """
    n = mean.size
    if k >= n:
        return np.ones(n), 0.0
    cov = 0.5 * (cov + cov.T)
    scale = float(np.abs(np.diag(cov)).max())
    if scale <= zero_tol * max(float(np.abs(mean).max()) ** 2, 1e-300):
        ind, _ = _strict_topk(mean[None, :], k, tie_priority)
        return ind[0], 0.0
    eig, vec = np.linalg.eigh(cov)
    clipped = float(np.abs(np.minimum(eig, 0.0)).sum())
    root = vec * np.sqrt(np.maximum(eig, 0.0))
    rng = np.random.default_rng(seed)
    draws = mean[None, :] + rng.standard_normal((n_draws, n)) @ root.T
    ind, _ = _strict_topk(draws, k, tie_priority)
    return ind.mean(axis=0), clipped


@dataclass
class AnalyticDecoding:
    """The conditional decoding for one observation outcome."""

    candidates: np.ndarray
    """The candidate set: the top L by the conditional p-average."""

    membership: np.ndarray
    """Pr(v in T(X) | o) on the candidate set."""

    report: np.ndarray
    """The reported top-k set: the top k by membership."""

    weight: float
    """The probability Pr(X_S = o) of this outcome."""

    clipped_eigenvalue_mass: float = 0.0
    """The magnitude clipped from negative eigenvalues of the covariance. A
    diagnostic for the soundness of the approximation."""

    n_uncertain_remaining: int = 0
    """How many uncertain links remain in this conditional graph. At zero, the
    covariance is skipped."""


def uncertain_edge_ids(graph: UncertainGraph) -> np.ndarray:
    """The **link ids in the original graph** of the uncertain links.

    There are two index spaces, and they must not be confused:

    - **link id in the original graph**: 0..m-1, including certain links. This
      is what condition_graph takes.
    - **position in the uncertain-link array**: 0..m_u-1. This is what the
      `edges` argument of acquisition.expected_loss takes.

    This function returns the former. Confusing the two produces the quiet error
    of "observing" a certain link and seeing nothing change.
    """
    return np.flatnonzero((graph.probs > 0.0) & (graph.probs < 1.0))


def positions_to_edge_ids(
    graph: UncertainGraph, positions: np.ndarray | list[int]
) -> np.ndarray:
    """Convert positions in the uncertain-link array to original link ids.

    Bridges the `edges` argument of acquisition.expected_loss, which takes
    positions, and observe_edge_ids here, which takes link ids.
    """
    ids = uncertain_edge_ids(graph)
    pos = np.asarray(list(positions), dtype=np.int64)
    if pos.size and (pos.min() < 0 or pos.max() >= ids.size):
        raise ValueError(f"position outside the uncertain-link range 0..{ids.size - 1}")
    return ids[pos]


def validate_edge_ids(graph: UncertainGraph, edge_ids: np.ndarray | list[int]) -> np.ndarray:
    """Check that the link ids to observe are valid.

    Silently accepting a certain link (q = 0 or 1), a duplicate or an out-of-range
    id produces the quiet error of observing something and seeing nothing
    change.
    """
    ids = np.asarray(list(edge_ids), dtype=np.int64)
    if ids.size == 0:
        return ids
    if ids.min() < 0 or ids.max() >= graph.m:
        raise ValueError(f"link id outside the range 0..{graph.m - 1}")
    if np.unique(ids).size != ids.size:
        raise ValueError("duplicate link id")
    bad = ids[~((graph.probs[ids] > 0.0) & (graph.probs[ids] < 1.0))]
    if bad.size:
        raise ValueError(
            f"certain links (q = 0 or 1) are among those to observe: link ids "
            f"{bad.tolist()}. Check that positions in the uncertain-link array "
            "were not passed instead; positions_to_edge_ids converts them"
        )
    return ids


def condition_graph(
    graph: UncertainGraph,
    edge_ids: np.ndarray | list[int],
    states: np.ndarray | list[int],
) -> UncertainGraph:
    """The graph with the observed links pinned to probability 0 or 1.

    Args:
        graph: the original uncertain graph.
        edge_ids: the **link ids in the original graph** of the links observed,
            not positions in the uncertain-link array.
        states: the observed states, 0 or 1.
    """
    probs = graph.probs.copy()
    probs[np.asarray(list(edge_ids), dtype=np.int64)] = np.asarray(list(states), dtype=np.float64)
    return graph.with_probs(probs)


def decode(
    graph: UncertainGraph,
    k: int,
    alpha: float = 0.85,
    z: np.ndarray | None = None,
    n_candidates: int = 50,
    n_draws: int = 50_000,
    seed: int = 0,
    tol: float = 1e-12,
    workers: int = 1,
    tie_priority: np.ndarray | None = None,
) -> AnalyticDecoding:
    """The top-k set to report for one conditional graph, analytically.

    Under the symmetric-difference loss the Bayes-optimal report is the top k by
    membership probability.
    """
    z = _resolve_z(z, graph.n)
    et = expected_transition(graph, method="dp", with_second_moment=False)
    y0 = pagerank(et, alpha=alpha, z=z, tol=tol).scores
    n_unc = int(np.count_nonzero((graph.probs > 0.0) & (graph.probs < 1.0)))
    cand_ind, _ = _strict_topk(
        y0[None, :], min(n_candidates, graph.n), tie_priority
    )
    cand = np.flatnonzero(cand_ind[0] > 0)
    cand = cand[np.argsort(-y0[cand], kind="stable")]
    cov = (
        score_covariance(graph, et, y0, cand, alpha=alpha, z=z, tol=tol, workers=workers)
        if n_unc else np.zeros((cand.size, cand.size))
    )
    prio = None if tie_priority is None else np.asarray(tie_priority)[cand]
    if np.count_nonzero((graph.probs > 0.0) & (graph.probs < 1.0)) == 0:
        # No uncertain link remains: the covariance is zero, so skip it and
        # return the strict decoder
        memb_ind, _ = _strict_topk(y0[cand][None, :], k, prio)
        memb, clipped = memb_ind[0], 0.0
    else:
        memb, clipped = _membership_strict(y0[cand], cov, k, n_draws, seed, prio)
    rep_ind, _ = _strict_topk(memb[None, :], k, prio)
    report = cand[np.flatnonzero(rep_ind[0] > 0)]
    return AnalyticDecoding(
        candidates=cand, membership=memb, report=report, weight=1.0,
        clipped_eigenvalue_mass=clipped, n_uncertain_remaining=n_unc,
    )


def expected_loss_analytic(
    graph: UncertainGraph,
    k: int,
    observe_edge_ids: np.ndarray | list[int],
    alpha: float = 0.85,
    z: np.ndarray | None = None,
    n_candidates: int = 50,
    n_draws: int = 50_000,
    seed: int = 0,
    tol: float = 1e-12,
    workers: int = 1,
    max_outcomes: int = 1 << 12,
    tie_priority: np.ndarray | None = None,
) -> tuple[float, list[AnalyticDecoding]]:
    """The expected loss R(S) for an observation set S, analytically.

    Enumerates the 2^|S| outcomes o and gets the conditional membership
    probabilities for each. Independent of the number of worlds 2^m.

    Args:
        graph: the uncertain graph.
        k: the size of the top set to report.
        observe_edge_ids: the **link ids in the original graph** of the links
            observed; empty for no observation. This is a different index space
            from the `edges` argument of acquisition.expected_loss, which takes
            positions; uncertain_edge_ids converts between them.
        n_candidates: the candidate-set size L. L >> k is needed.
        max_outcomes: the largest number of outcomes to enumerate.
    Returns:
        (R(S), the decoding of each outcome).
    """
    z = _resolve_z(z, graph.n)
    obs = validate_edge_ids(graph, observe_edge_ids)
    n_out = 1 << obs.size
    if n_out > max_outcomes:
        raise ValueError(f"2^{obs.size} = {n_out} outcomes exceeds the limit {max_outcomes}")

    q_obs = graph.probs[obs] if obs.size else np.empty(0)
    total = 0.0
    results: list[AnalyticDecoding] = []
    for code in range(n_out):
        states = (code >> np.arange(obs.size)) & 1 if obs.size else np.empty(0, dtype=np.int64)
        weight = float(np.prod(np.where(states > 0, q_obs, 1.0 - q_obs))) if obs.size else 1.0
        if weight <= 0.0:
            continue
        sub = condition_graph(graph, obs, states) if obs.size else graph
        d = decode(sub, k, alpha=alpha, z=z, n_candidates=n_candidates,
                   n_draws=n_draws, seed=seed + code, tol=tol, workers=workers,
                   tie_priority=tie_priority)
        d.weight = weight
        results.append(d)
        # The membership probabilities of the reported nodes sum to the
        # expected number of matches
        pos = {int(v): i for i, v in enumerate(d.candidates)}
        hit = sum(d.membership[pos[int(v)]] for v in d.report)
        total += weight * (1.0 - hit / k)
    return total, results


def operational_loss(
    ensemble,
    graph: UncertainGraph,
    k: int,
    observe_edge_ids: np.ndarray | list[int],
    alpha: float = 0.85,
    z: np.ndarray | None = None,
    n_candidates: int = 50,
    n_draws: int = 50_000,
    seed: int = 0,
    tol: float = 1e-12,
    tie_priority: np.ndarray | None = None,
) -> float:
    """The expected loss of actually using the analytic decoder, **scored on the
    exact worlds**.

    The exact expected_loss re-decodes with the exact Bayes decoder given an
    observation set, which measures the **quality of the selection** rather than
    the performance in use. In use it is the analytic decoder that reports, so its
    reported set has to be scored on the exact worlds.

    Keeping the two apart is what catches the case where the links are chosen well
    but the report is wrong.

    Args:
        ensemble: the enumerated worlds, used for the exact scoring.
        graph: the original uncertain graph.
        observe_edge_ids: the original link ids of the links observed.
    Returns:
        E_X[ |T(X) delta (analytic report)(X_S)| / (2k) ].
    """
    z = _resolve_z(z, graph.n)
    obs = validate_edge_ids(graph, observe_edge_ids)
    n_out = 1 << obs.size
    bits = ensemble.bits
    pos = {int(e): i for i, e in enumerate(ensemble.uncertain_edges)}
    cols = np.array([pos[int(e)] for e in obs], dtype=np.int64) if obs.size else np.empty(0, np.int64)

    total = 0.0
    for code in range(n_out):
        states = (code >> np.arange(obs.size)) & 1 if obs.size else np.empty(0, dtype=np.int64)
        if obs.size:
            match = np.all(bits[:, cols] == states[None, :], axis=1)
        else:
            match = np.ones(bits.shape[0], dtype=bool)
        mass = float(ensemble.weights[match].sum())
        if mass <= 0.0:
            continue
        sub = condition_graph(graph, obs, states) if obs.size else graph
        d = decode(sub, k, alpha=alpha, z=z, n_candidates=n_candidates,
                   n_draws=n_draws, seed=seed + code, tol=tol, tie_priority=tie_priority)
        report = np.zeros(graph.n)
        report[d.report] = 1.0
        hit = ensemble.topk[match] @ report
        total += float(ensemble.weights[match] @ (1.0 - hit / k))
    return total
