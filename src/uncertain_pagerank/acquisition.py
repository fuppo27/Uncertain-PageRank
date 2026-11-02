"""Recovering the top-k of one fixed latent world from limited observation.

Let the latent world be X ~ prod_e Bernoulli(q_e) and its true top-k set
T(X) = TopK(PR(P_X)). The q_e are known, so the s-average is a fixed computable
quantity: what observation reveals is not the s-average but the **one fixed,
unknown world X**.

A source probe S returns the states of all uncertain out-links of each source at
once. After observing X_S = o, the set reported is

    That_S(o) = TopK{ Pr(v in T(X) | X_S = o) }_v

Since the symmetric-difference loss is |T delta That| = 2k - 2|T and That|,
minimising E|T delta That| is the same as maximising
sum_{v in That} Pr(v in T(X) | o), so the **Bayes-optimal report is the top-k of
the membership probabilities**, not the top-k of the conditional mean scores.

The objective is the expected symmetric-difference loss

    R(S) = E_X[ |T(X) Δ That_S(X_S)| / (2k) ].

Scope: this applies only where the latent link states are fixed for the
duration of the decision and a query reveals them. Where q_e is a temporal
activity probability, one observation does not collapse q_e to 0 or 1.

This module computes the quantities exactly on small graphs. Every world is
enumerated, so the number of uncertain links is limited to about 20.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .graph import UncertainGraph
from .pagerank import _resolve_z, pagerank_solve
from .possible_worlds import _build_transitions, _world_chunks


@dataclass
class WorldEnsemble:
    """Every possible world with its top-k set; the basis of the exact computation."""

    weights: np.ndarray
    """Length W: the probability q(G_X) of each world."""

    topk: np.ndarray
    """(W, n): the 0/1 indicator of the top-k set of each world, ties broken by
    index."""

    bits: np.ndarray
    """(W, m_u): the connection state of the uncertain links."""

    uncertain_edges: np.ndarray
    """Length m_u: the link index of each uncertain link."""

    edge_source: np.ndarray
    """Length m_u: the source node of each uncertain link."""

    k: int

    n_tied_worlds: int = 0
    """How many worlds had a tie at the top-k boundary. Diagnostic.

    In the world where every uncertain link is absent, all nodes are dangling and
    PageRank is uniform, so on a small graph ties occur at a rate that cannot be
    ignored. The decision problem needs T(X) to be a set, so ties are broken by
    index, but the rate is reported."""

    tied_mass: float = 0.0
    """The probability mass of the worlds with a tie."""

    tie_priority: np.ndarray | None = None
    """The total order that breaks ties. The evaluation uses the same one."""

    @property
    def n_worlds(self) -> int:
        return int(self.weights.shape[0])

    @property
    def marginal(self) -> np.ndarray:
        """Pr(v in T(X)): the prior membership probability, with no observation."""
        return self.weights @ self.topk


TIE_RTOL = 1e-12
"""The relative tolerance within which scores count as tied. Used both to
detect ties and to order."""


def _strict_topk(
    y: np.ndarray, k: int, tie_priority: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """The 0/1 indicator of the top-k set, ties broken by a fixed total order.

    The decision problem needs T(X) to be a set. For P(v in TopK) as a
    descriptive probability use metrics.topk_indicator, which splits ties
    evenly; this function is for the set that a decision is made about.

    **The same tolerance must be used to detect ties and to order.** Ordering by
    the raw values lets rounding error of order 1e-17 decide the order even where
    the scores are mathematically tied, so tie_priority never takes effect. In
    the world where every node is dangling, for instance, PageRank is exactly
    uniform, yet a direct solve and an iterative one differ in the last digit, so
    the exact and analytic routes would return different sets. Quantising to the
    tolerance before ordering makes them agree.

    Args:
        y: (W, n) scores.
        k: the size of the top set.
        tie_priority: length n. Among tied nodes the smaller value wins. The
            node index by default.
    Returns:
        ((W, n) 0/1 indicator, (W,) tie flags).
    """
    w, n = y.shape
    scale = np.maximum(np.abs(y).max(axis=1, keepdims=True), 1.0)
    atol = TIE_RTOL * scale
    quantized = np.round(y / atol)  # values within the tolerance map to one integer
    prio = (
        np.broadcast_to(np.arange(n, dtype=np.float64), (w, n))
        if tie_priority is None
        else np.broadcast_to(np.asarray(tie_priority, dtype=np.float64), (w, n))
    )
    order = np.lexsort((prio, -quantized), axis=1)
    ind = np.zeros_like(y)
    rows = np.repeat(np.arange(w), min(k, n))
    ind[rows, order[:, :k].ravel()] = 1.0
    if k >= n:
        return ind, np.zeros(w, dtype=bool)
    part = np.take_along_axis(quantized, order, axis=1)
    tied = part[:, k - 1] == part[:, k]
    return ind, tied


def enumerate_worlds(
    graph: UncertainGraph,
    k: int,
    alpha: float = 0.85,
    z: np.ndarray | None = None,
    max_worlds: int = 1 << 20,
    chunk_size: int | None = None,
    tie_priority: np.ndarray | None = None,
) -> WorldEnsemble:
    """Enumerate the PageRank and the top-k set of every possible world.

    Args:
        tie_priority: the total order that breaks ties. The node index by
            default; varying it checks that the conclusions do not depend on it.
    """
    n = graph.n
    z = _resolve_z(z, n)
    edges, probs = graph.edge_list(), graph.probs
    is_certain = probs >= 1.0
    is_uncertain = ~(is_certain | (probs <= 0.0))
    certain, uncertain = edges[is_certain], edges[is_uncertain]
    q_u = probs[is_uncertain]
    m_u = int(uncertain.shape[0])
    if (1 << m_u) > max_worlds:
        raise ValueError(f"2^{m_u} worlds exceeds the limit {max_worlds}")
    if chunk_size is None:
        chunk_size = max(1, min(1 << m_u, int(2**22 / max(n * n, 1))))

    w_all, t_all, b_all, tie_flags, tie_mass = [], [], [], [], []
    for bits in _world_chunks(m_u, chunk_size):
        w = np.prod(np.where(bits > 0, q_u, 1.0 - q_u), axis=1) if m_u else np.ones(1)
        p = _build_transitions(bits, n, certain, uncertain, z)
        y = pagerank_solve(p, alpha, z)
        ind, tied = _strict_topk(y, k, tie_priority)
        w_all.append(w)
        t_all.append(ind)
        b_all.append(bits.astype(np.int8))
        tie_flags.append(tied)
        tie_mass.append(float(w[tied].sum()))
    return WorldEnsemble(
        weights=np.concatenate(w_all),
        topk=np.concatenate(t_all, axis=0),
        bits=np.concatenate(b_all, axis=0),
        uncertain_edges=np.flatnonzero(is_uncertain),
        edge_source=edges[is_uncertain][:, 0],
        k=k,
        n_tied_worlds=int(np.concatenate(tie_flags).sum()) if tie_flags else 0,
        tied_mass=float(sum(tie_mass)),
        tie_priority=None if tie_priority is None else np.asarray(tie_priority),
    )


def expected_loss(
    ensemble: WorldEnsemble,
    sources: np.ndarray | list[int] | None = None,
    edges: np.ndarray | list[int] | None = None,
) -> float:
    """R(S) = E_X[ |T(X) delta That_S(X_S)| / (2k) ], exactly.

    Worlds are grouped by the observed out-link states, and within each group the
    report is the top-k of the conditional membership probabilities.

    Observation can be by source probe, revealing a source's out-links together,
    or by edge probe, revealing individual links. Giving both observes the union.

    Args:
        ensemble: the enumerated worlds.
        sources: the source nodes observed.
        edges: the uncertain links observed, as positions in uncertain_edges.
    Returns:
        The expected symmetric-difference loss, in [0, 1].
    """
    w, ind, k = ensemble.weights, ensemble.topk, ensemble.k
    cols_list = []
    if sources is not None:
        src = np.asarray(list(sources), dtype=np.int64)
        if src.size:
            cols_list.append(np.flatnonzero(np.isin(ensemble.edge_source, src)))
    if edges is not None:
        e = np.asarray(list(edges), dtype=np.int64)
        if e.size:
            cols_list.append(e)
    cols = np.unique(np.concatenate(cols_list)) if cols_list else np.empty(0, np.int64)

    if cols.size == 0:
        group = np.zeros(w.shape[0], dtype=np.int64)
        n_groups = 1
    else:
        # Pack the observed link states into one integer code
        code = ensemble.bits[:, cols].astype(np.int64) @ (1 << np.arange(cols.size, dtype=np.int64))
        _, group = np.unique(code, return_inverse=True)
        n_groups = int(group.max()) + 1

    mass = np.bincount(group, weights=w, minlength=n_groups)
    memb = np.zeros((n_groups, ind.shape[1]))
    for v in range(ind.shape[1]):
        memb[:, v] = np.bincount(group, weights=w * ind[:, v], minlength=n_groups)
    report, _ = _strict_topk(
        memb / np.maximum(mass, 1e-300)[:, None], k, ensemble.tie_priority
    )

    hit = np.einsum("wv,wv->w", ind, report[group])
    return float(w @ (1.0 - hit / k))


def true_voi_edge(ensemble: WorldEnsemble) -> np.ndarray:
    """The value of information of observing one link, exactly.

    V(e) = R(empty) - R({e}), one value per uncertain link.
    """
    base = expected_loss(ensemble, [])
    return np.array([
        base - expected_loss(ensemble, edges=[i])
        for i in range(ensemble.uncertain_edges.size)
    ])


def true_voi(
    ensemble: WorldEnsemble, candidates: np.ndarray | list[int] | None = None
) -> np.ndarray:
    """The one-step value of information V(u) = R(empty) - R({u}), exactly.

    Args:
        ensemble: the enumerated worlds.
        candidates: the source nodes to evaluate. Every node with an uncertain
            out-link by default.
    Returns:
        Length len(candidates), the values of information.
    """
    base = expected_loss(ensemble, [])
    cand = (
        np.unique(ensemble.edge_source) if candidates is None
        else np.asarray(list(candidates), dtype=np.int64)
    )
    return np.array([base - expected_loss(ensemble, [int(u)]) for u in cand])


def greedy_probe(
    ensemble: WorldEnsemble, budget: int, candidates: np.ndarray | None = None
) -> tuple[list[int], list[float]]:
    """Greedy selection on the true loss: the oracle. Small scale only."""
    cand = list(np.unique(ensemble.edge_source) if candidates is None else candidates)
    chosen: list[int] = []
    losses = [expected_loss(ensemble, [])]
    for _ in range(min(budget, len(cand))):
        best, best_loss = None, np.inf
        for u in cand:
            if u in chosen:
                continue
            loss = expected_loss(ensemble, chosen + [int(u)])
            if loss < best_loss:
                best, best_loss = int(u), loss
        if best is None:
            break
        chosen.append(best)
        losses.append(best_loss)
    return chosen, losses
