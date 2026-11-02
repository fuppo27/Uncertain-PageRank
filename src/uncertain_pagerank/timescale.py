"""How the ratio of network update speed to diffusion speed picks the centrality.

Consider a random walk that redraws the possible world every L transitions.
Within a block the world is fixed, so the kernel of the chain seen at the block
boundaries is

    T_L = E_X[ G_X^L ],    G_X = alpha P_X + (1 - alpha) 1 z^T

With pi_L its stationary distribution, the visiting distribution over all steps is

    d_L = (1/L) sum_{j=0}^{L-1} E_X[ pi_L^T G_X^j ]

Both limits follow from the model:

    L = 1        : T_1 = E[G_X] is the Google matrix of Pbar, so
                   d_1 = PR(Pbar) = p-avg
    L -> infinity: G_X^L -> 1 y_X^T, so d_inf = E_X[y_X] = s-avg

so what is of interest lies in between. The time scale of PageRank diffusion is
the mean number of steps before a teleport, 1/(1-alpha). Normalising the update
interval L by it,

    tau = L (1 - alpha)

counts the teleports per block. When tau << 1 the world changes many times
within one excursion, so the behaviour is expected to sit near the p-average;
when tau >> 1 the world is fixed throughout an excursion, so near the
s-average.

On small graphs d_L can be computed exactly by enumerating every world, with
neither approximation nor sampling error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .graph import UncertainGraph
from .pagerank import _resolve_z
from .possible_worlds import _build_transitions, _world_chunks


@dataclass
class TimescaleResult:
    """The visiting distribution at update interval L."""

    visit: np.ndarray
    """The visiting distribution d_L over all steps."""

    block_start: np.ndarray
    """The distribution pi_L at the block boundaries."""

    update_interval: int
    tau: float
    """L (1 - alpha): the number of teleports per block."""

    iterations: int


def _google_matrices(
    graph: UncertainGraph, z: np.ndarray, alpha: float, max_worlds: int
) -> tuple[np.ndarray, np.ndarray]:
    """The transposed Google matrix of every possible world, with its probability."""
    n = graph.n
    edges, probs = graph.edge_list(), graph.probs
    certain = edges[probs >= 1.0]
    uncertain = edges[(probs > 0.0) & (probs < 1.0)]
    q_u = probs[(probs > 0.0) & (probs < 1.0)]
    m_u = int(uncertain.shape[0])
    if (1 << m_u) > max_worlds:
        raise ValueError(f"2^{m_u} worlds exceeds the limit {max_worlds}")

    mats, weights = [], []
    for bits in _world_chunks(m_u, 1 << 14):
        w = np.prod(np.where(bits > 0, q_u, 1.0 - q_u), axis=1) if m_u else np.ones(1)
        p = _build_transitions(bits, n, certain, uncertain, z)
        g = alpha * p + (1.0 - alpha) * z[None, None, :]
        mats.append(np.ascontiguousarray(np.swapaxes(g, 1, 2)))  # G^T
        weights.append(w)
    return np.concatenate(mats, axis=0), np.concatenate(weights)


def visit_distribution(
    graph: UncertainGraph,
    update_interval: int,
    alpha: float = 0.85,
    z: np.ndarray | None = None,
    tol: float = 1e-13,
    max_iter: int = 2000,
    max_worlds: int = 1 << 16,
) -> TimescaleResult:
    """The visiting distribution at update interval L, exactly by enumeration.

    Args:
        graph: the uncertain graph.
        update_interval: the interval L at which the world is redrawn, at least 1.
        alpha: the damping factor.
        z: the personalization vector.
        tol: the convergence threshold on the block-boundary distribution.
        max_iter: the iteration limit.
        max_worlds: the largest number of worlds to enumerate.
    """
    n = graph.n
    z = _resolve_z(z, n)
    L = int(update_interval)
    if L < 1:
        raise ValueError("update_interval must be at least one")
    gt, w = _google_matrices(graph, z, alpha, max_worlds)

    pi = z.copy()
    iterations = 0
    for iterations in range(1, max_iter + 1):
        x = np.broadcast_to(pi, (gt.shape[0], n)).copy()
        for _ in range(L):
            x = np.einsum("bij,bj->bi", gt, x)
        nxt = w @ x
        nxt /= nxt.sum()
        if float(np.abs(nxt - pi).sum()) < tol:
            pi = nxt
            break
        pi = nxt

    acc = np.zeros(n)
    x = np.broadcast_to(pi, (gt.shape[0], n)).copy()
    for _ in range(L):
        acc += w @ x
        x = np.einsum("bij,bj->bi", gt, x)
    acc /= acc.sum()
    return TimescaleResult(
        visit=acc,
        block_start=pi,
        update_interval=L,
        tau=L * (1.0 - alpha),
        iterations=iterations,
    )
