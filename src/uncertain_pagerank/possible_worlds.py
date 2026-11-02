"""The exact s-average by enumerating all possible worlds, Eq. (4).

    y_s = sum_X q(G_X) PRS(P_X; alpha)

is computed exactly by enumerating all 2^{m_u} worlds, where m_u is the number
of uncertain links. No sampling error enters, so this replaces the sample mean
over S=10,000 worlds that the conference version used as ground truth.

The PageRank of each world is obtained by solving the linear system directly
rather than by iterating, so no truncation error enters either.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .graph import UncertainGraph
from .metrics import topk_indicator
from .pagerank import _resolve_z, pagerank_solve


@dataclass
class ExactSavgResult:
    """The result of an exact s-average computation."""

    scores: np.ndarray
    """s-avg: E_X[PageRank(P_X)]。"""

    second_moment: np.ndarray
    """E_X[PageRank(P_X)^2]。"""

    n_worlds: int
    """The number of worlds enumerated, 2^{m_u}."""

    weight_sum: float
    """The total probability of the enumerated worlds; its deviation from one
    indicates the numerical error."""

    topk_prob: dict[int, np.ndarray] = field(default_factory=dict)
    """k -> P(v is in the top k of a realized graph)."""

    expected_transition: np.ndarray | None = None
    """sum_X q(G_X) P_X, held densely to check the exact expected transition
    matrix."""

    expected_transition_sq: np.ndarray | None = None
    """sum_X q(G_X) P_X^2, elementwise, to check the second moments."""

    @property
    def variance(self) -> np.ndarray:
        """The variance Var_X[y_X(v)] of each node's PageRank score."""
        return np.maximum(self.second_moment - self.scores**2, 0.0)


def _world_chunks(m_u: int, chunk_size: int):
    """Generate the connection patterns of the uncertain links in chunks."""
    total = 1 << m_u
    shifts = np.arange(m_u, dtype=np.int64)
    for start in range(0, total, chunk_size):
        codes = np.arange(start, min(start + chunk_size, total), dtype=np.int64)
        yield ((codes[:, None] >> shifts) & 1).astype(np.float64)


def _build_transitions(
    bits: np.ndarray,
    n: int,
    certain: np.ndarray,
    uncertain: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """Build the row-stochastic P_X of each world from its connection pattern."""
    a = np.zeros((bits.shape[0], n, n), dtype=np.float64)
    if certain.size:
        a[:, certain[:, 0], certain[:, 1]] = 1.0
    if uncertain.size:
        a[:, uncertain[:, 0], uncertain[:, 1]] = bits
    deg = a.sum(axis=2)
    live = deg > 0
    p = a / np.where(live, deg, 1.0)[:, :, None]
    return np.where(live[:, :, None], p, z[None, None, :])


def exact_savg_alphas(
    graph: UncertainGraph,
    alphas: Sequence[float],
    z: np.ndarray | None = None,
    topk: tuple[int, ...] = (),
    with_expected_transition: bool = False,
    max_worlds: int = 1 << 22,
    chunk_size: int | None = None,
) -> dict[float, ExactSavgResult]:
    """Enumerate all worlds and compute the exact s-average for several alphas.

    Building the worlds does not depend on alpha, so sweeping alpha is much
    faster when one enumeration is reused.

    Args:
        graph: the uncertain graph. Links with q_e equal to 0 or 1 are treated
            as certain and left out of the enumeration.
        alphas: the damping factors.
        z: the personalization vector, also the jump target of dangling nodes.
            Uniform by default.
        topk: the values of k for which to compute P(v in top-k).
        with_expected_transition: whether to also compute sum_X q(G_X) P_X.
        max_worlds: the largest number of worlds to enumerate.
        chunk_size: how many worlds to process at once. Chosen from the
            available memory by default.
    Returns:
        alpha -> ExactSavgResult。
    """
    n = graph.n
    z = _resolve_z(z, n)
    alphas = [float(a) for a in alphas]
    edges, probs = graph.edge_list(), graph.probs

    is_certain = probs >= 1.0
    is_absent = probs <= 0.0
    is_uncertain = ~(is_certain | is_absent)
    certain, uncertain = edges[is_certain], edges[is_uncertain]
    q_u = probs[is_uncertain]
    m_u = int(uncertain.shape[0])

    n_worlds = 1 << m_u
    if n_worlds > max_worlds:
        raise ValueError(
            f"2^{m_u} = {n_worlds} worlds exceeds the limit {max_worlds}. "
            "Reduce the number of uncertain links or raise max_worlds."
        )
    if chunk_size is None:
        chunk_size = max(1, min(n_worlds, int(2**22 / max(n * n, 1))))

    sum_w = 0.0
    sum_y = {a: np.zeros(n) for a in alphas}
    sum_y2 = {a: np.zeros(n) for a in alphas}
    topk_acc = {a: {k: np.zeros(n) for k in topk} for a in alphas}
    sum_p = np.zeros((n, n)) if with_expected_transition else None
    sum_p2 = np.zeros((n, n)) if with_expected_transition else None

    for bits in _world_chunks(m_u, chunk_size):
        w = np.prod(np.where(bits > 0, q_u, 1.0 - q_u), axis=1) if m_u else np.ones(1)
        p = _build_transitions(bits, n, certain, uncertain, z)
        sum_w += float(w.sum())
        if sum_p is not None and sum_p2 is not None:
            sum_p += np.einsum("b,bij->ij", w, p)
            sum_p2 += np.einsum("b,bij->ij", w, p * p)
        for a in alphas:
            y = pagerank_solve(p, a, z)
            sum_y[a] += w @ y
            sum_y2[a] += w @ (y * y)
            for k in topk:
                topk_acc[a][k] += w @ topk_indicator(y, k)

    return {
        a: ExactSavgResult(
            scores=sum_y[a],
            second_moment=sum_y2[a],
            n_worlds=n_worlds,
            weight_sum=sum_w,
            topk_prob=topk_acc[a],
            expected_transition=sum_p,
            expected_transition_sq=sum_p2,
        )
        for a in alphas
    }


def exact_savg(
    graph: UncertainGraph,
    alpha: float = 0.85,
    z: np.ndarray | None = None,
    topk: tuple[int, ...] = (),
    with_expected_transition: bool = False,
    max_worlds: int = 1 << 22,
    chunk_size: int | None = None,
) -> ExactSavgResult:
    """The exact s-average by enumerating all worlds, for a single alpha.

    The arguments mean the same as in exact_savg_alphas.
    """
    return exact_savg_alphas(
        graph,
        [alpha],
        z=z,
        topk=topk,
        with_expected_transition=with_expected_transition,
        max_worlds=max_worlds,
        chunk_size=chunk_size,
    )[float(alpha)]


def exact_savg_sparse(
    graph: UncertainGraph,
    alpha: float = 0.85,
    z: np.ndarray | None = None,
    topk: tuple[int, ...] = (),
    max_worlds: int = 1 << 20,
    tol: float = 1e-12,
    n_workers: int = 1,
) -> ExactSavgResult:
    """The exact s-average by enumeration, keeping the matrices sparse.

    exact_savg builds a dense (worlds, n, n) array and so handles only a few
    dozen nodes. This form is for real networks of 10^4 to 10^5 nodes with the
    uncertain links restricted to a small set. Each world needs one iterative
    PageRank solve, so the time grows with the number of worlds.

    Args:
        graph: the uncertain graph. Links with q_e equal to 0 or 1 are treated
            as certain.
        alpha: the damping factor.
        z: the personalization vector.
        topk: the values of k for which to compute P(v in top-k).
        max_worlds: the largest number of worlds to enumerate.
        tol: the PageRank convergence threshold within each world.
        n_workers: the number of parallel processes.
    """
    import multiprocessing as mp

    n = graph.n
    z = _resolve_z(z, n)
    probs = graph.probs
    uncertain = np.flatnonzero((probs > 0.0) & (probs < 1.0))
    m_u = int(uncertain.size)
    n_worlds = 1 << m_u
    if n_worlds > max_worlds:
        raise ValueError(
            f"2^{m_u} = {n_worlds} worlds exceeds the limit {max_worlds}. "
            "Reduce the number of uncertain links or raise max_worlds."
        )

    global _SPARSE_CTX
    _SPARSE_CTX = {
        "graph": graph,
        "alpha": alpha,
        "z": z,
        "tol": tol,
        "uncertain": uncertain,
        "q_u": probs[uncertain],
        "topk": topk,
    }

    bounds = np.linspace(0, n_worlds, max(1, n_workers) + 1).astype(np.int64)
    jobs = [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    if n_workers > 1 and len(jobs) > 1:
        with mp.get_context("fork").Pool(processes=len(jobs)) as pool:
            parts = pool.map(_sparse_chunk, jobs)
    else:
        parts = [_sparse_chunk(job) for job in jobs]

    sum_w = float(sum(p[0] for p in parts))
    sum_y = np.sum([p[1] for p in parts], axis=0)
    sum_y2 = np.sum([p[2] for p in parts], axis=0)
    topk_acc = {k: np.sum([p[3][k] for p in parts], axis=0) for k in topk}
    return ExactSavgResult(
        scores=sum_y,
        second_moment=sum_y2,
        n_worlds=n_worlds,
        weight_sum=sum_w,
        topk_prob=topk_acc,
    )


_SPARSE_CTX: dict = {}


def _sparse_chunk(job: tuple[int, int]):
    """Process one slice of the worlds; the worker used by exact_savg_sparse."""
    from .pagerank import pagerank
    from .sampling import WorldSampler

    start, stop = job
    ctx = _SPARSE_CTX
    graph, uncertain, q_u = ctx["graph"], ctx["uncertain"], ctx["q_u"]
    alpha, z, tol, topk = ctx["alpha"], ctx["z"], ctx["tol"], ctx["topk"]
    m_u = uncertain.size
    shifts = np.arange(m_u, dtype=np.int64)

    sampler = WorldSampler(graph, np.random.default_rng(0))
    base_mask = (graph.probs > 0.0).astype(np.float64)  # a q=0 link is never present

    sum_w = 0.0
    sum_y = np.zeros(graph.n)
    sum_y2 = np.zeros(graph.n)
    acc = {k: np.zeros(graph.n) for k in topk}

    for code in range(start, stop):
        bits = (code >> shifts) & 1
        mask = base_mask.copy()
        mask[uncertain] = bits
        w = float(np.prod(np.where(bits > 0, q_u, 1.0 - q_u))) if m_u else 1.0
        y = pagerank(sampler.transition(mask), alpha=alpha, z=z, tol=tol).scores
        sum_w += w
        sum_y += w * y
        sum_y2 += w * y * y
        for k in topk:
            acc[k] += w * topk_indicator(y, k)
    return sum_w, sum_y, sum_y2, acc
