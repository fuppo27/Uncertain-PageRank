"""Propagate the covariance of the PageRank scores without sampling worlds.

The first-order expansion of the PageRank fixed point gives

    y_X - y0 ~= alpha R Delta_X^T y0,    R = (I - alpha Pbar^T)^{-1}

so the covariance of the scores is

    Cov(y) ~= alpha^2 R M R^T,
    M(v,w) = sum_u y0(u)^2 Cov(p_X(u,v), p_X(u,w))

Because the links of different sources are independent, the cross terms vanish
and M is a weighted sum of the per-row covariances. M follows exactly from the
second moments of the transition probabilities, and applying it to a vector
costs O(sum_u D_u^2), the same as building the expected transition matrix.

Since R^T = (I - alpha Pbar)^{-1}, the variance of node v is

    Var(y_v) ~= alpha^2 c_v^T M c_v,   c_v = (I - alpha Pbar)^{-1} e_v

where c_v is the PageRank contribution vector of v, obtained from one solve.
Computing it for a candidate set yields every variance and covariance within
that set, from which P(v in top-k) can be evaluated. No world is ever sampled.

Being a first-order expansion, it carries error when the uncertainty is large.
Its accuracy is measured against exhaustive enumeration on small graphs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .expected_transition import ExpectedTransition, _degree_buckets
from .graph import UncertainGraph
from .pagerank import _resolve_z
from .pair_moment import _pair_moment_rows_batched


def pair_moment_edges(
    graph: UncertainGraph,
    coeff: np.ndarray,
    max_batch_elements: int = 1 << 24,
) -> np.ndarray:
    """(A_u c_u)_i per link, for every node, by the exact O(D^2) dynamic program.

    Rows of out-degree 64 or more used to switch to an O(D) moment
    approximation. On real networks that only cost accuracy, so it was removed:
    Follow NW goes from 1.7 s to 67 s, and the per-source rank correlation rises
    from 0.960 to 0.981.

    Args:
        graph: the uncertain graph.
        coeff: length m, the coefficient of link e=(u,v), usually x[v].
        max_batch_elements: the most array elements to process at once.
    Returns:
        Length m, (A_u c_u)_i.
    """
    degrees = graph.out_degrees()
    out = np.zeros(graph.m, dtype=np.float64)
    last = max(graph.m - 1, 0)
    for d_max, nodes in _degree_buckets(degrees):
        cols = np.arange(d_max, dtype=np.int64)
        chunk = max(1, min(nodes.shape[0], max_batch_elements // d_max))
        for start in range(0, nodes.shape[0], chunk):
            part = nodes[start : start + chunk]
            idx = graph.indptr[part][:, None] + cols[None, :]
            valid = cols[None, :] < degrees[part][:, None]
            safe = np.minimum(idx, last)
            q = np.where(valid, graph.probs[safe], 0.0)
            c = np.where(valid, coeff[safe], 0.0)
            out[idx[valid]] = _pair_moment_rows_batched(q, c)[valid]
    return out


@dataclass
class CovarianceOperator:
    """Multiplication by M, where M(v,w) = sum_u y0(u)^2 Cov(p_X(u,v), p_X(u,w))."""

    graph: UncertainGraph
    et: ExpectedTransition
    y0: np.ndarray
    z: np.ndarray

    def __post_init__(self) -> None:
        self._w = self.y0 * self.y0
        self._bt = self.et.B.T.tocsr()
        self._src_w = np.repeat(self._w, np.diff(self.et.B.indptr))
        self._d1 = float(self._w @ self.et.delta)

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Return M x, in O(sum_u D_u^2 + |E| + |V|) time."""
        tgt = self.et.B.indices
        delta = self.et.delta
        zx = float(self.z @ x)

        pair = pair_moment_edges(self.graph, x[tgt])
        term1 = np.bincount(tgt, weights=self._src_w * pair, minlength=self.graph.n)

        s = self.et.B @ x + delta * zx  # Pbar(u,.) . x
        term3 = self._bt @ (self._w * s)
        d2 = float(self._w @ (delta * s))
        return term1 + self.z * (zx * self._d1 - d2) - term3

    def to_dense(self) -> np.ndarray:
        """Return M densely. For small-scale checks only."""
        n = self.graph.n
        return np.column_stack([self.apply(np.eye(n)[:, i]) for i in range(n)])


def contribution_vectors(
    et: ExpectedTransition,
    nodes: np.ndarray,
    alpha: float = 0.85,
    z: np.ndarray | None = None,
    tol: float = 1e-12,
    max_iter: int = 10_000,
) -> np.ndarray:
    """Solve c_v = (I - alpha Pbar)^{-1} e_v for the given nodes at once.

    Iterates c <- e_v + alpha (B c + delta (z . c)), holding the nodes as
    columns so that the sparse product is a single call.

    Returns:
        An (n, len(nodes)) array.
    """
    n = et.n
    z = _resolve_z(z, n)
    b = et.B
    delta = et.delta
    e = np.zeros((n, nodes.size))
    e[nodes, np.arange(nodes.size)] = 1.0

    c = e.copy()
    for _ in range(max_iter):
        nxt = e + alpha * (b @ c + np.outer(delta, z @ c))
        if float(np.abs(nxt - c).max()) < tol:
            return nxt
        c = nxt
    return c


_APPLY_CTX: dict = {}


def _apply_one(i: int) -> np.ndarray:
    """Apply M to one vector; the worker used by score_covariance."""
    return _APPLY_CTX["op"].apply(_APPLY_CTX["c"][:, i])


def score_covariance(
    graph: UncertainGraph,
    et: ExpectedTransition,
    y0: np.ndarray,
    nodes: np.ndarray,
    alpha: float = 0.85,
    z: np.ndarray | None = None,
    tol: float = 1e-12,
    workers: int = 1,
) -> np.ndarray:
    """The first-order covariance of the PageRank scores of the given nodes.

    Cov(y_v, y_w) ~= alpha^2 c_v^T M c_w。

    Args:
        graph: the uncertain graph.
        et: the expected transition matrix.
        y0: the p-average scores.
        nodes: the indices of the nodes of interest.
        alpha: the damping factor.
        z: the personalization vector.
        workers: processes over which to apply M. The vectors are independent,
            so they split directly.
    Returns:
        A (len(nodes), len(nodes)) covariance matrix.
    """
    z = _resolve_z(z, graph.n)
    c = contribution_vectors(et, nodes, alpha=alpha, z=z, tol=tol)
    op = CovarianceOperator(graph, et, y0, z)
    if workers > 1 and nodes.size > 1:
        import multiprocessing as mp

        _APPLY_CTX.update({"op": op, "c": c})
        with mp.get_context("fork").Pool(processes=min(workers, nodes.size)) as pool:
            columns = pool.map(_apply_one, range(nodes.size))
    else:
        columns = [op.apply(c[:, i]) for i in range(nodes.size)]
    mc = np.column_stack(columns)
    cov = alpha * alpha * (c.T @ mc)
    return 0.5 * (cov + cov.T)  # even out the asymmetry from rounding


def topk_probability(
    mean: np.ndarray,
    cov: np.ndarray,
    k: int,
    n_draws: int = 100_000,
    seed: int = 0,
) -> np.ndarray:
    """P(v is in the top k) under a multivariate normal approximation.

    Args:
        mean: the expected scores of the nodes of interest.
        cov: their covariance.
        k: the size of the top set.
        n_draws: how many draws to take from the normal law.
        seed: the random seed.
    Returns:
        Length len(mean), the probabilities.
    """
    from .metrics import topk_indicator

    n = mean.size
    if k >= n:
        return np.ones(n)
    eig, vec = np.linalg.eigh(cov)
    root = vec * np.sqrt(np.maximum(eig, 0.0))
    rng = np.random.default_rng(seed)
    draws = mean[None, :] + rng.standard_normal((n_draws, n)) @ root.T
    return topk_indicator(draws, k).mean(axis=0)


@dataclass
class Attribution:
    """A decomposition of a variance over source nodes."""

    total: float
    """The variance of the quantity in question."""

    per_source: np.ndarray
    """Length n: the share of the total carried by source u. Non-negative, and
    the shares sum to total."""

    @property
    def share(self) -> np.ndarray:
        """The share as a fraction."""
        return self.per_source / self.total if self.total > 0 else np.zeros_like(self.per_source)

    def top_sources(self, count: int = 10) -> np.ndarray:
        """The indices of the largest-contributing sources, descending."""
        return np.argsort(-self.per_source)[:count]


def variance_attribution(
    graph: UncertainGraph,
    et: ExpectedTransition,
    y0: np.ndarray,
    direction: np.ndarray,
    alpha: float = 0.85,
    z: np.ndarray | None = None,
) -> Attribution:
    """Decompose the variance of a linear quantity over source nodes, exactly.

    The quantity is w^T y for an arbitrary coefficient vector w, and

        Var(w^T y) ~= alpha^2 x^T M x,    x = (I - alpha Pbar)^{-1} w
        = sum_u alpha^2 y0(u)^2 (x^T Cov_u x)

    Each term is non-negative because Cov_u is a covariance matrix, and the
    terms sum to the whole variance. Each says how much the uncertainty in
    source u's out-links moves the quantity.

    With w = e_v this is Var(y_v); with w = e_a - e_b it is the variance of the
    rank margin y_a - y_b, which says directly which uncertain links blur a
    given top-k boundary.

    Args:
        graph: the uncertain graph.
        et: the expected transition matrix.
        y0: the p-average scores.
        direction: the x above, as returned by contribution_vectors.
        alpha: the damping factor.
        z: the personalization vector.
    Returns:
        Attribution。
    """
    z = _resolve_z(z, graph.n)
    x = np.asarray(direction, dtype=np.float64)
    tgt = et.B.indices
    delta = et.delta
    w = y0 * y0
    zx = float(z @ x)

    pair = pair_moment_edges(graph, x[tgt])  # (A_u x)_i
    # Aggregate x^T A_u x per node. np.add.reduceat runs out of range on a node
    # of out-degree zero (indptr[u] == m), so use differences of a cumulative
    # sum instead.
    csum = np.zeros(graph.m + 1, dtype=np.float64)
    np.cumsum(x[tgt] * pair, out=csum[1:])
    indptr = et.B.indptr
    quad = csum[indptr[1:]] - csum[indptr[:-1]]

    s = et.B @ x + delta * zx  # Pbar(u,.) . x
    per_source = alpha * alpha * w * np.maximum(quad + delta * zx * zx - s * s, 0.0)
    return Attribution(total=float(per_source.sum()), per_source=per_source)


def edge_acquisition_scores(
    graph: UncertainGraph,
    et: ExpectedTransition,
    y0: np.ndarray,
    direction: np.ndarray,
    alpha: float = 0.85,
    z: np.ndarray | None = None,
) -> np.ndarray:
    """The expected reduction in surrogate variance from observing one link.

    The quantity is the variance of w^T y with x = (I - alpha Pbar)^{-1} w.
    Writing Z_u = x^T p_X(u,.) for the contribution of source u's row, the law
    of total variance makes the expected reduction the variance of the
    conditional mean:

        Var(Z_u) - E_{xi_e}[ Var(Z_u | xi_e) ] = Var( E[Z_u | xi_e] )
                                                = q_e (1 - q_e) (m_e^1 - m_e^0)^2

    so no second moment is needed: the **conditional first moments alone**
    suffice. With g the connection-count distribution excluding link
    e = (u,v) and h[k] = E[ sum_{w != v} xi_w x_w * 1(K' = k) ],

        m_e^1 = x_v * sum_k g[k]/(1+k) + sum_k h[k]/(1+k),
        m_e^0 = sum_{k>=1} h[k]/k + (z . x) * g[0]

    where xi_e = 1 makes u non-dangling and xi_e = 0 with K' = 0 makes it
    dangling. Both g and h come from the same deconvolution as pair_moment in
    O(D_u) each, so all links cost O(sum_u D_u^2).

    Non-negativity follows from the law of total variance. The per-link values
    are **not additive**, however: the effect of observing several links at once
    is not the sum of the individual values, and must not be used as such.

    Args:
        graph: the uncertain graph.
        et: the expected transition matrix.
        y0: the p-average scores.
        direction: x = (I - alpha Pbar)^{-1} w。
        alpha: the damping factor.
        z: the personalization vector.
    Returns:
        Length m, non-negative. Zero on certain links (q = 0 or 1).
    """
    from .expected_transition import poisson_binomial_pmf
    from .pair_moment import _deconvolve_pmf, _deconvolve_weighted

    z = _resolve_z(z, graph.n)
    x = np.asarray(direction, dtype=np.float64)
    zx = float(z @ x)
    out = np.zeros(graph.m, dtype=np.float64)
    degrees = graph.out_degrees()

    for u in range(graph.n):
        d = int(degrees[u])
        if d == 0:
            continue
        s = int(graph.indptr[u])
        q = graph.probs[s : s + d]
        c = x[graph.indices[s : s + d]]
        phi = poisson_binomial_pmf(q)

        psi = np.zeros(d + 1, dtype=np.float64)
        loo = []
        for j in range(d):
            g_j = _deconvolve_pmf(phi, float(q[j]))
            loo.append(g_j)
            psi[1:] += c[j] * q[j] * g_j

        inv1 = 1.0 / np.arange(1, d + 1, dtype=np.float64)  # 1/(1+k), k=0..d-1
        inv0 = np.zeros(d, dtype=np.float64)
        inv0[1:] = 1.0 / np.arange(1, d, dtype=np.float64)  # 1/k for k>=1, 0 at k=0
        weight = alpha * alpha * y0[u] * y0[u]

        for i in range(d):
            qi = float(q[i])
            if qi <= 0.0 or qi >= 1.0:
                continue
            g_i = loo[i]
            h_i = _deconvolve_weighted(psi, g_i, qi, float(c[i]))
            m1 = float(c[i]) * float(g_i @ inv1) + float(h_i @ inv1)
            m0 = float(h_i @ inv0) + zx * float(g_i[0])
            out[s + i] = weight * qi * (1.0 - qi) * (m1 - m0) ** 2
    return out
