"""A second-order approximation to the gap between p-avg and s-avg.

The PageRank fixed point gives, exactly,

    mu - y0 = alpha R E[ Delta_X^T (y_X - y0) ],   R = (I - alpha Pbar^T)^{-1}

and substituting y_X - y0 = alpha (I - alpha P_X^T)^{-1} Delta_X^T y0 yields

    mu - y0 = alpha^2 R E[ Delta_X^T (I - alpha P_X^T)^{-1} Delta_X^T y0 ]

Replacing (I - alpha P_X^T)^{-1} by R gives the second-order approximation

    mu - y0 ~= alpha^2 R b,
    b_v = sum_u y0(u) sum_w R(u,w) Cov(p_X(u,v), p_X(u,w))

Because the links of different sources are independent, the covariance closes
row by row. What is needed is, for two links out of the same source,

    E[ x_v x_w / (D_u^X)^2 ]
      = q_v q_w sum_k (1/k^2) P(connection count = k | x_v = x_w = 1)

which follows from shifting the leave-one-out distribution by one more link
(leave-two-out). The direction of the deconvolution is chosen by q so that the
error amplification stays at most one.

This costs O(sum_u D_u^3 + n^3) and holds R densely, so it is for small graphs
only: first measure how well the approximation does against the exact value,
and only then consider a form that scales.
"""

from __future__ import annotations

import numpy as np

from .expected_transition import poisson_binomial_pmf
from .graph import UncertainGraph
from .pagerank import _resolve_z


def _deconvolve(phi: np.ndarray, qi: float) -> np.ndarray:
    """Remove one link with probability qi from the distribution phi.

    Solves phi[k] = qi * g[k-1] + (1-qi) * g[k] for g, running forward when
    qi < 1/2 (amplification qi/(1-qi) < 1) and backward otherwise
    (amplification (1-qi)/qi <= 1).

    Args:
        phi: length L+1, the distribution.
        qi: the connection probability of the link removed.
    Returns:
        Length L, the distribution.
    """
    length = phi.shape[0] - 1
    g = np.empty(length, dtype=np.float64)
    if qi < 0.5:
        den = 1.0 - qi
        g[0] = phi[0] / den
        for k in range(1, length):
            g[k] = (phi[k] - qi * g[k - 1]) / den
    else:
        g[length - 1] = phi[length] / qi
        for k in range(length - 2, -1, -1):
            g[k] = (phi[k + 1] - (1.0 - qi) * g[k + 1]) / qi
    return g


def row_second_moment_matrix(q: np.ndarray) -> np.ndarray:
    """A[v,w] = E[x_v x_w / (D^X)^2] for every pair of links of one node.

    The diagonal is E[x_v / (D^X)^2], since x_v^2 = x_v.

    Args:
        q: length D, the connection probabilities.
    Returns:
        A (D, D) symmetric matrix.
    """
    d = q.shape[0]
    a = np.zeros((d, d), dtype=np.float64)
    phi = poisson_binomial_pmf(q)
    w1 = 1.0 / np.arange(1, d + 1, dtype=np.float64) ** 2  # k = j+1
    loo = [_deconvolve(phi, float(q[i])) for i in range(d)]
    for v in range(d):
        a[v, v] = q[v] * float(loo[v] @ w1)
    if d < 2:
        return a
    w2 = 1.0 / np.arange(2, d + 1, dtype=np.float64) ** 2  # k = j+2
    for v in range(d):
        for w in range(v + 1, d):
            h = _deconvolve(loo[v], float(q[w]))  # length D-1
            value = float(q[v]) * float(q[w]) * float(h[: d - 1] @ w2)
            a[v, w] = a[w, v] = value
    return a


def bias_estimate(
    graph: UncertainGraph,
    y0: np.ndarray,
    alpha: float,
    z: np.ndarray | None = None,
    delta: np.ndarray | None = None,
    b_rows: np.ndarray | None = None,
) -> np.ndarray:
    """A second-order approximation to mu - y0. Small graphs only, O(sum D^3 + n^3).

    Args:
        graph: the uncertain graph.
        y0: the p-average scores.
        alpha: the damping factor.
        z: the personalization vector.
        delta: the dangling probabilities, computed when not given.
        b_rows: the non-dangling part of the expected transition, computed when
            not given.
    Returns:
        Length n, the estimated bias: y0 plus this is the corrected estimate.
    """
    from .expected_transition import expected_transition

    n = graph.n
    z = _resolve_z(z, n)
    if delta is None or b_rows is None:
        et = expected_transition(graph)
        delta, b_rows = et.delta, et.B.data

    p_bar = np.zeros((n, n))
    src = np.repeat(np.arange(n), graph.out_degrees())
    p_bar[src, graph.indices] = b_rows
    p_bar += np.outer(delta, z)

    r_mat = np.linalg.inv(np.eye(n) - alpha * p_bar.T)

    b = np.zeros(n)
    for u in range(n):
        s, e = int(graph.indptr[u]), int(graph.indptr[u + 1])
        targets = graph.indices[s:e]
        r_u = r_mat[u]  # R(u, :)
        mean_u = p_bar[u]
        c_r = z * (delta[u] * float(z @ r_u)) - mean_u * float(mean_u @ r_u)
        if targets.size:
            a = row_second_moment_matrix(graph.probs[s:e])
            c_r[targets] += a @ r_u[targets]
        b += y0[u] * c_r

    return alpha * alpha * (r_mat @ b)


def _solve_resolvent(
    matvec, v: np.ndarray, alpha: float, tol: float = 1e-12, max_iter: int = 10_000
) -> np.ndarray:
    """Solve (I - alpha A^T)^{-1} v by its Neumann series.

    Iterates w <- v + alpha A^T w. The rate is alpha, so this costs the same as
    one PageRank computation.

    Args:
        matvec: a function computing x -> A^T x.
        v: the right-hand side.
        alpha: the damping factor.
        tol: the threshold on ||w_t - w_{t-1}||_1.
    """
    w = v.copy()
    for _ in range(max_iter):
        nxt = v + alpha * matvec(w)
        if float(np.abs(nxt - w).sum()) < tol:
            return nxt
        w = nxt
    return w


def bias_estimate_sampled(
    graph: UncertainGraph,
    y0: np.ndarray,
    alpha: float,
    z: np.ndarray | None = None,
    n_samples: int = 32,
    seed: int = 0,
    tol: float = 1e-12,
    expected: "object | None" = None,
) -> np.ndarray:
    """Estimate the second-order term of mu - y0 from samples, for large graphs.

    The exact identity

        mu - y0 = alpha^2 R E[ Delta_X^T (I - alpha P_X^T)^{-1} Delta_X^T y0 ]

    is estimated by sampling worlds. Unlike bias_estimate, R_X is not replaced
    by R, so the estimate converges to the true value as the sample size grows.

    The first-order term Delta_X^T y0 has expectation zero, so the integrand has
    a far smaller variance than the PageRank scores themselves, and reaches a
    given accuracy in fewer samples than estimating s-avg by plain sampling.

    One sample costs a sparse matrix-vector product and one solve, about the
    same as one PageRank computation.

    Args:
        graph: the uncertain graph.
        y0: the p-average scores.
        alpha: the damping factor.
        z: the personalization vector.
        n_samples: the number of samples.
        seed: the random seed.
        tol: the convergence threshold of the solve.
        expected: a precomputed ExpectedTransition, computed when not given.
    Returns:
        Length n, the estimated bias.
    """
    from .expected_transition import expected_transition
    from .sampling import WorldSampler

    n = graph.n
    z = _resolve_z(z, n)
    et = expected_transition(graph) if expected is None else expected
    bar_t = et.B.T.tocsr()
    delta = et.delta

    def bar_matvec(x: np.ndarray) -> np.ndarray:
        """Pbar^T x = B^T x + z (delta . x)."""
        return bar_t @ x + z * float(delta @ x)

    sampler = WorldSampler(graph, np.random.default_rng(seed))
    bar_y0 = bar_matvec(y0)

    acc = np.zeros(n)
    for _ in range(n_samples):
        world = sampler.transition(sampler.sample_mask())
        world_t = world.B.T.tocsr()
        world_delta = world.delta

        def world_matvec(x: np.ndarray, _t=world_t, _d=world_delta) -> np.ndarray:
            return _t @ x + z * float(_d @ x)

        d_y0 = world_matvec(y0) - bar_y0  # Delta_X^T y0
        w = _solve_resolvent(world_matvec, d_y0, alpha, tol=tol)
        acc += world_matvec(w) - bar_matvec(w)  # Delta_X^T w

    inner = acc / n_samples
    return alpha * alpha * _solve_resolvent(bar_matvec, inner, alpha, tol=tol)
