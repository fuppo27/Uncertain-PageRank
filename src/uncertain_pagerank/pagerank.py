"""PageRank, Eq. (1) and (2) of the conference paper.

The iteration is

    y_t^T = alpha y_{t-1}^T P + (1 - alpha) z^T,

and it stops at ||y_t - y_{t-1}||_1 < tol.

For the expected transition matrix Pbar = B + delta z^T no dense matrix is
needed: the iteration uses

    Pbar^T y = B^T y + z (delta . y)

at O(|E| + |V|) per iteration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .expected_transition import ExpectedTransition


@dataclass(frozen=True)
class PageRankResult:
    """The result of a PageRank computation."""

    scores: np.ndarray
    iterations: int
    residual: float
    converged: bool


def uniform_vector(n: int) -> np.ndarray:
    """The uniform personalization vector z."""
    return np.full(n, 1.0 / n, dtype=np.float64)


def _resolve_z(z: np.ndarray | None, n: int) -> np.ndarray:
    if z is None:
        return uniform_vector(n)
    z = np.asarray(z, dtype=np.float64)
    if z.shape != (n,):
        raise ValueError("len(z) does not match the node count")
    if not np.isclose(z.sum(), 1.0):
        raise ValueError("z must be a probability distribution")
    return z


def pagerank(
    et: ExpectedTransition,
    alpha: float = 0.85,
    z: np.ndarray | None = None,
    tol: float = 1e-12,
    max_iter: int = 10_000,
) -> PageRankResult:
    """One PageRank computation on the sparse expected transition matrix (p-avg)."""
    n = et.n
    z = _resolve_z(z, n)
    bt = et.B.T.tocsr()
    delta = et.delta

    y = z.copy()
    residual, iterations, converged = np.inf, 0, False
    for iterations in range(1, max_iter + 1):
        y_next = alpha * (bt @ y + z * float(delta @ y)) + (1.0 - alpha) * z
        residual = float(np.abs(y_next - y).sum())
        y = y_next
        if residual < tol:
            converged = True
            break
    return PageRankResult(scores=y, iterations=iterations, residual=residual, converged=converged)


def pagerank_dense(
    p: np.ndarray,
    alpha: float = 0.85,
    z: np.ndarray | None = None,
    tol: float = 1e-12,
    max_iter: int = 10_000,
) -> PageRankResult:
    """Iterate PageRank on a dense row-stochastic P. For small-scale checks."""
    n = p.shape[0]
    z = _resolve_z(z, n)
    pt = np.ascontiguousarray(p.T)

    y = z.copy()
    residual, iterations, converged = np.inf, 0, False
    for iterations in range(1, max_iter + 1):
        y_next = alpha * (pt @ y) + (1.0 - alpha) * z
        residual = float(np.abs(y_next - y).sum())
        y = y_next
        if residual < tol:
            converged = True
            break
    return PageRankResult(scores=y, iterations=iterations, residual=residual, converged=converged)


def pagerank_solve(p: np.ndarray, alpha: float = 0.85, z: np.ndarray | None = None) -> np.ndarray:
    """Solve the linear system (I - alpha P^T) y = (1 - alpha) z directly.

    No iteration error enters, which is what the exact s-average from exhaustive
    enumeration needs as its reference. P is (n, n) or (batch, n, n).
    """
    n = p.shape[-1]
    z = _resolve_z(z, n)
    pt = np.swapaxes(p, -1, -2)
    a = np.eye(n) - alpha * pt
    b = np.broadcast_to((1.0 - alpha) * z, p.shape[:-1])
    return np.linalg.solve(a, b[..., None])[..., 0]
