"""Solve the PageRank of many possible worlds at once, without forming matrices.

Every world is a subset of the same link set, so rather than building a
transition matrix per world we compute `P_X^T y` directly from a connection
mask of shape `(S, m)`.

    (P_X^T y)_v = sum_{e: tgt(e)=v} mask_e * y_{src(e)} / deg_{src(e)}(X)
                  + z_v * sum_{u: deg_u(X)=0} y_u

Building a dense `(S, n, n)` breaks down as n grows, so this is the route used
at medium scale. The iteration and the handling of dangling nodes match
`pagerank.pagerank` exactly.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .graph import UncertainGraph
from .pagerank import _resolve_z


def batch_world_scores(
    graph: UncertainGraph,
    masks: np.ndarray,
    alpha: float = 0.85,
    z: np.ndarray | None = None,
    tol: float = 1e-12,
    max_iter: int = 10_000,
    chunk: int = 512,
) -> np.ndarray:
    """The PageRank of the world given by each row of the connection mask.

    Convergence is tested on the largest residual over all worlds rather than
    per world. Each is a contraction towards its own fixed point, so a world
    that converged early only converges further.

    Args:
        masks: (S, m) of 0/1, the connection state of each link.
        chunk: how many worlds to solve at once.
    Returns:
        (S, n)。
    """
    n, m = graph.n, graph.m
    z = _resolve_z(z, n)
    masks = np.asarray(masks, dtype=np.float64)
    if masks.ndim != 2 or masks.shape[1] != m:
        raise ValueError("masks must have shape (S, m)")

    src = np.repeat(np.arange(n), graph.out_degrees())
    # Scatter to the targets with a sparse product: nnz = m, so this stays
    # cheap even for large n
    scatter = sp.csr_matrix(
        (np.ones(m), (graph.indices, np.arange(m))), shape=(n, m)
    )
    indptr = graph.indptr
    out = np.empty((masks.shape[0], n))

    for start in range(0, masks.shape[0], chunk):
        mk = masks[start : start + chunk]
        s_n = mk.shape[0]
        csum = np.zeros((s_n, m + 1))
        np.cumsum(mk, axis=1, out=csum[:, 1:])
        deg = csum[:, indptr[1:]] - csum[:, indptr[:-1]]  # (s_n, n)
        live = deg > 0
        inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=live)

        y = np.broadcast_to(z, (s_n, n)).copy()
        for _ in range(max_iter):
            contrib = mk * (y * inv)[:, src]  # (s_n, m)
            walk = (scatter @ contrib.T).T  # (s_n, n)
            dang = (y * ~live).sum(axis=1)[:, None]
            nxt = alpha * (walk + dang * z[None, :]) + (1.0 - alpha) * z[None, :]
            resid = np.abs(nxt - y).sum(axis=1)
            y = nxt
            if resid.max() < tol:
                break
        out[start : start + s_n] = y
    return out


def sample_masks(
    graph: UncertainGraph, n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw a connection mask by independent Bernoulli sampling.

    A link with `q_e = 1` is always connected and one with `q_e = 0` never is.
    """
    return (rng.random((n_samples, graph.m)) < graph.probs[None, :]).astype(np.float64)
