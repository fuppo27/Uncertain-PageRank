"""Exact computation of the expected transition matrix <P_X>_X, Eq. (6)-(10).

The expectation of the transition probabilities over all possible worlds,

    <p_X(u,v)>_X = E[ x_uv / D_u^X ]   (when D_u^X > 0)

is obtained exactly in O(sum_u D_u^2) without enumerating any world.

The matrix is held in the decomposed form

    Pbar = B + delta z^T

where B is sparse, nonzero only on the original link set,
delta_u = P(D_u^X = 0) = prod_v (1 - q_uv) is the probability that u is
dangling, and z is the jump distribution of a dangling node. No dense
|V| x |V| matrix is ever formed.

On numerical stability:
    Eq. (9) of the conference paper recovers the distribution without link i
    from gamma by a forward recurrence, whose error amplification is
    q_i/(1-q_i) per step and so becomes unstable as q_i -> 1. Here the
    recurrence runs forward when q_i < 1/2 and backward otherwise, keeping the
    amplification at most one. The two are algebraically equivalent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .graph import UncertainGraph


@dataclass(frozen=True)
class ExpectedTransition:
    """The sparse form Pbar = B + delta z^T of the expected transition matrix.

    Attributes:
        B: an (n, n) sparse matrix, nonzero only on the original links, with
            B[u,v] = <x_uv / D_u^X>.
        delta: length n, delta[u] = P(D_u^X = 0). One at a node of out-degree 0.
        M2: the non-dangling part of the second moment, present only when
            with_second_moment=True.
    """

    B: sp.csr_matrix
    delta: np.ndarray
    M2: sp.csr_matrix | None = None
    """An (n, n) sparse matrix, M2[u,v] = E[x_uv / (D_u^X)^2]: the non-dangling
    part of the second moment of the transition probabilities, so that
    E[p_X(u,v)^2] = M2[u,v] + z(v)^2 delta_u."""

    @property
    def n(self) -> int:
        return self.B.shape[0]

    def to_dense(self, z: np.ndarray) -> np.ndarray:
        """Return Pbar densely. For small-scale checks only."""
        return np.asarray(self.B.todense()) + np.outer(self.delta, z)

    def row_sum_error(self, ) -> float:
        """The largest deviation of a row sum (sum_v B[u,v] + delta_u) from one."""
        s = np.asarray(self.B.sum(axis=1)).ravel() + self.delta
        return float(np.abs(s - 1.0).max())


def poisson_binomial_pmf(q: np.ndarray) -> np.ndarray:
    """The Poisson-binomial mass function phi[k] = P(k links connected).

    Eq. (7) of the conference paper, advanced to j = D. Costs O(D^2).

    Args:
        q: length D, the connection probabilities.
    Returns:
        Length D+1, the probability mass function.
    """
    d = q.shape[0]
    phi = np.zeros(d + 1, dtype=np.float64)
    phi[0] = 1.0
    for j in range(d):
        qj = q[j]
        phi[1 : j + 2] = qj * phi[0 : j + 1] + (1.0 - qj) * phi[1 : j + 2]
        phi[0] *= 1.0 - qj
    return phi


def _expected_row_dp(q: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """<p_X(u,i)> for every out-link i of one node at once.

    With g^(i) the distribution of the connection count excluding link i,

        <p_X(u,i)> = q_i * sum_{k=0}^{D-1} g^(i)[k] / (k+1)

    which is Eq. (10) of the conference paper with psi_{i,k} = q_i g^(i)[k-1].
    Each g^(i) comes from deconvolving one link out of phi.

    Args:
        q: length D, the connection probabilities.
        phi: poisson_binomial_pmf(q).
    Returns:
        Length D, the expected transition probabilities.
    """
    d = q.shape[0]
    w = 1.0 / np.arange(1, d + 1, dtype=np.float64)  # w[k] = 1/(k+1)
    acc = np.zeros(d, dtype=np.float64)

    fwd = np.flatnonzero(q < 0.5)
    bwd = np.flatnonzero(q >= 0.5)

    # Forward: g[0] = phi[0]/(1-q), g[k] = (phi[k] - q g[k-1])/(1-q).
    # Amplification q/(1-q) < 1.
    if fwd.size:
        qf = q[fwd]
        one_minus = 1.0 - qf
        g = phi[0] / one_minus
        acc[fwd] = g * w[0]
        for k in range(1, d):
            g = (phi[k] - qf * g) / one_minus
            acc[fwd] += g * w[k]

    # Backward: g[D-1] = phi[D]/q, g[k] = (phi[k+1] - (1-q) g[k+1])/q.
    # Amplification (1-q)/q <= 1.
    if bwd.size:
        qb = q[bwd]
        one_minus = 1.0 - qb
        g = phi[d] / qb
        acc[bwd] = g * w[d - 1]
        for k in range(d - 2, -1, -1):
            g = (phi[k + 1] - one_minus * g) / qb
            acc[bwd] += g * w[k]

    return np.maximum(q * acc, 0.0)


def _expected_row_uniform(qv: float, d: int) -> np.ndarray:
    """The closed form for a uniform connection probability, Eq. (6)."""
    if d == 0:
        return np.zeros(0, dtype=np.float64)
    return np.full(d, (1.0 - (1.0 - qv) ** d) / d, dtype=np.float64)


def expected_transition(
    graph: UncertainGraph,
    method: str = "auto",
    max_batch_elements: int = 1 << 24,
    with_second_moment: bool = False,
) -> ExpectedTransition:
    """Compute the expected transition matrix exactly.

    Args:
        graph: the uncertain graph.
        method: "auto" | "uniform" | "dp".
            "uniform" is the closed form of Eq. (6), valid when every link has
            the same connection probability. "dp" is the dynamic program of
            Eq. (7)-(10) for general non-uniform q_e. "auto" picks "uniform"
            when all q_e agree and "dp" otherwise.
        max_batch_elements: the most array elements to process at once when the
            dynamic program is batched by out-degree. Caps the memory use.
        with_second_moment: whether to also compute the second moment
            E[x_uv/(D_u^X)^2], which the variance Var(p_X(u,v)) needs.
    Returns:
        An ExpectedTransition: the sparse B and the dangling probabilities
        delta.
    """
    n, indptr, probs = graph.n, graph.indptr, graph.probs
    degrees = graph.out_degrees()

    if method == "auto":
        uniform = graph.m == 0 or bool(np.all(probs == probs[0]))
        method = "uniform" if uniform else "dp"
    if method == "uniform" and graph.m and not bool(np.all(probs == probs[0])):
        raise ValueError("method='uniform' needs the same connection probability "
                         "on every link")
    if method not in ("uniform", "dp"):
        raise ValueError(f"unknown method: {method}")

    data = np.zeros(graph.m, dtype=np.float64)
    data2 = np.zeros(graph.m, dtype=np.float64) if with_second_moment else None
    delta = np.ones(n, dtype=np.float64)  # a node with no out-link is always dangling

    if method == "uniform" and graph.m:
        # The second moment has no closed form even for uniform q, so use the DP
        if with_second_moment:
            _fill_dp_batched(graph, data, delta, max_batch_elements, data2)
        else:
            qv = float(probs[0])
            nonzero = degrees > 0
            d_nz = degrees[nonzero].astype(np.float64)
            delta[nonzero] = (1.0 - qv) ** d_nz
            data[:] = np.repeat((1.0 - (1.0 - qv) ** d_nz) / d_nz, degrees[nonzero])
    elif method == "dp":
        _fill_dp_batched(graph, data, delta, max_batch_elements, data2)

    b = sp.csr_matrix((data, graph.indices.copy(), indptr.copy()), shape=(n, n))
    m2 = (
        sp.csr_matrix((data2, graph.indices.copy(), indptr.copy()), shape=(n, n))
        if data2 is not None
        else None
    )
    return ExpectedTransition(B=b, delta=delta, M2=m2)


def expected_transition_loo_reference(graph: UncertainGraph) -> ExpectedTransition:
    """Reference implementation: recompute each leave-one-out distribution from
    scratch by the dynamic program, at O(sum_u D_u^3).

    Numerically stable, since it uses no deconvolution, but slow. Used to
    cross-check in the tests.
    """
    n, indptr, probs = graph.n, graph.indptr, graph.probs
    degrees = graph.out_degrees()
    data = np.zeros(graph.m, dtype=np.float64)
    delta = np.ones(n, dtype=np.float64)

    for u in range(n):
        d = int(degrees[u])
        if d == 0:
            continue
        s = int(indptr[u])
        q = probs[s : s + d]
        delta[u] = poisson_binomial_pmf(q)[0]
        w = 1.0 / np.arange(1, d + 1, dtype=np.float64)
        for i in range(d):
            g = poisson_binomial_pmf(np.delete(q, i))
            data[s + i] = q[i] * float(g[: d] @ w)

    b = sp.csr_matrix((data, graph.indices.copy(), indptr.copy()), shape=(n, n))
    return ExpectedTransition(B=b, delta=delta)


def _degree_buckets(degrees: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """Group nodes of similar out-degree into geometric buckets (2^(L-1), 2^L].

    Grouping by exact out-degree needs one numpy call per distinct out-degree.
    The number of dynamic-program iterations is set by the largest out-degree in
    a bucket, so bucketing brings the call count down to O(max_u D_u). Degrees
    within a bucket differ by at most a factor of two, so the padding wastes at
    most a factor of two as well.

    Returns:
        A list of (largest out-degree in the bucket, node indices). Nodes of
        out-degree 0 are excluded.
    """
    nz = np.flatnonzero(degrees > 0)
    if nz.size == 0:
        return []
    d = degrees[nz]
    level = np.ceil(np.log2(d)).astype(np.int64)
    order = np.argsort(level, kind="stable")
    nz, level = nz[order], level[order]
    cuts = np.flatnonzero(np.diff(level)) + 1
    return [(int(degrees[nodes].max()), nodes) for nodes in np.split(nz, cuts)]


def _expected_rows_batched(
    q: np.ndarray, second_moment: bool = False
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Expected transition probabilities for G nodes of the same out-degree D.

    The same arithmetic as _expected_row_dp, on (G, D) arrays. The number of
    numpy calls becomes O(D) regardless of the node count, which is much faster
    on graphs with many nodes.

    The second moment E[x_ui / (D_u^X)^2] follows from the same leave-one-out
    distributions by changing the weight from 1/k to 1/k^2.

    Args:
        q: (G, D) connection probabilities.
        second_moment: whether to also compute the second moment.
    Returns:
        ((G, D) expected transition probabilities, (G, D) second moments or
        None, (G,) dangling probabilities).
    """
    g, d = q.shape
    phi = np.zeros((g, d + 1), dtype=np.float64)
    phi[:, 0] = 1.0
    for j in range(d):
        qj = q[:, j : j + 1]
        phi[:, 1 : j + 2] = qj * phi[:, 0 : j + 1] + (1.0 - qj) * phi[:, 1 : j + 2]
        phi[:, 0] *= 1.0 - qj[:, 0]

    w = 1.0 / np.arange(1, d + 1, dtype=np.float64)
    w2 = w * w
    one_minus = 1.0 - q
    use_fwd = q < 0.5
    # The unselected recurrence is computed too, so its denominator is replaced
    # by one to avoid dividing by zero; that value is discarded at the end.
    # Neither recurrence diverges, since |g_k| <= k+1 bounds both.
    den_f = np.where(use_fwd, one_minus, 1.0)
    den_b = np.where(use_fwd, 1.0, q)

    acc_f = np.zeros((g, d), dtype=np.float64)
    acc_f2 = np.zeros((g, d), dtype=np.float64) if second_moment else None
    gk = phi[:, 0:1] / den_f
    acc_f += gk * w[0]
    if acc_f2 is not None:
        acc_f2 += gk * w2[0]
    for k in range(1, d):
        gk = (phi[:, k : k + 1] - q * gk) / den_f
        acc_f += gk * w[k]
        if acc_f2 is not None:
            acc_f2 += gk * w2[k]

    acc_b = np.zeros((g, d), dtype=np.float64)
    acc_b2 = np.zeros((g, d), dtype=np.float64) if second_moment else None
    gk = phi[:, d : d + 1] / den_b
    acc_b += gk * w[d - 1]
    if acc_b2 is not None:
        acc_b2 += gk * w2[d - 1]
    for k in range(d - 2, -1, -1):
        gk = (phi[:, k + 1 : k + 2] - one_minus * gk) / den_b
        acc_b += gk * w[k]
        if acc_b2 is not None:
            acc_b2 += gk * w2[k]

    values = np.maximum(q * np.where(use_fwd, acc_f, acc_b), 0.0)
    values2 = None
    if acc_f2 is not None and acc_b2 is not None:
        values2 = np.maximum(q * np.where(use_fwd, acc_f2, acc_b2), 0.0)
    return values, values2, phi[:, 0]


def _fill_dp_batched(
    graph: UncertainGraph,
    data: np.ndarray,
    delta: np.ndarray,
    max_batch_elements: int,
    data2: np.ndarray | None = None,
) -> None:
    """Run the dynamic program over buckets of similar out-degree.

    Pads with q = 0 links up to the largest out-degree in the bucket. A link
    with q = 0 does not change the connection-count distribution, so the result
    is unchanged.
    """
    degrees = graph.out_degrees()
    probs, indptr = graph.probs, graph.indptr
    last = max(graph.m - 1, 0)
    for d_max, nodes in _degree_buckets(degrees):
        cols = np.arange(d_max, dtype=np.int64)
        chunk = max(1, min(nodes.shape[0], max_batch_elements // d_max))
        for start in range(0, nodes.shape[0], chunk):
            part = nodes[start : start + chunk]
            idx = indptr[part][:, None] + cols[None, :]
            valid = cols[None, :] < degrees[part][:, None]
            # Out-of-range indices are read but discarded; clamp them anyway
            q = np.where(valid, probs[np.minimum(idx, last)], 0.0)
            values, values2, delta_part = _expected_rows_batched(q, second_moment=data2 is not None)
            data[idx[valid]] = values[valid]
            if data2 is not None and values2 is not None:
                data2[idx[valid]] = values2[valid]
            delta[part] = delta_part


def expected_transition_dp_pernode(graph: UncertainGraph) -> ExpectedTransition:
    """The dynamic program node by node. Used to cross-check the batched form."""
    n, indptr, probs = graph.n, graph.indptr, graph.probs
    degrees = graph.out_degrees()
    data = np.zeros(graph.m, dtype=np.float64)
    delta = np.ones(n, dtype=np.float64)
    for u in range(n):
        d = int(degrees[u])
        if d == 0:
            continue
        s = int(indptr[u])
        q = probs[s : s + d]
        phi = poisson_binomial_pmf(q)
        data[s : s + d] = _expected_row_dp(q, phi)
        delta[u] = phi[0]
    b = sp.csr_matrix((data, graph.indices.copy(), indptr.copy()), shape=(n, n))
    return ExpectedTransition(B=b, delta=delta)


def _expected_row_eq9_naive(q: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Eq. (9) and (10) of the conference paper, as the forward recurrence.

        psi_{i,k} = q_i / (1 - q_i) * (phi_{k-1} - psi_{i,k-1}),
        <p_X(u,i)> = sum_k psi_{i,k} / k.

    The error amplification is q_i/(1-q_i) per step, so rounding error grows
    exponentially when the out-degree D is large and q_i is close to one. Kept
    for the numerical-stability comparison; use _expected_row_dp in practice.
    """
    d = q.shape[0]
    out = np.zeros(d, dtype=np.float64)
    ratio = q / np.where(q >= 1.0, np.nan, 1.0 - q)
    for i in range(d):
        psi, acc = 0.0, 0.0
        for k in range(1, d + 1):
            psi = ratio[i] * (phi[k - 1] - psi)
            acc += psi / k
        out[i] = acc
    return out


def expected_transition_naive_eq9(graph: UncertainGraph) -> ExpectedTransition:
    """Eq. (9) as the forward recurrence. For the stability comparison."""
    n, indptr, probs = graph.n, graph.indptr, graph.probs
    degrees = graph.out_degrees()
    data = np.zeros(graph.m, dtype=np.float64)
    delta = np.ones(n, dtype=np.float64)
    for u in range(n):
        d = int(degrees[u])
        if d == 0:
            continue
        s = int(indptr[u])
        q = probs[s : s + d]
        phi = poisson_binomial_pmf(q)
        delta[u] = phi[0]
        data[s : s + d] = _expected_row_eq9_naive(q, phi)
    b = sp.csr_matrix((data, graph.indices.copy(), indptr.copy()), shape=(n, n))
    return ExpectedTransition(B=b, delta=delta)


def deterministic_transition(graph: UncertainGraph) -> ExpectedTransition:
    """The ordinary transition matrix, for when every link is certain (q_e = 1)."""
    return expected_transition(graph.with_probs(np.ones(graph.m)), method="uniform")


def transition_moment(
    graph: UncertainGraph, order: int = 1, max_batch_elements: int = 1 << 24
) -> np.ndarray:
    """E[x_uv / (D_u^X)^order] per link, exactly.

    order=1 is the non-dangling part B of the expected transition matrix,
    order=2 the second moment, and order=3 and above come from the same dynamic
    program: only the weight on the leave-one-out distribution g changes, to
    1/(k+1)^order.

    The third moment is used to bound the residual left after the second-order
    bias correction.

    Args:
        graph: the uncertain graph.
        order: the order of the moment, at least one.
        max_batch_elements: the most array elements to process at once.
    Returns:
        Length m.
    """
    if order < 1:
        raise ValueError("order must be at least one")
    degrees = graph.out_degrees()
    out = np.zeros(graph.m, dtype=np.float64)
    last = max(graph.m - 1, 0)
    for d_max, nodes in _degree_buckets(degrees):
        cols = np.arange(d_max, dtype=np.int64)
        chunk = max(1, min(nodes.shape[0], max_batch_elements // d_max))
        w = 1.0 / np.arange(1, d_max + 1, dtype=np.float64) ** order
        for start in range(0, nodes.shape[0], chunk):
            part = nodes[start : start + chunk]
            idx = graph.indptr[part][:, None] + cols[None, :]
            valid = cols[None, :] < degrees[part][:, None]
            q = np.where(valid, graph.probs[np.minimum(idx, last)], 0.0)
            g_n, d = q.shape
            phi = np.zeros((g_n, d + 1))
            phi[:, 0] = 1.0
            for j in range(d):
                qj = q[:, j : j + 1]
                phi[:, 1 : j + 2] = qj * phi[:, 0 : j + 1] + (1.0 - qj) * phi[:, 1 : j + 2]
                phi[:, 0] *= 1.0 - qj[:, 0]
            one_minus = 1.0 - q
            use_fwd = q < 0.5
            den_f = np.where(use_fwd, one_minus, 1.0)
            den_b = np.where(use_fwd, 1.0, q)
            acc_f = np.zeros((g_n, d))
            gk = phi[:, 0:1] / den_f
            acc_f += gk * w[0]
            for k in range(1, d):
                gk = (phi[:, k : k + 1] - q * gk) / den_f
                acc_f += gk * w[k]
            acc_b = np.zeros((g_n, d))
            gk = phi[:, d : d + 1] / den_b
            acc_b += gk * w[d - 1]
            for k in range(d - 2, -1, -1):
                gk = (phi[:, k + 1 : k + 2] - one_minus * gk) / den_b
                acc_b += gk * w[k]
            values = np.maximum(q * np.where(use_fwd, acc_f, acc_b), 0.0)
            out[idx[valid]] = values[valid]
    return out
