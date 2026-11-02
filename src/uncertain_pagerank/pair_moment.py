"""Apply the pair second moments of one source's out-links without forming them.

Number the out-links of node u as 1..D, let xi_j indicate connection and
K = sum_j xi_j be the connection count. What is needed is

    A(i,j) = E[ xi_i xi_j / K^2 ]

applied to a coefficient vector c, that is

    (A c)_i = E[ xi_i * T / K^2 ],     T = sum_j xi_j c_j

Forming A explicitly costs O(D^3), since each of its D^2 entries needs a
leave-two-out distribution. The identity below brings that down to O(D^2).

Conditioning on xi_i = 1 and writing K = 1 + K', T = c_i + T' for the
quantities that exclude link i,

    (A c)_i = q_i * [ c_i * sum_k g_i[k]/(1+k)^2 + sum_k h_i[k]/(1+k)^2 ],
    g_i[k] = P(K' = k),   h_i[k] = E[T' 1(K' = k)].

where g_i is the leave-one-out distribution, obtainable for all links in
O(D^2). The h_i follow from the full weighted profile

    psi[k] = E[T 1(K = k)] = sum_j c_j q_j g_j[k-1]

by solving, for link i,

    psi[k] = c_i q_i g_i[k-1] + q_i h_i[k-1] + (1 - q_i) h_i[k]

in O(D) each. The direction of the deconvolution is chosen by q_i so that the
error amplification stays at most one, as in the expected transition matrix.

Applying the covariance therefore costs O(sum_u D_u^2), the same as building
the expected transition matrix itself.
"""

from __future__ import annotations

import numpy as np

from .expected_transition import poisson_binomial_pmf


def _deconvolve_pmf(phi: np.ndarray, qi: float) -> np.ndarray:
    """Remove one link with probability qi from the distribution. Length drops by one."""
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


def _deconvolve_weighted(
    psi: np.ndarray, g_i: np.ndarray, qi: float, ci: float
) -> np.ndarray:
    """Solve psi[k] = ci*qi*g_i[k-1] + qi*h[k-1] + (1-qi)*h[k] for h.

    Args:
        psi: length D+1, E[T 1(K=k)] over all links.
        g_i: length D, the connection-count distribution without link i.
        qi: the connection probability of link i.
        ci: the coefficient of link i.
    Returns:
        Length D, h_i[k] = E[T' 1(K'=k)].
    """
    d = g_i.shape[0]
    h = np.empty(d, dtype=np.float64)
    if qi < 0.5:
        den = 1.0 - qi
        prev = 0.0  # h[-1]
        for k in range(d):
            g_prev = g_i[k - 1] if k >= 1 else 0.0
            h[k] = (psi[k] - ci * qi * g_prev - qi * prev) / den
            prev = h[k]
    else:
        nxt = 0.0  # h[d] is 0, since K' <= d-1
        for k in range(d - 1, -1, -1):
            h[k] = (psi[k + 1] - ci * qi * g_i[k] - (1.0 - qi) * nxt) / qi
            nxt = h[k]
    return h


def pair_moment_apply(q: np.ndarray, c: np.ndarray) -> np.ndarray:
    """(A c)_i = E[xi_i * (sum_j xi_j c_j) / K^2], in O(D^2).

    Args:
        q: length D, the connection probabilities.
        c: length D, the coefficient vector.
    Returns:
        Length D, (A c).
    """
    d = q.shape[0]
    if d == 0:
        return np.zeros(0, dtype=np.float64)
    phi = poisson_binomial_pmf(q)
    w = 1.0 / np.arange(1, d + 1, dtype=np.float64) ** 2  # w[k] = 1/(1+k)^2

    # First pass: build psi[k] = sum_j c_j q_j g_j[k-1]
    psi = np.zeros(d + 1, dtype=np.float64)
    for j in range(d):
        psi[1:] += c[j] * q[j] * _deconvolve_pmf(phi, float(q[j]))

    # Second pass: solve h_i per link and take the weighted sum
    out = np.empty(d, dtype=np.float64)
    for i in range(d):
        g_i = _deconvolve_pmf(phi, float(q[i]))
        h_i = _deconvolve_weighted(psi, g_i, float(q[i]), float(c[i]))
        out[i] = q[i] * (c[i] * float(g_i @ w) + float(h_i @ w))
    return out


def _pair_moment_rows_batched(q: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Apply the same computation to G nodes of equal (padded) out-degree.

    The same arithmetic as pair_moment_apply, on (G, D) arrays. The number of
    numpy calls drops to O(D), which is faster on graphs with many nodes.
    Padding uses q = 0 (and c = 0); a link with q = 0 does not change the
    distribution.

    Args:
        q: (G, D) connection probabilities.
        c: (G, D) coefficients.
    Returns:
        (G, D), (A c).
    """
    g_n, d = q.shape
    phi = np.zeros((g_n, d + 1), dtype=np.float64)
    phi[:, 0] = 1.0
    for j in range(d):
        qj = q[:, j : j + 1]
        phi[:, 1 : j + 2] = qj * phi[:, 0 : j + 1] + (1.0 - qj) * phi[:, 1 : j + 2]
        phi[:, 0] *= 1.0 - qj[:, 0]

    one_minus = 1.0 - q
    use_fwd = q < 0.5
    den_f = np.where(use_fwd, one_minus, 1.0)
    den_b = np.where(use_fwd, 1.0, q)
    cq = c * q

    # First pass: psi[k] = sum_j c_j q_j g_j[k-1]
    psi = np.zeros((g_n, d + 1), dtype=np.float64)
    gk_f = phi[:, 0:1] / den_f
    gk_b = phi[:, d : d + 1] / den_b
    contrib = np.where(use_fwd, gk_f, 0.0) * cq
    psi[:, 1] += contrib.sum(axis=1)
    contrib = np.where(use_fwd, 0.0, gk_b) * cq
    psi[:, d] += contrib.sum(axis=1)
    for k in range(1, d):
        gk_f = (phi[:, k : k + 1] - q * gk_f) / den_f
        psi[:, k + 1] += (np.where(use_fwd, gk_f, 0.0) * cq).sum(axis=1)
    for k in range(d - 2, -1, -1):
        gk_b = (phi[:, k + 1 : k + 2] - one_minus * gk_b) / den_b
        psi[:, k + 1] += (np.where(use_fwd, 0.0, gk_b) * cq).sum(axis=1)

    # Second pass: advance g_i and h_i together and take the weighted sum
    w = 1.0 / np.arange(1, d + 1, dtype=np.float64) ** 2
    acc_f = np.zeros((g_n, d), dtype=np.float64)
    gk = phi[:, 0:1] / den_f
    hk = (psi[:, 0:1] - 0.0 - 0.0) / den_f  # g_i[-1] = 0, h_i[-1] = 0
    acc_f += (c * gk + hk) * w[0]
    g_prev = gk
    for k in range(1, d):
        hk = (psi[:, k : k + 1] - cq * g_prev - q * hk) / den_f
        gk = (phi[:, k : k + 1] - q * g_prev) / den_f
        acc_f += (c * gk + hk) * w[k]
        g_prev = gk

    acc_b = np.zeros((g_n, d), dtype=np.float64)
    gk = phi[:, d : d + 1] / den_b
    hk = (psi[:, d : d + 1] - cq * gk) / den_b  # h_i[d] = 0
    acc_b += (c * gk + hk) * w[d - 1]
    for k in range(d - 2, -1, -1):
        gk = (phi[:, k + 1 : k + 2] - one_minus * gk) / den_b
        hk = (psi[:, k + 1 : k + 2] - cq * gk - one_minus * hk) / den_b
        acc_b += (c * gk + hk) * w[k]

    return q * np.where(use_fwd, acc_f, acc_b)


def pair_moment_quadratic(q: np.ndarray, c: np.ndarray) -> float:
    """The quadratic form c^T A c = E[T^2 / K^2 ; K >= 1] by a direct recursion.

    The attribution needs only this scalar, and it can be had without ever
    forming a leave-two-out distribution. Carrying, over the first j links,

        p_j[k] = Pr(K_j = k),
        a_j[k] = E[T_j 1(K_j = k)],
        b_j[k] = E[T_j^2 1(K_j = k)]

    and updating them when link (q_j, c_j) is appended,

        b[k] <- (1-q) b[k] + q (b[k-1] + 2 c a[k-1] + c^2 p[k-1]),
        a[k] <- (1-q) a[k] + q (a[k-1] + c p[k-1]),
        p[k] <- (1-q) p[k] + q p[k-1]

    gives c^T A c = sum_{k>=1} b_D[k] / k^2 in O(D^2) time and O(D) memory.
    The update order matters: b uses the old a and p, so b comes first.

    pair_moment_apply is the more general of the two, since it returns the full
    vector `A c`, but for the quadratic form alone this recursion is faster,
    performing no deconvolution.

    Args:
        q: length D, the connection probabilities.
        c: length D, the coefficient vector.
    Returns:
        The scalar c^T A c.
    """
    d = q.shape[0]
    if d == 0:
        return 0.0
    p = np.zeros(d + 1, dtype=np.float64)
    p[0] = 1.0
    a = np.zeros(d + 1, dtype=np.float64)
    b = np.zeros(d + 1, dtype=np.float64)
    for j in range(d):
        qj = float(q[j])
        cj = float(c[j])
        rest = 1.0 - qj
        hi = j + 1
        lo, up = slice(0, hi), slice(1, hi + 1)
        b[up] = rest * b[up] + qj * (b[lo] + 2.0 * cj * a[lo] + cj * cj * p[lo])
        a[up] = rest * a[up] + qj * (a[lo] + cj * p[lo])
        p[up] = rest * p[up] + qj * p[lo]
        b[0] *= rest
        a[0] *= rest
        p[0] *= rest
    k = np.arange(1, d + 1, dtype=np.float64)
    return float(b[1:] @ (1.0 / (k * k)))
