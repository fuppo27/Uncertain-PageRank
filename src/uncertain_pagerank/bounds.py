"""Computable error bounds on the gap between p-avg and s-avg.

Notation:
    Pbar = E[P_X],  y0 = PageRank(Pbar) (p-avg),  mu = E[PageRank(P_X)] (s-avg),
    Delta_X = P_X - Pbar,  R = (I - alpha Pbar^T)^{-1}.

From the PageRank fixed point,

    y_X - y0 = alpha (I - alpha P_X^T)^{-1} Delta_X^T y0,
    mu - y0  = alpha R E[ Delta_X^T (y_X - y0) ]

where the second line reflects the first-order term vanishing because
E[Delta_X] = 0.

The quantity this rests on is the following. The rows are independent, so with

    V(v) = sum_u y0(u)^2 Var(p_X(u,v))

Jensen's inequality gives

    E|| Delta_X^T y0 ||_1 = sum_v E| sum_u Delta_X(u,v) y0(u) | <= sum_v sqrt(V(v))

Var(p_X(u,v)) follows exactly from the second moments of the expected
transition matrix, so V(v) costs O(|E| + |V|).

The bounds are conservative, so how far they are actually usable is measured
against the exact values on small graphs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .expected_transition import ExpectedTransition
from .pagerank import _resolve_z


@dataclass(frozen=True)
class ErrorEstimates:
    """Bounds and proxies for the gap between p-avg and s-avg.

    The bound_* fields are guaranteed upper bounds; the proxy_* fields are
    indicative only and carry no guarantee.
    """

    v_per_node: np.ndarray
    """V(v) = sum_u y0(u)^2 Var(p_X(u,v))。"""

    sum_sqrt_v: float
    max_sqrt_v: float
    sum_v: float

    bound_l1_first_order: float
    """An upper bound on ||mu - y0||_1, not using E[Delta]=0."""

    bound_l1_second_order: float
    """An upper bound on ||mu - y0||_1 that uses the vanishing first-order term,
    with a loose constant."""

    proxy_linf_max: float
    """An indication of ||mu - y0||_inf. No guarantee."""

    proxy_l2: float
    """An indication of ||mu - y0||_2. No guarantee."""

    @property
    def best_bound(self) -> float:
        """The smaller of the two guaranteed bounds."""
        return min(self.bound_l1_first_order, self.bound_l1_second_order)


def transition_variance_terms(
    et: ExpectedTransition, y0: np.ndarray, z: np.ndarray | None = None
) -> np.ndarray:
    """V(v) = sum_u y0(u)^2 Var(p_X(u,v)), in O(|E| + |V|).

    Even where (u,v) is not a link the variance is nonzero, because a dangling u
    gives p_X(u,v) = z(v). That contribution is

        Var(p_X(u,v)) = z(v)^2 delta_u (1 - delta_u)     (v not a child of u)

    whose only dependence on v is through z(v)^2, so it collapses to a single
    scalar.

    Args:
        et: an expected transition matrix built with with_second_moment=True.
        y0: the p-average scores.
        z: the personalization vector. Uniform by default.
    Returns:
        Length n, V(v).
    """
    if et.M2 is None:
        raise ValueError("the second moments are required: build with "
                         "expected_transition(..., with_second_moment=True)")
    n = et.n
    z = _resolve_z(z, n)
    delta = et.delta
    y0_sq = y0 * y0

    # The non-link part: a scalar T shared by every node, times z(v)^2
    t_scalar = float(y0_sq @ (delta * (1.0 - delta)))

    indptr, tgt = et.B.indptr, et.B.indices
    src = np.repeat(np.arange(n), np.diff(indptr))
    z_t = z[tgt]
    d_s = delta[src]
    mean_link = et.B.data + z_t * d_s
    second_link = et.M2.data + z_t * z_t * d_s
    var_link = np.maximum(second_link - mean_link * mean_link, 0.0)
    var_nonlink = z_t * z_t * d_s * (1.0 - d_s)

    correction = y0_sq[src] * (var_link - var_nonlink)
    return z * z * t_scalar + np.bincount(tgt, weights=correction, minlength=n)


def error_estimates(
    et: ExpectedTransition,
    y0: np.ndarray,
    alpha: float,
    z: np.ndarray | None = None,
) -> ErrorEstimates:
    """Compute the bounds and the proxies together.

    The first-order bound:
        ||mu - y0||_1 <= E||y_X - y0||_1 <= (alpha/(1-alpha)) E||Delta_X^T y0||_1
                       <= (alpha/(1-alpha)) sum_v sqrt(V(v))

    The second-order bound, using the vanishing first-order term:
        ||mu - y0||_1 <= (2 alpha^2/(1-alpha)^2) sum_v sqrt(V(v))

        using ||R||_1, ||R_X||_1 <= 1/(1-alpha) and the per-row L1 deviation
        max_u ||P_X(u,.) - Pbar(u,.)||_1 <= 2. At alpha = 0.85 the constants
        are 5.67 against 64.2, so the first-order bound is the smaller one.
        The second-order bound has the better order in the size of the
        uncertainty, so which one binds depends on the instance.
    """
    v = transition_variance_terms(et, y0, z)
    sqrt_v = np.sqrt(v)
    sum_sqrt_v = float(sqrt_v.sum())
    ratio = alpha / (1.0 - alpha)
    return ErrorEstimates(
        v_per_node=v,
        sum_sqrt_v=sum_sqrt_v,
        max_sqrt_v=float(sqrt_v.max()),
        sum_v=float(v.sum()),
        bound_l1_first_order=ratio * sum_sqrt_v,
        bound_l1_second_order=2.0 * ratio * ratio * sum_sqrt_v,
        proxy_linf_max=ratio * float(sqrt_v.max()),
        proxy_l2=ratio * float(np.sqrt(v.sum())),
    )


def certified_topk_size(y0: np.ndarray, eps: float, ks: tuple[int, ...]) -> dict[int, bool]:
    """Whether the top-k sets are guaranteed to agree under an error bound eps.

    If the margin between ranks k and k+1 of the p-average exceeds 2 eps, then
    ||mu - y0||_inf <= eps forces the two top-k sets to agree.
    """
    s = np.sort(y0)[::-1]
    return {k: bool(k < s.size and (s[k - 1] - s[k]) > 2.0 * eps) for k in ks}


def largest_certified_k(y0: np.ndarray, eps: float, k_max: int | None = None) -> int:
    """The largest k that can be guaranteed. Zero means not even the top node."""
    s = np.sort(y0)[::-1]
    limit = min(k_max or s.size - 1, s.size - 1)
    gaps = s[:limit] - s[1 : limit + 1]
    ok = np.flatnonzero(gaps > 2.0 * eps)
    return int(ok[-1] + 1) if ok.size else 0
