"""Source-level uncertainty attribution.

The target is a contrast matrix `C`, **fixed in advance**, through

    Z = C y_X,
    Psi_u(C) = tr Cov( E[Z | X_u] ).

`Psi_u(C)` is the expected reduction in squared error about `Z` from observing
the out-link states of source `u`. By the law of total variance it satisfies,
exactly and for every source,

    Psi_u = Var(Z) - E[ Var(Z | X_u) ].

**The shares do not sum to the total.** For the true nonlinear PageRank,

    sum_u Psi_u != Var(Z)

in general, and the gap

    I_inter = Var(Z) - sum_u Psi_u

is the interaction between sources. The sum is preserved only inside the
first-order additive surrogate `phi_hat`, so `I_inter` is always measured
alongside it.

Two quantities that are easy to confuse:
  - `w_k = e_{v_k} - e_{v_{k+1}}` (scaled by 1/sqrt(2)) is the **fixed
    contrast**, determined from the p-average ranking before any world is seen.
  - `x_k = (I - alpha Pbar)^{-1} w_k` is the **propagated sensitivity**, and it
    is this one that appears in the variance.

The target is the variance of the fixed linear quantity
`y_X(v_k) - y_X(v_{k+1})`, so worlds in which the margin vanishes contribute the
value 0 and need no tie-breaking rule. It is the uncertainty of a fixed margin,
not the probability that the top-k set changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .expected_transition import ExpectedTransition
from .graph import UncertainGraph
from .pagerank import _resolve_z, pagerank_solve
from .possible_worlds import _build_transitions, _world_chunks
from .variance import variance_attribution


@dataclass(frozen=True)
class ContrastSet:
    """A set of fixed contrasts.

    Row k of `C` is `sqrt(lam_k) * W[k]`. For a set of rank boundaries we take
    `sum_k lam_k = 1` (equal weights unless stated otherwise). For the
    task-agnostic `C = I`, `W = I` and `lam = 1`.
    """

    W: np.ndarray
    """(r, n). The raw contrast vectors."""

    lam: np.ndarray
    """(r,). The weights."""

    labels: list[str]
    """Where each contrast came from; kept as fixed in advance."""

    def __post_init__(self) -> None:
        if self.W.ndim != 2:
            raise ValueError("W must have shape (r, n)")
        if self.lam.shape != (self.W.shape[0],):
            raise ValueError("len(lam) does not match the number of contrasts")
        if np.any(self.lam < 0):
            raise ValueError("lam must be non-negative")
        if len(self.labels) != self.W.shape[0]:
            raise ValueError("len(labels) does not match the number of contrasts")

    @property
    def C(self) -> np.ndarray:
        """The (r, n) contrast matrix."""
        return np.sqrt(self.lam)[:, None] * self.W

    @property
    def r(self) -> int:
        return int(self.W.shape[0])


def boundary_contrasts(y0: np.ndarray, ks, lam=None) -> ContrastSet:
    """Rank-boundary contrasts from ranks k and k+1 of the p-average ranking.

    Determined from `y0` alone, **before any possible world is examined**. The
    set `ks` is fixed in advance from the intended use; boundaries that happen
    to work well are never selected afterwards.

    Args:
        y0: p-average scores.
        ks: the ranks k to use (1-indexed; the margin between k and k+1).
        lam: weights. Equal weights summing to one by default.
    """
    y0 = np.asarray(y0, dtype=np.float64)
    order = np.argsort(-y0, kind="stable")
    ks = [int(k) for k in ks]
    if any(k < 1 or k + 1 > y0.size for k in ks):
        raise ValueError("k out of range")
    rows, labels = [], []
    for k in ks:
        w = np.zeros(y0.size)
        w[order[k - 1]] = 1.0
        w[order[k]] = -1.0
        rows.append(w / np.sqrt(2.0))
        labels.append(f"boundary_k{k}(v{order[k - 1]}-v{order[k]})")
    lam = np.full(len(ks), 1.0 / len(ks)) if lam is None else np.asarray(lam, float)
    return ContrastSet(W=np.array(rows), lam=lam, labels=labels)


def node_contrasts(nodes, n: int, lam=None) -> ContrastSet:
    """w_v = e_v - (1/n) 1: an auxiliary contrast for one node's own spread."""
    nodes = [int(v) for v in nodes]
    rows = []
    for v in nodes:
        w = np.full(n, -1.0 / n)
        w[v] += 1.0
        rows.append(w)
    lam = np.full(len(nodes), 1.0 / len(nodes)) if lam is None else np.asarray(lam, float)
    return ContrastSet(W=np.array(rows), lam=lam, labels=[f"node{v}" for v in nodes])


def identity_contrast(n: int) -> ContrastSet:
    """C = I。Psi_u(I) = sum_v Var(E[y_X(v) | X_u])。

    Used for a task-agnostic robustness check that singles out no `w`.
    """
    return ContrastSet(W=np.eye(n), lam=np.ones(n), labels=[f"e{v}" for v in range(n)])


def gap_stratified_contrasts(
    y0: np.ndarray, per_stratum: int = 2, rng: np.random.Generator | None = None
) -> ContrastSet:
    """Pairwise contrasts stratified in advance by small, medium and large gap.

    An auxiliary family, used to show that the main result does not depend on
    one particular boundary. The strata follow from `y0` alone, so no possible
    world is examined.
    """
    y0 = np.asarray(y0, dtype=np.float64)
    n = y0.size
    order = np.argsort(-y0, kind="stable")
    gaps = y0[order[:-1]] - y0[order[1:]]  # gaps between adjacent ranks
    rng = np.random.default_rng(0) if rng is None else rng
    ranks = np.argsort(gaps, kind="stable")  # smallest gap first
    thirds = np.array_split(ranks, 3)
    rows, labels = [], []
    for name, part in zip(("small", "mid", "large"), thirds):
        if part.size == 0:
            continue
        pick = rng.choice(part, size=min(per_stratum, part.size), replace=False)
        for i in np.sort(pick):
            w = np.zeros(n)
            w[order[i]] = 1.0
            w[order[i + 1]] = -1.0
            rows.append(w / np.sqrt(2.0))
            labels.append(f"gap_{name}(rank{i + 1})")
    lam = np.full(len(rows), 1.0 / len(rows))
    return ContrastSet(W=np.array(rows), lam=lam, labels=labels)


def resolvent_columns(
    et: ExpectedTransition,
    rhs: np.ndarray,
    alpha: float = 0.85,
    z: np.ndarray | None = None,
    tol: float = 1e-14,
    max_iter: int = 100_000,
) -> np.ndarray:
    """Solve x_k = (I - alpha Pbar)^{-1} rhs_k for every column at once.

    `contribution_vectors` handles only rhs = e_v; this generalises it to an
    arbitrary right-hand side, and hence to an arbitrary contrast.

    Args:
        rhs: (n, r)。
    Returns:
        (n, r)。
    """
    n = et.n
    z = _resolve_z(z, n)
    rhs = np.asarray(rhs, dtype=np.float64)
    if rhs.ndim == 1:
        rhs = rhs[:, None]
    x = rhs.copy()
    for _ in range(max_iter):
        nxt = rhs + alpha * (et.B @ x + np.outer(et.delta, z @ x))
        if float(np.abs(nxt - x).max()) < tol:
            return nxt
        x = nxt
    raise RuntimeError("the resolvent solve did not converge")


def analytic_attribution(
    graph: UncertainGraph,
    et: ExpectedTransition,
    y0: np.ndarray,
    contrasts: ContrastSet,
    alpha: float = 0.85,
    z: np.ndarray | None = None,
) -> np.ndarray:
    """Per-source attribution Phi_hat_u(C) under the first-order surrogate.

        Phi_hat_u(C) = sum_k lam_k * alpha^2 y0(u)^2 (x_k^T Cov_u x_k)

    Non-negative, and the shares sum to the variance of the linearized
    quantity exactly.

    Returns:
        Length n: the share carried by each source node.
    """
    z = _resolve_z(z, graph.n)
    x = resolvent_columns(et, contrasts.W.T, alpha=alpha, z=z)
    out = np.zeros(graph.n)
    for k in range(contrasts.r):
        att = variance_attribution(graph, et, y0, x[:, k], alpha=alpha, z=z)
        out += contrasts.lam[k] * att.per_source
    return out


@dataclass(frozen=True)
class WorldScores:
    """PageRank scores over all possible worlds."""

    bits: np.ndarray
    """(W, m_u) int8: the connection state of each uncertain link."""

    weights: np.ndarray
    """(W,): the probability of each world."""

    scores: np.ndarray
    """(W, n): the PageRank of each world."""

    uncertain_edges: np.ndarray
    """Length m_u: the link index of each uncertain link."""

    edge_source: np.ndarray
    """Length m_u: the source node of each uncertain link."""


def enumerate_world_scores(
    graph: UncertainGraph,
    alpha: float = 0.85,
    z: np.ndarray | None = None,
    max_worlds: int = 1 << 20,
    chunk_size: int | None = None,
) -> WorldScores:
    """Enumerate the PageRank scores of all possible worlds.

    `pagerank_solve` solves the linear system directly, so no iteration error
    enters. That is what the machine-precision gates need.
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
        raise ValueError(f"2^{m_u} possible worlds exceeds the limit {max_worlds}")
    if chunk_size is None:
        chunk_size = max(1, min(1 << m_u, int(2**22 / max(n * n, 1))))

    b_all, w_all, s_all = [], [], []
    for bits in _world_chunks(m_u, chunk_size):
        w = np.prod(np.where(bits > 0, q_u, 1.0 - q_u), axis=1) if m_u else np.ones(1)
        p = _build_transitions(bits, n, certain, uncertain, z)
        b_all.append(bits.astype(np.int8))
        w_all.append(w)
        s_all.append(pagerank_solve(p, alpha, z))
    return WorldScores(
        bits=np.concatenate(b_all, axis=0),
        weights=np.concatenate(w_all),
        scores=np.concatenate(s_all, axis=0),
        uncertain_edges=np.flatnonzero(is_uncertain),
        edge_source=edges[is_uncertain][:, 0],
    )


def linear_contrast_values(
    graph: UncertainGraph,
    et: ExpectedTransition,
    y0: np.ndarray,
    contrasts: ContrastSet,
    bits: np.ndarray,
    alpha: float = 0.85,
    z: np.ndarray | None = None,
) -> np.ndarray:
    """Z_lin = C y_lin over all worlds, on the linearized scores.

    The first-order expansion of PageRank about Pbar is

        y_X ~= y0 + alpha (I - alpha Pbar^T)^{-1} (P_X - Pbar)^T y0

    so the contrast of it is

        w^T y_lin(X) = w^T y0 + alpha * sum_u y0(u) * ( (P_X x)_u - (Pbar x)_u )

    with x = (I - alpha Pbar)^{-1} w. Because the rows are independent, the
    variance of this quantity equals the sum returned by
    `analytic_attribution` exactly.

    Returns:
        (W, r): Z_lin, scaled by `sqrt(lam)`.
    """
    n = graph.n
    z = _resolve_z(z, n)
    edges, probs = graph.edge_list(), graph.probs
    is_certain = probs >= 1.0
    is_uncertain = ~(is_certain | (probs <= 0.0))
    certain, uncertain = edges[is_certain], edges[is_uncertain]

    x = resolvent_columns(et, contrasts.W.T, alpha=alpha, z=z)  # (n, r)
    sbar = et.B @ x + np.outer(et.delta, z @ x)  # (n, r) = Pbar x
    base = contrasts.W @ y0  # (r,)
    scale = np.sqrt(contrasts.lam)

    out = np.empty((bits.shape[0], contrasts.r))
    step = max(1, int(2**22 / max(n * n, 1)))
    for start in range(0, bits.shape[0], step):
        chunk = bits[start : start + step].astype(np.float64)
        p = _build_transitions(chunk, n, certain, uncertain, z)  # (c, n, n)
        s = np.matmul(p, x)  # (c, n, r) = P_X x
        # alpha * sum_u y0(u) * (s_u - sbar_u)
        delta_term = alpha * np.einsum("u,cur->cr", y0, s - sbar[None, :, :])
        out[start : start + step] = (base[None, :] + delta_term) * scale[None, :]
    return out


def true_contrast_values(scores: np.ndarray, contrasts: ContrastSet) -> np.ndarray:
    """Z = C y_X on the true PageRank. Returns (W, r)."""
    return scores @ contrasts.C.T


def exact_attribution(
    bits: np.ndarray,
    weights: np.ndarray,
    values: np.ndarray,
    edge_source: np.ndarray,
    n_nodes: int,
) -> np.ndarray:
    """Psi_u = tr Cov( E[Z | X_u] ), computed exactly by enumeration.

    Worlds are grouped by the state of the uncertain out-links of `u`, and the
    spread of the within-group means is measured.

    Args:
        bits: (W, m_u)。
        weights: (W,), the probability of each world.
        values: (W, r), the values of Z.
        edge_source: length m_u, the source of each uncertain link.
        n_nodes: the number of nodes.
    Returns:
        Length n_nodes.
    """
    weights = np.asarray(weights, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    ez = weights @ values  # (r,)
    psi = np.zeros(n_nodes)
    for u in range(n_nodes):
        pos = np.flatnonzero(edge_source == u)
        if pos.size == 0:
            continue  # a source with no uncertain out-link contributes 0
        code = (bits[:, pos].astype(np.int64) << np.arange(pos.size)).sum(axis=1)
        ng = 1 << pos.size
        pw = np.bincount(code, weights=weights, minlength=ng)
        ok = pw > 0
        cond = np.empty((int(ok.sum()), values.shape[1]))
        for j in range(values.shape[1]):
            num = np.bincount(code, weights=weights * values[:, j], minlength=ng)
            cond[:, j] = num[ok] / pw[ok]
        psi[u] = float(pw[ok] @ ((cond - ez[None, :]) ** 2).sum(axis=1))
    return psi


def total_variance(weights: np.ndarray, values: np.ndarray) -> float:
    """tr Cov(Z) = sum_j Var(Z_j)。"""
    weights = np.asarray(weights, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    ez = weights @ values
    return float(weights @ (values**2).sum(axis=1) - (ez**2).sum())


def interaction_term(
    weights: np.ndarray, values: np.ndarray, psi: np.ndarray
) -> float:
    """I_inter = Var(Z) - sum_u Psi_u。

    It is not zero for the true nonlinear PageRank. Where it is large, ranking
    single sources is not the whole story, and that has to be said.
    """
    return total_variance(weights, values) - float(psi.sum())


def exact_set_attribution(
    bits: np.ndarray,
    weights: np.ndarray,
    values: np.ndarray,
    positions: np.ndarray,
) -> float:
    """Psi_S = tr Cov( E[Z | X_S] ), computed exactly by enumeration.

    The set-valued version of `exact_attribution`. `positions` are positions in
    the uncertain-link array; a caller passing a set of sources expands it to
    those sources' out-link positions first.

    The value of a set of sources is not additive, so
    `Psi_S != sum_{u in S} Psi_u`.
    """
    weights = np.asarray(weights, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.int64)
    ez = weights @ values
    if positions.size == 0:
        return 0.0
    if positions.size > 24:
        raise ValueError("too many observed links to group the worlds")
    code = (bits[:, positions].astype(np.int64) << np.arange(positions.size)).sum(axis=1)
    ng = 1 << positions.size
    pw = np.bincount(code, weights=weights, minlength=ng)
    ok = pw > 0
    total = 0.0
    cond = np.empty((int(ok.sum()), values.shape[1]))
    for j in range(values.shape[1]):
        num = np.bincount(code, weights=weights * values[:, j], minlength=ng)
        cond[:, j] = num[ok] / pw[ok]
    total = float(pw[ok] @ ((cond - ez[None, :]) ** 2).sum(axis=1))
    return total


def source_positions(edge_source: np.ndarray, sources) -> np.ndarray:
    """Expand a set of sources to positions in the uncertain-link array."""
    return np.flatnonzero(np.isin(edge_source, np.asarray(list(sources), dtype=np.int64)))


@dataclass(frozen=True)
class PickFreezeResult:
    """The result of estimating Psi_u by a pick-freeze design."""

    sources: np.ndarray
    psi: np.ndarray
    """Length len(sources): the estimates of Psi_u."""

    psi_lo: np.ndarray
    psi_hi: np.ndarray
    """A 95% bootstrap interval over the samples. NaN when n_boot=0."""

    var_z: float
    n_samples: int
    n_solves: int
    seconds: float


def pickfreeze_attribution(
    graph: UncertainGraph,
    contrasts: ContrastSet,
    sources=None,
    n_samples: int = 2000,
    seed: int = 0,
    alpha: float = 0.85,
    z: np.ndarray | None = None,
    tol: float = 1e-12,
    chunk: int = 512,
    n_boot: int = 0,
) -> PickFreezeResult:
    """The standard pick-freeze estimator of Psi_u beyond enumeration scale.

    If `X` and `X'` share `X_u` and are independent elsewhere, then

        Psi_u = tr Cov( E[Z | X_u] ) = sum_j Cov( Z_j(X), Z_j(X') )

    holds. We draw `A`, build `C^u` by replacing only the u columns of `B` with
    those of `A`, and estimate the covariance, using Janon-style centering to
    reduce the variance.

    This costs `n_samples * (1 + |sources|)` model evaluations, against one
    resolvent solve per contrast for the analytic route. That gap is what
    "computable without sampling" means here.

    Returns:
        PickFreezeResult。
    """
    import time as _time

    from .world_batch import batch_world_scores, sample_masks

    t0 = _time.perf_counter()
    z = _resolve_z(z, graph.n)
    rng = np.random.default_rng(seed)
    a = sample_masks(graph, n_samples, rng)
    b = sample_masks(graph, n_samples, rng)
    if sources is None:
        sources = np.unique(np.repeat(np.arange(graph.n), graph.out_degrees())[
            (graph.probs > 0.0) & (graph.probs < 1.0)
        ])
    sources = np.asarray(sources, dtype=np.int64)

    cmat = contrasts.C
    za = batch_world_scores(graph, a, alpha, z, tol, chunk=chunk) @ cmat.T
    solves = n_samples

    psi = np.empty(sources.size)
    lo = np.full(sources.size, np.nan)
    hi = np.full(sources.size, np.nan)
    boot_rng = np.random.default_rng(seed + 977)

    for i, u in enumerate(sources):
        cols = slice(int(graph.indptr[u]), int(graph.indptr[u + 1]))
        cu = b.copy()
        cu[:, cols] = a[:, cols]  # share only u's out-links with A
        zu = batch_world_scores(graph, cu, alpha, z, tol, chunk=chunk) @ cmat.T
        solves += n_samples
        prod = (za * zu).sum(axis=1)
        mid = (za + zu) / 2.0
        psi[i] = prod.mean() - (mid.mean(axis=0) ** 2).sum()
        if n_boot:
            idx = boot_rng.integers(0, n_samples, size=(n_boot, n_samples))
            vals = prod[idx].mean(axis=1) - (mid[idx].mean(axis=1) ** 2).sum(axis=1)
            lo[i], hi[i] = np.percentile(vals, [2.5, 97.5])

    var_z = float((za**2).mean(axis=0).sum() - (za.mean(axis=0) ** 2).sum())
    return PickFreezeResult(
        sources=sources, psi=psi, psi_lo=lo, psi_hi=hi, var_z=var_z,
        n_samples=n_samples, n_solves=solves, seconds=_time.perf_counter() - t0,
    )


def shapley_effects(
    bits: np.ndarray,
    weights: np.ndarray,
    values: np.ndarray,
    edge_source: np.ndarray,
    sources: np.ndarray,
    max_observed_edges: int = 24,
) -> np.ndarray:
    """Variance-based Shapley effects, computed exactly by enumeration.

    With the characteristic function `c(S) = tr Cov( E[Z | X_S] )`, we compute

        Sh_u = sum_{S subset D\\{u}} |S|!(d-|S|-1)!/d! * ( c(S+u) - c(S) )

    Unlike `Psi_u` (the first-order Sobol index), these **sum to `tr Cov(Z)`
    exactly**: they allocate the interaction `I_inter` over the sources.

    All 2^d subsets are enumerated, so `sources` is limited to about 12.

    Args:
        sources: the sources to attribute over. d = len(sources).
    Returns:
        Length d, in the same order as `sources`.
    """
    from math import factorial

    sources = np.asarray(sources, dtype=np.int64)
    d = int(sources.size)
    if d > 20:
        raise ValueError("too many sources to enumerate the subsets")

    # the uncertain-link positions of each source
    pos = [np.flatnonzero(edge_source == int(u)) for u in sources]
    cache = np.empty(1 << d)
    for code in range(1 << d):
        sel = [pos[i] for i in range(d) if (code >> i) & 1]
        p = np.concatenate(sel) if sel else np.empty(0, dtype=np.int64)
        if p.size > max_observed_edges:
            raise ValueError("too many observed links to group the worlds")
        cache[code] = exact_set_attribution(bits, weights, values, p)

    w = np.array([factorial(s) * factorial(d - s - 1) / factorial(d) for s in range(d)])
    out = np.zeros(d)
    for code in range(1 << d):
        size = int(bin(code).count("1"))
        base = cache[code]
        for i in range(d):
            if (code >> i) & 1:
                continue
            out[i] += w[size] * (cache[code | (1 << i)] - base)
    return out


def pickfreeze_groups(
    graph: UncertainGraph,
    contrasts: ContrastSet,
    groups: list[np.ndarray],
    n_samples: int = 2000,
    seed: int = 0,
    alpha: float = 0.85,
    z: np.ndarray | None = None,
    tol: float = 1e-12,
    chunk: int = 256,
    n_boot: int = 0,
) -> PickFreezeResult:
    """Estimate `Psi_G = tr Cov(E[Z | X_G])` for a **group** of sources.

    Per-source pick-freeze does not work at real-network scale. `Psi_u` is of
    order `Var(Z) / (number of sources)` while the standard error of the
    estimator is of order `Var(Z) / sqrt(N)`, so balancing the two needs
    `N ~ (number of sources)^2`, which is 10^7 to 10^9 samples on our networks.
    Grouping makes `Psi_G` of order `Var(Z) / (number of groups)`, which is
    estimable.

    Under the first-order surrogate the variance is additive over sources, so
    the matching analytic quantity is `sum_{u in G} phi_hat_u`. The true
    `Psi_G` also contains the interaction **within** the group, so the analytic
    quantity falls short by that amount --- which is measurable.

    Args:
        groups: one array of source ids per group.
    Returns:
        A PickFreezeResult whose `sources` field holds the group indices.
    """
    import time as _time

    from .world_batch import batch_world_scores, sample_masks

    t0 = _time.perf_counter()
    z = _resolve_z(z, graph.n)
    rng = np.random.default_rng(seed)
    a = sample_masks(graph, n_samples, rng)
    b = sample_masks(graph, n_samples, rng)

    cmat = contrasts.C
    za = batch_world_scores(graph, a, alpha, z, tol, chunk=chunk) @ cmat.T
    solves = n_samples

    psi = np.empty(len(groups))
    lo = np.full(len(groups), np.nan)
    hi = np.full(len(groups), np.nan)
    boot_rng = np.random.default_rng(seed + 977)

    for i, g in enumerate(groups):
        cols = np.concatenate(
            [np.arange(graph.indptr[u], graph.indptr[u + 1]) for u in np.asarray(g)]
        ) if len(g) else np.empty(0, dtype=np.int64)
        cu = b.copy()
        if cols.size:
            cu[:, cols] = a[:, cols]
        zu = batch_world_scores(graph, cu, alpha, z, tol, chunk=chunk) @ cmat.T
        solves += n_samples
        prod = (za * zu).sum(axis=1)
        mid = (za + zu) / 2.0
        psi[i] = prod.mean() - (mid.mean(axis=0) ** 2).sum()
        if n_boot:
            idx = boot_rng.integers(0, n_samples, size=(n_boot, n_samples))
            vals = prod[idx].mean(axis=1) - (mid[idx].mean(axis=1) ** 2).sum(axis=1)
            lo[i], hi[i] = np.percentile(vals, [2.5, 97.5])

    var_z = float((za**2).mean(axis=0).sum() - (za.mean(axis=0) ** 2).sum())
    return PickFreezeResult(
        sources=np.arange(len(groups)), psi=psi, psi_lo=lo, psi_hi=hi,
        var_z=var_z, n_samples=n_samples, n_solves=solves,
        seconds=_time.perf_counter() - t0,
    )


def pickfreeze_attribution_owen(
    graph: UncertainGraph,
    contrasts: ContrastSet,
    sources=None,
    n_samples: int = 2000,
    seed: int = 0,
    alpha: float = 0.85,
    z: np.ndarray | None = None,
    tol: float = 1e-12,
    chunk: int = 256,
    n_boot: int = 0,
) -> PickFreezeResult:
    """Estimate `Psi_u` with Owen's "Correlation 2" estimator.

    The naive pick-freeze estimator has a very large variance when the target
    `tau_u^2` is small relative to the squared mean output `mu^2`. A rank margin
    is exactly that case: `Psi_u` is minute while the mean of the margin is not
    small. To avoid it, Owen uses three independent input vectors `x, y, z` in

        (1/n) sum_i ( f(x_i) - f(z_{i,u}:x_{i,-u}) )
                    ( f(x_{i,u}:y_{i,-u}) - f(y_i) )

    where `a_u:b_{-u}` takes the u component from a and the rest from b. Both
    sides are differences, so when u does not matter both factors are small.

    This costs `2 n_samples (1 + |sources|)` solves, about twice the naive
    design.

    Args:
        n_samples: n。
    Returns:
        PickFreezeResult。
    """
    import time as _time

    from .world_batch import batch_world_scores, sample_masks

    t0 = _time.perf_counter()
    z = _resolve_z(z, graph.n)
    rng = np.random.default_rng(seed)
    x = sample_masks(graph, n_samples, rng)
    y = sample_masks(graph, n_samples, rng)
    w = sample_masks(graph, n_samples, rng)  # Owen's z, renamed: z is the
    #                                          personalization vector here
    if sources is None:
        sources = np.unique(np.repeat(np.arange(graph.n), graph.out_degrees())[
            (graph.probs > 0.0) & (graph.probs < 1.0)
        ])
    sources = np.asarray(sources, dtype=np.int64)

    cmat = contrasts.C
    fx = batch_world_scores(graph, x, alpha, z, tol, chunk=chunk) @ cmat.T
    fy = batch_world_scores(graph, y, alpha, z, tol, chunk=chunk) @ cmat.T
    solves = 2 * n_samples

    psi = np.empty(sources.size)
    lo = np.full(sources.size, np.nan)
    hi = np.full(sources.size, np.nan)
    boot_rng = np.random.default_rng(seed + 977)

    for i, u in enumerate(sources):
        cols = slice(int(graph.indptr[u]), int(graph.indptr[u + 1]))
        a = x.copy()
        a[:, cols] = w[:, cols]  # w_u : x_{-u}
        b = y.copy()
        b[:, cols] = x[:, cols]  # x_u : y_{-u}
        fa = batch_world_scores(graph, a, alpha, z, tol, chunk=chunk) @ cmat.T
        fb = batch_world_scores(graph, b, alpha, z, tol, chunk=chunk) @ cmat.T
        solves += 2 * n_samples
        per = ((fx - fa) * (fb - fy)).sum(axis=1)
        psi[i] = float(per.mean())
        if n_boot:
            idx = boot_rng.integers(0, n_samples, size=(n_boot, n_samples))
            vals = per[idx].mean(axis=1)
            lo[i], hi[i] = np.percentile(vals, [2.5, 97.5])

    var_z = float((fx**2).mean(axis=0).sum() - (fx.mean(axis=0) ** 2).sum())
    return PickFreezeResult(
        sources=sources, psi=psi, psi_lo=lo, psi_hi=hi, var_z=var_z,
        n_samples=n_samples, n_solves=solves, seconds=_time.perf_counter() - t0,
    )


def selfloop_dangling_transition(et: ExpectedTransition) -> ExpectedTransition:
    """The expected transition matrix under the other dangling convention.

    The convention used elsewhere sends a node with `K_u=0` to `z`, giving the
    rank-one form `Pbar = B + delta z^T`. The alternative here keeps such a node
    where it is, `e_u`, so that

        Pbar' = B + diag(delta)

    and **every transition with `K_u>0` is unchanged**. That is what separates
    this from adding a self-loop at every node, which would turn `xi/K_u` into
    `xi/(K_u+1)` everywhere.

    Folding the diagonal into `B` and setting `delta=0` lets the existing
    PageRank and resolvent code run unchanged. The variance does change, since
    the second moment of the dangling branch becomes `x_u` rather than `z^T x`,
    so use `attribution_selfloop_dangling` for it.
    """
    import scipy.sparse as sp

    b = et.B.tolil(copy=True)
    for u in np.flatnonzero(et.delta > 0):
        b[u, u] = b[u, u] + et.delta[u]
    return ExpectedTransition(B=sp.csr_matrix(b), delta=np.zeros_like(et.delta))


def attribution_selfloop_dangling(
    graph: UncertainGraph,
    et: ExpectedTransition,
    y0: np.ndarray,
    contrasts: ContrastSet,
    alpha: float = 0.85,
    z: np.ndarray | None = None,
) -> np.ndarray:
    """Per-source attribution under the stay-put dangling convention.

    `variance_attribution` assumes the dangling branch jumps to `z`, using
    `delta (z^T x)^2` as the second moment and `(Bx + delta z^T x)_u` as the
    first. Under the other convention that branch contributes `x_u`, giving

        Var(P_u^T x) = quad_u + delta_u x_u^2 - ((Bx)_u + delta_u x_u)^2

    where `quad_u` is the second moment of the `K_u>0` branch and does not
    depend on the convention.

    Args:
        et: the **original** expected transition matrix, with B and delta
            held separately.
        y0: the p-average scores solved under this convention.
    Returns:
        Length n.
    """
    from .variance import pair_moment_edges

    z = _resolve_z(z, graph.n)
    et_mod = selfloop_dangling_transition(et)
    x = resolvent_columns(et_mod, contrasts.W.T, alpha=alpha, z=z)  # delta=0 here
    tgt = et.B.indices
    indptr = et.B.indptr
    out = np.zeros(graph.n)
    for k in range(contrasts.r):
        xk = x[:, k]
        pair = pair_moment_edges(graph, xk[tgt])
        csum = np.zeros(graph.m + 1)
        np.cumsum(xk[tgt] * pair, out=csum[1:])
        quad = csum[indptr[1:]] - csum[indptr[:-1]]
        s = et.B @ xk + et.delta * xk           # the convention enters here
        var = np.maximum(quad + et.delta * xk**2 - s * s, 0.0)
        out += contrasts.lam[k] * alpha * alpha * y0 * y0 * var
    return out


def trace_row_covariance(
    graph: UncertainGraph,
    et: ExpectedTransition,
    z: np.ndarray | None = None,
) -> np.ndarray:
    """The trace `tr(Cov_u)` of each source's row covariance.

    Used to build `y0(u)^2 tr(Cov_u)`, which is
    `Phi_hat_u = alpha^2 y0(u)^2 x^T Cov_u x` with **only the direction `x`
    removed**. The cruder `y0(u)^2 sum_e q_e(1-q_e)` also discards the random
    row normalisation and the within-row covariance, so this is the better
    control for measuring what the direction contributes.

    On the `K_u>0` branch, `sum_i P_ui^2 = 1/K_u`, so

        E[sum_i P_ui^2] = sum_{k>=1} phi_u[k]/k + delta_u ||z||^2,

    and `sum_i Pbar_ui^2` is subtracted from it.
    """
    from .pair_moment import poisson_binomial_pmf

    n = graph.n
    z = _resolve_z(z, n)
    zz = float(z @ z)
    out = np.zeros(n)
    b = et.B.tocsr()
    for u in range(n):
        s0, s1 = int(graph.indptr[u]), int(graph.indptr[u + 1])
        q = graph.probs[s0:s1]
        d = q.size
        delta = float(et.delta[u])
        if d == 0:
            # A node with no out-link is always dangling: the row is z
            # deterministically, so the variance is 0
            out[u] = 0.0
            continue
        pmf = poisson_binomial_pmf(q)
        second = float((pmf[1:] / np.arange(1, d + 1)).sum()) + delta * zz
        rs, re = int(b.indptr[u]), int(b.indptr[u + 1])
        bu = b.data[rs:re]
        zt = z[b.indices[rs:re]]
        mean_sq = float((bu * bu).sum()) + 2.0 * delta * float(bu @ zt) + delta * delta * zz
        out[u] = max(second - mean_sq, 0.0)
    return out
