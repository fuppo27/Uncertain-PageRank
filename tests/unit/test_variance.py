"""Checks on covariance propagation and the top-k probabilities."""

import numpy as np

from uncertain_pagerank import (
    UncertainGraph,
    exact_savg,
    expected_transition,
    pagerank,
    uniform_vector,
)
from uncertain_pagerank.pair_moment import (
    pair_moment_apply,
    pair_moment_quadratic,
    _pair_moment_rows_batched,
)
from uncertain_pagerank.second_order import row_second_moment_matrix
from uncertain_pagerank.variance import (
    CovarianceOperator,
    pair_moment_edges,
    score_covariance,
    topk_probability,
)


def random_graph(rng, n=8, m=14, low=0.15, high=0.9):
    codes = rng.choice(n * (n - 1), size=m, replace=False)
    src, off = np.divmod(codes, n - 1)
    return UncertainGraph.from_edges(
        n, np.stack([src, off + (off >= src)], axis=1), rng.uniform(low, high, size=m)
    )


def test_pair_moment_apply_matches_dense_matrix():
    """The O(D^2) application agrees with the O(D^3) version that forms the matrix."""
    rng = np.random.default_rng(0)
    for d in [1, 2, 3, 7, 15, 30]:
        q = rng.uniform(0.02, 0.98, size=d)
        c = rng.normal(size=d)
        assert np.allclose(pair_moment_apply(q, c), row_second_moment_matrix(q) @ c, atol=1e-12), d


def _quadratic_by_enumeration(q, c):
    """E[T^2/K^2 ; K>=1] by enumerating all 2^D worlds."""
    d = q.shape[0]
    total = 0.0
    for mask in range(1 << d):
        bits = np.array([(mask >> j) & 1 for j in range(d)], dtype=np.float64)
        k = bits.sum()
        if k == 0:
            continue
        prob = float(np.prod(np.where(bits > 0, q, 1.0 - q)))
        total += prob * (bits @ c) ** 2 / (k * k)
    return total


def test_quadratic_recursion_matches_enumeration():
    """The quadratic form from the direct recursion agrees with enumeration."""
    rng = np.random.default_rng(11)
    for d in [1, 2, 3, 6, 10, 14]:
        q = rng.uniform(0.02, 0.98, size=d)
        c = rng.normal(size=d)
        assert np.isclose(
            pair_moment_quadratic(q, c), _quadratic_by_enumeration(q, c), atol=1e-13
        ), d


def test_quadratic_recursion_matches_vector_route():
    """The quadratic-form recursion agrees with the route through the full A c."""
    rng = np.random.default_rng(12)
    for d in [1, 2, 5, 17, 40, 120]:
        q = rng.uniform(0.02, 0.98, size=d)
        c = rng.normal(size=d)
        assert np.isclose(
            pair_moment_quadratic(q, c), float(c @ pair_moment_apply(q, c)), atol=1e-12
        ), d


def test_quadratic_recursion_is_stable_near_q_one():
    """Using no deconvolution, it does not degrade as q approaches one."""
    rng = np.random.default_rng(13)
    for q_value in [0.9, 0.99, 0.999, 1.0]:
        d = 60
        q = np.full(d, q_value)
        c = rng.normal(size=d)
        got = pair_moment_quadratic(q, c)
        assert got > 0.0
        if q_value == 1.0:
            assert np.isclose(got, (c.sum() / d) ** 2, atol=1e-12)


def test_pair_moment_is_stable_near_q_one():
    """The accuracy holds as q approaches one."""
    rng = np.random.default_rng(1)
    for q_value in [0.9, 0.99, 0.999]:
        d = 25
        q = np.full(d, q_value)
        c = rng.normal(size=d)
        assert np.allclose(pair_moment_apply(q, c), row_second_moment_matrix(q) @ c, atol=1e-11)


def test_batched_pair_moment_matches_sequential():
    """The batched form agrees with the sequential one and ignores the padding."""
    rng = np.random.default_rng(2)
    q = np.array([[0.3, 0.7, 0.5, 0.0], [0.9, 0.2, 0.0, 0.0]])
    c = np.array([[1.0, -2.0, 0.5, 0.0], [0.3, 1.0, 0.0, 0.0]])
    out = _pair_moment_rows_batched(q, c)
    for i, d in enumerate([3, 2]):
        assert np.allclose(out[i, :d], pair_moment_apply(q[i, :d], c[i, :d]), atol=1e-13)
        assert np.allclose(out[i, d:], 0.0)


def test_covariance_operator_diagonal_matches_enumeration():
    """The diagonal of M agrees with the enumerated transition-probability variance."""
    rng = np.random.default_rng(3)
    for _ in range(3):
        g = random_graph(rng, n=5, m=12)
        z = uniform_vector(g.n)
        et = expected_transition(g, with_second_moment=True)
        y0 = pagerank(et, alpha=0.85, z=z).scores
        brute = exact_savg(g, alpha=0.85, z=z, with_expected_transition=True)
        var_exact = brute.expected_transition_sq - brute.expected_transition**2
        m_dense = CovarianceOperator(g, et, y0, z).to_dense()
        assert np.allclose(np.diag(m_dense), (y0**2) @ var_exact, atol=1e-14)
        assert np.allclose(m_dense, m_dense.T, atol=1e-15)


def test_score_variance_approximates_exact_variance():
    """The first-order standard deviation is within 20% of the exact value."""
    rng = np.random.default_rng(4)
    ratios = []
    for _ in range(4):
        g = random_graph(rng)
        z = uniform_vector(g.n)
        et = expected_transition(g, with_second_moment=True)
        y0 = pagerank(et, alpha=0.85, z=z).scores
        exact = exact_savg(g, alpha=0.85, z=z)
        cov = score_covariance(g, et, y0, np.arange(g.n), alpha=0.85, z=z)
        sd_est = np.sqrt(np.maximum(np.diag(cov), 0.0))
        sd_true = np.sqrt(exact.variance)
        assert np.corrcoef(sd_true, sd_est)[0, 1] > 0.95
        ratios.append(np.median(sd_est / np.maximum(sd_true, 1e-300)))
    assert 0.8 < np.median(ratios) < 1.2, ratios


def test_topk_probability_approximates_exact():
    """P(v in top-k) under the normal approximation is close to the exact value."""
    rng = np.random.default_rng(5)
    errors = []
    for _ in range(3):
        g = random_graph(rng)
        z = uniform_vector(g.n)
        et = expected_transition(g, with_second_moment=True)
        y0 = pagerank(et, alpha=0.85, z=z).scores
        exact = exact_savg(g, alpha=0.85, z=z, topk=(1, 3))
        cov = score_covariance(g, et, y0, np.arange(g.n), alpha=0.85, z=z)
        for k in (1, 3):
            est = topk_probability(y0, cov, k, n_draws=50_000, seed=0)
            errors.append(np.abs(exact.topk_prob[k] - est).mean())
            assert np.isclose(est.sum(), k, atol=1e-9)
    assert np.mean(errors) < 0.08, np.mean(errors)


def test_pair_moment_edges_is_exact_at_high_degree():
    """Rows of large out-degree go through the exact dynamic program.

    Out-degrees of 64 and above used to switch to a moment approximation. On real
    networks that only cost accuracy, so it was removed; this pins it out.
    """
    rng = np.random.default_rng(14)
    n = 3
    degrees = [70, 5, 130]  # one below the old threshold, two above
    src, dst = [], []
    for u, d in enumerate(degrees):
        targets = rng.choice([v for v in range(n + 200) if v != u], size=d, replace=False)
        src += [u] * d
        dst += list(targets)
    g = UncertainGraph.from_edges(n + 200, np.stack([src, dst], axis=1),
                                  rng.uniform(0.05, 0.95, size=len(src)))
    coeff = rng.standard_normal(g.m)
    got = pair_moment_edges(g, coeff)
    for u, d in enumerate(degrees):
        lo, hi = g.indptr[u], g.indptr[u + 1]
        assert hi - lo == d
        want = pair_moment_apply(g.probs[lo:hi], coeff[lo:hi])
        assert np.allclose(got[lo:hi], want, atol=1e-14), (u, d)
