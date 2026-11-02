"""Checks on the exact expected transition matrix."""

import numpy as np
import pytest

from uncertain_pagerank import (
    UncertainGraph,
    deterministic_transition,
    exact_savg,
    expected_transition,
    expected_transition_dp_pernode,
    expected_transition_loo_reference,
    expected_transition_naive_eq9,
    poisson_binomial_pmf,
    uniform_vector,
)

EDGES = [(0, 1), (0, 2), (0, 3), (1, 0), (1, 3), (2, 0), (2, 1), (3, 2), (4, 0)]


def random_graph(rng, n=6, m=14, low=0.05, high=0.95):
    """A random digraph with non-uniform probabilities, including out-degree 0."""
    codes = rng.choice(n * (n - 1), size=m, replace=False)
    src, off = np.divmod(codes, n - 1)
    tgt = off + (off >= src)
    probs = rng.uniform(low, high, size=m)
    return UncertainGraph.from_edges(n, np.stack([src, tgt], axis=1), probs)


def test_poisson_binomial_matches_enumeration():
    """phi[k] agrees with enumerating every pattern."""
    rng = np.random.default_rng(0)
    q = rng.uniform(0.0, 1.0, size=10)
    bits = ((np.arange(1 << 10)[:, None] >> np.arange(10)) & 1).astype(bool)
    w = np.prod(np.where(bits, q, 1.0 - q), axis=1)
    expected = np.bincount(bits.sum(axis=1), weights=w, minlength=11)
    assert np.allclose(poisson_binomial_pmf(q), expected)
    assert np.isclose(poisson_binomial_pmf(q).sum(), 1.0)


def test_dp_matches_leave_one_out_reference():
    """The deconvolution route agrees with recomputing the DP each time."""
    rng = np.random.default_rng(1)
    for _ in range(20):
        g = random_graph(rng)
        a = expected_transition(g, method="dp")
        b = expected_transition_loo_reference(g)
        assert np.allclose(a.B.data, b.B.data, atol=1e-14)
        assert np.allclose(a.delta, b.delta, atol=1e-14)


def test_uniform_closed_form_matches_dp():
    """The closed form of Eq. (6) for uniform q agrees with the general DP."""
    rng = np.random.default_rng(2)
    g0 = random_graph(rng)
    for q in [0.0, 0.1, 0.5, 0.85, 0.999, 1.0]:
        g = g0.with_probs(np.full(g0.m, q))
        u = expected_transition(g, method="uniform")
        d = expected_transition(g, method="dp")
        assert np.allclose(u.B.data, d.B.data, atol=1e-13), q
        assert np.allclose(u.delta, d.delta, atol=1e-13), q


def test_row_sums_are_stochastic():
    """The row sums sum_v B[u,v] + delta_u equal one."""
    rng = np.random.default_rng(3)
    for _ in range(20):
        g = random_graph(rng)
        assert expected_transition(g).row_sum_error() < 1e-13


def test_numerically_stable_near_q_one():
    """Accuracy holds as q -> 1, where the forward recurrence of Eq. (9) is
    unstable with amplification q/(1-q)."""
    rng = np.random.default_rng(4)
    for q_hi in [0.99, 0.999, 0.9999, 1.0 - 1e-12]:
        g = random_graph(rng, n=8, m=24, low=q_hi, high=q_hi + (1.0 - q_hi) * 0.5)
        a = expected_transition(g, method="dp")
        b = expected_transition_loo_reference(g)
        assert np.allclose(a.B.data, b.B.data, atol=1e-12), q_hi
        assert a.row_sum_error() < 1e-12, q_hi


def test_matches_brute_force_over_possible_worlds():
    """The exact expected transition matrix agrees with sum_X q(G_X) P_X."""
    rng = np.random.default_rng(5)
    for _ in range(5):
        g = random_graph(rng, n=5, m=12)
        z = uniform_vector(g.n)
        brute = exact_savg(g, alpha=0.85, z=z, with_expected_transition=True)
        assert np.allclose(expected_transition(g).to_dense(z), brute.expected_transition, atol=1e-12)


def test_certain_graph_reduces_to_plain_transition():
    """With q_e = 1 it reduces to the ordinary transition matrix."""
    g = UncertainGraph.from_edges(5, EDGES, 1.0)
    et = deterministic_transition(g)
    dense = et.to_dense(uniform_vector(5))
    assert np.allclose(dense[0], [0, 1 / 3, 1 / 3, 1 / 3, 0])
    assert np.allclose(dense[4], [1, 0, 0, 0, 0])
    assert np.allclose(et.delta, 0.0)


def test_zero_probability_makes_node_dangling():
    """A node with q_e = 0 is dangling with probability one."""
    g = UncertainGraph.from_edges(3, [(0, 1), (0, 2), (1, 2)], [0.0, 0.0, 0.5])
    et = expected_transition(g, method="dp")
    assert np.isclose(et.delta[0], 1.0)
    assert np.allclose(et.B.toarray()[0], 0.0)
    assert np.isclose(et.delta[1], 0.5)
    assert np.isclose(et.B.toarray()[1, 2], 0.5)


def test_isolated_node_is_dangling():
    """A node with no out-link has delta = 1."""
    g = UncertainGraph.from_edges(3, [(0, 1)], 0.5)
    et = expected_transition(g)
    assert np.allclose(et.delta, [0.5, 1.0, 1.0])


def test_multi_edge_is_rejected():
    with pytest.raises(ValueError):
        UncertainGraph.from_edges(3, [(0, 1), (0, 1)], 0.5)


def test_naive_eq9_diverges_on_high_degree_nodes():
    """The naive forward form of Eq. (9) breaks at high out-degree and high q;
    the stable form does not.

    Rounding error grows exponentially once the amplification q/(1-q) exceeds
    one. The conference experiments used a uniform q, for which the closed form
    of Eq. (6) applies, so this path was never exercised there. With non-uniform
    q_e, selecting the direction of the recurrence is necessary.
    """
    d = 120
    q = np.full(d, 0.9)
    g = UncertainGraph.from_edges(d + 1, [(0, v) for v in range(1, d + 1)], q)
    ref = expected_transition_loo_reference(g).B.data
    stable = expected_transition(g, method="dp").B.data
    naive = expected_transition_naive_eq9(g).B.data
    assert np.abs(stable / ref - 1.0).max() < 1e-12
    assert np.abs(naive / ref - 1.0).max() > 1.0


def test_batched_dp_matches_per_node_loop():
    """The form batched by out-degree agrees with the node-by-node loop."""
    rng = np.random.default_rng(6)
    for n, m in [(30, 120), (80, 600), (150, 1800)]:
        codes = rng.choice(n * (n - 1), size=m, replace=False)
        src, off = np.divmod(codes, n - 1)
        g = UncertainGraph.from_edges(
            n, np.stack([src, off + (off >= src)], axis=1), rng.uniform(0.001, 0.999, size=m)
        )
        a = expected_transition(g, method="dp")
        b = expected_transition_dp_pernode(g)
        assert np.allclose(a.B.data, b.B.data, atol=1e-14), (n, m)
        assert np.allclose(a.delta, b.delta, atol=1e-14), (n, m)


def test_batched_dp_respects_memory_cap():
    """Splitting by max_batch_elements does not change the result."""
    rng = np.random.default_rng(7)
    g = random_graph(rng, n=40, m=300, low=0.01, high=0.99)
    full = expected_transition(g, method="dp", max_batch_elements=1 << 24)
    tiny = expected_transition(g, method="dp", max_batch_elements=4)
    assert np.allclose(full.B.data, tiny.B.data, atol=1e-15)
    assert np.allclose(full.delta, tiny.delta, atol=1e-15)


def test_uniform_vectorized_path_matches_dp_on_wide_degree_range():
    """The vectorised closed form for uniform q agrees with the DP even under a
    skewed out-degree."""
    edges = [(0, v) for v in range(1, 60)] + [(1, 2), (2, 3), (3, 1), (4, 5)]
    g = UncertainGraph.from_edges(61, edges, 0.6)
    u = expected_transition(g, method="uniform")
    d = expected_transition(g, method="dp")
    assert np.allclose(u.B.data, d.B.data, atol=1e-14)
    assert np.allclose(u.delta, d.delta, atol=1e-14)
    assert u.row_sum_error() < 1e-14


def test_second_moment_matches_brute_force():
    """The second moments agree with the average over all worlds."""
    rng = np.random.default_rng(8)
    for _ in range(4):
        g = random_graph(rng, n=5, m=12)
        z = uniform_vector(g.n)
        et = expected_transition(g, with_second_moment=True)
        brute = exact_savg(g, alpha=0.85, z=z, with_expected_transition=True)
        dense = np.asarray(et.M2.todense()) + np.outer(et.delta, z * z)
        assert np.allclose(dense, brute.expected_transition_sq, atol=1e-13)


def test_variance_terms_match_brute_force():
    """V(v) = sum_u y0(u)^2 Var(p_X(u,v)) agrees with the exact variance."""
    from uncertain_pagerank.bounds import transition_variance_terms
    from uncertain_pagerank.pagerank import pagerank

    rng = np.random.default_rng(9)
    for _ in range(4):
        g = random_graph(rng, n=5, m=12)
        z = uniform_vector(g.n)
        et = expected_transition(g, with_second_moment=True)
        y0 = pagerank(et, alpha=0.85, z=z).scores
        brute = exact_savg(g, alpha=0.85, z=z, with_expected_transition=True)
        var_exact = brute.expected_transition_sq - brute.expected_transition**2
        assert np.allclose((y0**2) @ var_exact, transition_variance_terms(et, y0, z), atol=1e-15)


def test_pair_second_moment_matches_enumeration():
    """E[x_v x_w / D^2] for two links of one source agrees with enumeration."""
    from uncertain_pagerank.second_order import row_second_moment_matrix

    rng = np.random.default_rng(10)
    for d in [2, 3, 5, 7]:
        q = rng.uniform(0.05, 0.95, size=d)
        bits = ((np.arange(1 << d)[:, None] >> np.arange(d)) & 1).astype(float)
        w = np.prod(np.where(bits > 0, q, 1.0 - q), axis=1)
        k = bits.sum(axis=1)
        inv = np.where(k > 0, 1.0 / np.maximum(k, 1) ** 2, 0.0)
        brute = np.einsum("b,bi,bj,b->ij", w, bits, bits, inv)
        assert np.allclose(row_second_moment_matrix(q), brute, atol=1e-14), d


def test_bias_estimate_reduces_the_gap():
    """The second-order bias correction narrows the gap between p-avg and s-avg."""
    from uncertain_pagerank.pagerank import pagerank
    from uncertain_pagerank.second_order import bias_estimate

    rng = np.random.default_rng(11)
    ratios = []
    for _ in range(5):
        g = random_graph(rng, n=7, m=13, low=0.15, high=0.9)
        z = uniform_vector(g.n)
        et = expected_transition(g)
        y0 = pagerank(et, alpha=0.85, z=z).scores
        mu = exact_savg(g, alpha=0.85, z=z).scores
        est = bias_estimate(g, y0, 0.85, z, et.delta, et.B.data)
        ratios.append(np.abs(mu - y0 - est).max() / np.abs(mu - y0).max())
    assert np.median(ratios) < 0.6, ratios


def test_sampled_bias_estimate_converges():
    """The sampled second-order term approaches the true bias as the sample grows."""
    from uncertain_pagerank.pagerank import pagerank
    from uncertain_pagerank.second_order import bias_estimate_sampled

    rng = np.random.default_rng(12)
    g = random_graph(rng, n=7, m=13, low=0.2, high=0.9)
    z = uniform_vector(g.n)
    et = expected_transition(g)
    y0 = pagerank(et, alpha=0.85, z=z).scores
    mu = exact_savg(g, alpha=0.85, z=z).scores
    base = np.abs(mu - y0).max()
    few = np.median(
        [
            np.abs(mu - y0 - bias_estimate_sampled(g, y0, 0.85, z, 8, seed=t, expected=et)).max()
            for t in range(9)
        ]
    )
    many = np.median(
        [
            np.abs(mu - y0 - bias_estimate_sampled(g, y0, 0.85, z, 512, seed=t, expected=et)).max()
            for t in range(9)
        ]
    )
    assert many < few
    assert many < 0.5 * base
