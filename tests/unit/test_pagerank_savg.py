"""Checks on PageRank, the exact s-average and sampling."""

import numpy as np

from uncertain_pagerank import (
    UncertainGraph,
    deterministic_transition,
    exact_savg,
    expected_transition,
    monte_carlo_savg,
    pagerank,
    pagerank_dense,
    pagerank_solve,
    uniform_vector,
)
from uncertain_pagerank.metrics import linf, rank_gap, topk_overlap

EDGES = [(0, 1), (0, 2), (0, 3), (1, 0), (1, 3), (2, 0), (2, 1), (3, 2), (4, 0)]


def test_pagerank_sparse_dense_and_solve_agree():
    """The sparse iteration, the dense iteration and the direct solve agree."""
    rng = np.random.default_rng(10)
    g = UncertainGraph.from_edges(5, EDGES, rng.uniform(0.2, 0.9, size=len(EDGES)))
    z = uniform_vector(5)
    et = expected_transition(g)
    dense = et.to_dense(z)
    a = pagerank(et, alpha=0.85, z=z).scores
    b = pagerank_dense(dense, alpha=0.85, z=z).scores
    c = pagerank_solve(dense, alpha=0.85, z=z)
    assert np.allclose(a, b, atol=1e-12)
    assert np.allclose(a, c, atol=1e-11)
    assert np.isclose(a.sum(), 1.0)


def test_pagerank_matches_networkx():
    """On a certain graph, PageRank agrees with the networkx implementation."""
    nx = __import__("networkx")
    g = UncertainGraph.from_edges(5, EDGES, 1.0)
    ref = nx.pagerank(nx.DiGraph(list(EDGES)), alpha=0.85, tol=1e-14)
    ours = pagerank(deterministic_transition(g), alpha=0.85).scores
    assert np.allclose(ours, [ref[i] for i in range(5)], atol=1e-9)


def test_solve_supports_batch():
    """A (batch, n, n) stack solves at once."""
    rng = np.random.default_rng(11)
    p = rng.random((7, 4, 4))
    p /= p.sum(axis=2, keepdims=True)
    z = uniform_vector(4)
    batched = pagerank_solve(p, 0.85, z)
    for i in range(7):
        assert np.allclose(batched[i], pagerank_solve(p[i], 0.85, z), atol=1e-12)


def test_savg_equals_pavg_when_graph_is_certain():
    """With q_e = 1 there is one world, so s-avg and p-avg agree."""
    g = UncertainGraph.from_edges(5, EDGES, 1.0)
    z = uniform_vector(5)
    s = exact_savg(g, alpha=0.85, z=z)
    p = pagerank(expected_transition(g), alpha=0.85, z=z).scores
    assert s.n_worlds == 1
    assert np.allclose(s.scores, p, atol=1e-11)
    assert np.allclose(s.variance, 0.0, atol=1e-20)


def test_exact_savg_weights_sum_to_one():
    """The probabilities of the enumerated worlds sum to one."""
    rng = np.random.default_rng(12)
    g = UncertainGraph.from_edges(5, EDGES, rng.uniform(0.1, 0.9, size=len(EDGES)))
    res = exact_savg(g, alpha=0.85, topk=(1, 2))
    assert res.n_worlds == 2 ** len(EDGES)
    assert np.isclose(res.weight_sum, 1.0, atol=1e-12)
    assert np.isclose(res.scores.sum(), 1.0, atol=1e-12)
    assert np.isclose(res.topk_prob[1].sum(), 1.0, atol=1e-12)
    assert np.isclose(res.topk_prob[2].sum(), 2.0, atol=1e-12)


def test_monte_carlo_converges_to_exact_savg():
    """The sample mean approaches the exact s-average as the sample grows."""
    rng = np.random.default_rng(13)
    g = UncertainGraph.from_edges(5, EDGES, rng.uniform(0.3, 0.9, size=len(EDGES)))
    exact = exact_savg(g, alpha=0.85).scores
    mc = monte_carlo_savg(g, alpha=0.85, n_samples=4_000, seed=0, checkpoints=(20,))
    assert linf(exact, mc.checkpoints[20]) > linf(exact, mc.scores)
    assert linf(exact, mc.scores) < 5e-3


def test_pavg_and_savg_can_disagree_on_the_top_node():
    """On a three-node example, p-avg and s-avg disagree about the top node.

    A counterexample showing that matching local expected transitions do not
    guarantee the same top ranking. Confirmed against the exact value from
    exhaustive enumeration.
    """
    q = np.array([0.7, 0.7, 0.9, 0.9, 0.5, 0.1])
    edges = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
    g = UncertainGraph.from_edges(3, edges, q)
    z = uniform_vector(3)
    p = pagerank(expected_transition(g), alpha=0.85, z=z).scores
    s = exact_savg(g, alpha=0.85, z=z).scores
    assert int(np.argmax(p)) == 0
    assert int(np.argmax(s)) == 2
    assert topk_overlap(s, p, 1) == 0.0
    assert rank_gap(p, 1) < 2.0 * linf(s, p)
