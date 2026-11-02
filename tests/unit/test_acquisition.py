"""Regression tests for information acquisition.

These pin the implementation to the problem definition. Three things in
particular break quietly:

1. that the risk is exactly zero under full observation, which is where the
   definition and the implementation have to agree
2. that the two index spaces are not confused: link ids in the original graph
   against positions in the uncertain-link array
3. that the oracle reference does not shift when the budgets are not evenly
   spaced
"""

import numpy as np
import pytest

from uncertain_pagerank import UncertainGraph, uniform_vector
from uncertain_pagerank.acquisition import (
    enumerate_worlds,
    expected_loss,
    greedy_probe,
    true_voi,
    true_voi_edge,
)
from uncertain_pagerank.acquisition_scalable import (
    condition_graph,
    decode,
    expected_loss_analytic,
    positions_to_edge_ids,
    uncertain_edge_ids,
    validate_edge_ids,
)


def toy(seed=20, n=8, m=10, low=0.2, high=0.9):
    r = np.random.default_rng(seed)
    codes = r.choice(n * (n - 1), size=m, replace=False)
    src, off = np.divmod(codes, n - 1)
    return UncertainGraph.from_edges(
        n, np.stack([src, off + (off >= src)], axis=1), r.uniform(low, high, size=m)
    )


def star(n=7, seed=1):
    edges = [(0, i) for i in range(1, n)] + [(i, 0) for i in range(1, n)]
    return UncertainGraph.from_edges(
        n, edges, np.random.default_rng(seed).uniform(0.3, 0.8, size=len(edges))
    )


def mixed():
    """A graph mixing certain and uncertain links."""
    return UncertainGraph.from_edges(4, [(0, 1), (0, 2), (1, 2), (2, 3)], [1.0, 0.5, 0.6, 1.0])


@pytest.mark.parametrize("graph", [toy(), star()])
def test_full_observation_gives_zero_loss_exact(graph):
    """Observing every uncertain link fixes the world, so the exact risk is zero."""
    z = uniform_vector(graph.n)
    ens = enumerate_worlds(graph, 3, alpha=0.85, z=z)
    assert expected_loss(ens, edges=range(ens.uncertain_edges.size)) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("graph", [toy(), star()])
def test_full_observation_gives_zero_loss_analytic(graph):
    """The same for the analytic route, where tie handling and rounding error in
    the covariance both tend to break it."""
    z = uniform_vector(graph.n)
    ids = uncertain_edge_ids(graph)
    loss, decodings = expected_loss_analytic(
        graph, 3, list(ids), alpha=0.85, z=z, n_candidates=graph.n,
        n_draws=5_000, max_outcomes=1 << 13,
    )
    assert loss == pytest.approx(0.0, abs=1e-12)
    assert all(d.n_uncertain_remaining == 0 for d in decodings)


def test_loss_is_monotone_in_observations():
    """More observation never raises the risk, by the law of total variance."""
    graph = toy()
    z = uniform_vector(graph.n)
    ens = enumerate_worlds(graph, 3, alpha=0.85, z=z)
    prev = expected_loss(ens, [])
    for size in range(1, 6):
        cur = expected_loss(ens, edges=range(size))
        assert cur <= prev + 1e-12, (size, prev, cur)
        assert cur >= -1e-12
        prev = cur


def test_voi_is_nonnegative():
    graph = toy()
    z = uniform_vector(graph.n)
    ens = enumerate_worlds(graph, 3, alpha=0.85, z=z)
    assert (true_voi_edge(ens) >= -1e-12).all()
    assert (true_voi(ens) >= -1e-12).all()


def test_source_probe_equals_its_edge_set():
    """Observing a source is the same as observing all its uncertain out-links."""
    graph = toy()
    z = uniform_vector(graph.n)
    ens = enumerate_worlds(graph, 3, alpha=0.85, z=z)
    for u in np.unique(ens.edge_source):
        cols = np.flatnonzero(ens.edge_source == u)
        assert expected_loss(ens, [int(u)]) == pytest.approx(
            expected_loss(ens, edges=cols), abs=1e-12
        )


def test_edge_id_space_is_validated():
    """A certain link, a duplicate or an out-of-range id is not accepted silently."""
    g = mixed()
    assert list(uncertain_edge_ids(g)) == [1, 2]
    assert list(positions_to_edge_ids(g, [0, 1])) == [1, 2]
    for bad in ([0], [1, 1], [9]):
        with pytest.raises(ValueError):
            validate_edge_ids(g, bad)
    with pytest.raises(ValueError):
        positions_to_edge_ids(g, [5])


def test_condition_graph_only_touches_named_edges():
    g = mixed()
    out = condition_graph(g, [1], [0])
    assert out.probs[1] == 0.0
    assert list(out.probs[[0, 2, 3]]) == [1.0, 0.6, 1.0]


def test_oracle_reference_uses_budget_not_position():
    """With unevenly spaced budgets, the oracle still matches the same count.

    oracle_losses[j] is the loss after observing j links. With budgets=[1,3,5],
    indexing by array position would compare budget=5 against the oracle for two
    observations.
    """
    graph = toy(seed=7, n=8, m=9)
    z = uniform_vector(graph.n)
    ens = enumerate_worlds(graph, 3, alpha=0.85, z=z)
    sources = np.unique(ens.edge_source)
    budgets = [1, 3, 5]
    chosen, losses = greedy_probe(ens, max(budgets), sources)
    assert len(losses) == min(max(budgets), sources.size) + 1
    for b in budgets:
        ref = losses[min(b, len(losses) - 1)]
        direct = expected_loss(ens, chosen[:b])
        assert ref == pytest.approx(direct, abs=1e-12), b


def test_analytic_matches_exact_when_no_uncertainty():
    """With no uncertain link, the analytic and exact routes agree exactly."""
    g = UncertainGraph.from_edges(4, [(0, 1), (1, 2), (2, 3), (3, 0)], 1.0)
    z = uniform_vector(4)
    ens = enumerate_worlds(g, 2, alpha=0.85, z=z)
    loss, _ = expected_loss_analytic(g, 2, [], alpha=0.85, z=z, n_candidates=4, n_draws=1_000)
    assert expected_loss(ens, []) == pytest.approx(0.0, abs=1e-12)
    assert loss == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("tie_seed", [None, 1, 2, 3])
def test_analytic_report_equals_exact_true_set_under_full_observation(tie_seed):
    """Under full observation the analytic report **equals** the exact true set.

    "Zero risk against its own reported set" is too weak a check: unless the
    reported set itself matches the exact one, the links can be chosen well and
    the report still be wrong.

    In the world where every node is dangling, PageRank is exactly uniform, so
    this fails unless both routes break ties the same way.
    """
    n = 5
    graph = UncertainGraph.from_edges(
        n, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)], [0.4] * 5
    )
    z = uniform_vector(n)
    k = 2
    prio = (
        None if tie_seed is None
        else np.random.default_rng(tie_seed).permutation(n).astype(float)
    )
    ids = uncertain_edge_ids(graph)
    for states in ([0] * ids.size, [1] * ids.size, [1, 0, 1, 0, 1]):
        sub = condition_graph(graph, ids, states)
        ens = enumerate_worlds(sub, k, alpha=0.85, z=z, tie_priority=prio)
        d = decode(sub, k, alpha=0.85, z=z, n_candidates=n, n_draws=2_000,
                   tie_priority=prio)
        exact = set(np.flatnonzero(ens.topk[0]).tolist())
        assert set(d.report.tolist()) == exact, (states, tie_seed, exact, d.report)


def test_tie_ordering_uses_priority_not_roundoff():
    """Where scores are mathematically tied, tie_priority decides, not rounding.

    An earlier version used `0.2 + 1e-17` as the rounding error, but in float64
    that rounds back to `0.2` exactly, so a 1 ULP difference was never actually
    tested. `np.nextafter` produces genuinely distinct values instead.
    """
    from uncertain_pagerank.acquisition import _strict_topk

    y = np.full((1, 5), 0.2)
    for _ in range(3):  # three ULP up, far below the 1e-12 tolerance
        y[0, 2] = np.nextafter(y[0, 2], np.inf)
    assert y[0, 2] != 0.2, "the test itself failed to produce a 1 ULP difference"
    assert y[0, 2] - 0.2 < 1e-15

    prio = np.arange(5)[::-1].astype(float)
    ind, tied = _strict_topk(y, 2, prio)
    assert bool(tied[0])
    # node 2 has the larger value, but the gap is within tolerance, so the
    # priority decides
    assert set(np.flatnonzero(ind[0]).tolist()) == {3, 4}


def test_tie_quantization_does_not_swallow_real_differences():
    """A difference above the tolerance outranks tie_priority.

    Guards against the quantisation collapsing a real difference into a tie.
    """
    from uncertain_pagerank.acquisition import TIE_RTOL, _strict_topk

    y = np.full((1, 5), 0.2)
    y[0, 2] += 10.0 * TIE_RTOL  # clearly above the tolerance
    prio = np.arange(5)[::-1].astype(float)  # the priority prefers nodes 4, 3
    ind, _ = _strict_topk(y, 2, prio)
    # node 2 ranks last by priority but wins on the real difference
    assert 2 in np.flatnonzero(ind[0]).tolist()
    assert set(np.flatnonzero(ind[0]).tolist()) == {2, 4}
