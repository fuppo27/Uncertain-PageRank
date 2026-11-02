"""Regression tests for the Monte Carlo decoder.

The top-k set of each sample must be cut by the **same deterministic rule** as
the exact T(X). Using metrics.topk_indicator, which splits ties evenly, trains a
different target, and then the decoder does not converge to the exact Bayes
decoder however many samples are drawn. On the star that is fatal: worlds tied at
the top-k boundary carry 0.70 to 0.95 of the probability mass.
"""

import numpy as np
import pytest

from uncertain_pagerank import uniform_vector
from uncertain_pagerank.acquisition import _strict_topk, enumerate_worlds, expected_loss
from uncertain_pagerank.acquisition_scalable import (
    positions_to_edge_ids,
    uncertain_edge_ids,
)
from uncertain_pagerank.experiments.exp17_timescale_gate import build
from uncertain_pagerank.experiments.exp21_mc_decoder import (
    mc_decode,
    mc_operational_loss,
)

STAR = {"topology": "star", "n": 16, "m": 12, "q": "beta",
        "q_mean": 0.6, "q_concentration": 3.0}
ER = {"topology": "er", "n": 16, "m": 12, "q": "beta",
      "q_mean": 0.6, "q_concentration": 3.0}


def _make_case(spec, seed=1, k=3, alpha=0.85):
    g = build(spec, np.random.default_rng(seed))
    z = uniform_vector(g.n)
    return g, z, k, alpha, enumerate_worlds(g, k, alpha=alpha, z=z)


@pytest.mark.parametrize("spec", [ER, STAR])
def test_full_observation_gives_zero_mc_loss(spec):
    """Observing every uncertain link fixes the world, so even the MC decoder
    attains zero loss."""
    g, z, k, alpha, ens = _make_case(spec)
    pick = list(range(ens.uncertain_edges.size))
    loss, _ = mc_operational_loss(ens, g, k, pick, alpha, z, 50, seed=1, tol=1e-12)
    assert loss == pytest.approx(0.0, abs=1e-12)


def test_mc_membership_converges_to_exact_on_tie_heavy_star():
    """Even on the tie-heavy star, more samples converge to the exact membership."""
    g, z, k, alpha, ens = _make_case(STAR)
    assert ens.tied_mass > 0.5, "this check assumes a tie-heavy setting"
    exact = ens.marginal
    errors = []
    for n_mc in (100, 2_000):
        acc = np.zeros(g.n)
        reps = 5
        for rep in range(reps):
            from uncertain_pagerank.sampling import WorldSampler
            from uncertain_pagerank.pagerank import pagerank

            sampler = WorldSampler(g, np.random.default_rng(rep))
            tally = np.zeros(g.n)
            for _ in range(n_mc):
                y = pagerank(
                    sampler.transition(sampler.sample_mask()), alpha=alpha, z=z, tol=1e-12
                ).scores
                tally += _strict_topk(y[None, :], k, None)[0][0]
            acc += tally / n_mc
        errors.append(float(np.abs(acc / reps - exact).mean()))
    assert errors[1] < errors[0], errors
    assert errors[1] < 0.02, errors


@pytest.mark.parametrize("tie_seed", [None, 1, 2])
def test_mc_report_matches_exact_under_full_observation(tie_seed):
    """Under full observation the MC report matches the exact set for any tie
    priority."""
    g = build(STAR, np.random.default_rng(1))
    z = uniform_vector(g.n)
    k, alpha = 3, 0.85
    prio = (None if tie_seed is None
            else np.random.default_rng(tie_seed).permutation(g.n).astype(float))
    ens = enumerate_worlds(g, k, alpha=alpha, z=z, tie_priority=prio)
    ids = uncertain_edge_ids(g)
    for states in ([0] * ids.size, [1] * ids.size):
        report, _ = mc_decode(g, k, ids, np.array(states), alpha, z, 50,
                              seed=1, tol=1e-12, tie_priority=prio)
        # The world is fixed, so this must equal that world's true top-k set
        match = np.all(ens.bits == np.array(states)[None, :], axis=1)
        exact = set(np.flatnonzero(ens.topk[np.flatnonzero(match)[0]]).tolist())
        assert set(report.tolist()) == exact, (states, tie_seed)


def test_result_is_invariant_to_pick_order():
    """For the same set of links, the order of selection does not matter."""
    g, z, k, alpha, ens = _make_case(ER)
    a, _ = mc_operational_loss(ens, g, k, [3, 1, 7], alpha, z, 200, seed=5, tol=1e-12)
    b, _ = mc_operational_loss(ens, g, k, [7, 3, 1], alpha, z, 200, seed=5, tol=1e-12)
    assert a == pytest.approx(b, abs=1e-12)


def test_mc_loss_is_at_least_the_bayes_optimum():
    """The MC decoder never beats the exact Bayes decoder on the same
    observation set."""
    g, z, k, alpha, ens = _make_case(ER)
    for pick in ([0], [0, 2], [1, 3, 5]):
        star = expected_loss(ens, edges=pick)
        mc, _ = mc_operational_loss(ens, g, k, pick, alpha, z, 500, seed=2, tol=1e-12)
        assert mc >= star - 1e-12, (pick, star, mc)
