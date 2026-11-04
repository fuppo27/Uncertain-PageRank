"""Regression tests for the source-level uncertainty attribution.

Do not name a helper `setup` or `teardown`: pytest 7.x treats those as
nose-style xUnit hooks and calls them, which errors out every test in the file.
"""
from __future__ import annotations

import numpy as np
import pytest

from uncertain_pagerank.attribution import (
    ContrastSet,
    analytic_attribution,
    boundary_contrasts,
    enumerate_world_scores,
    exact_attribution,
    exact_set_attribution,
    gap_stratified_contrasts,
    identity_contrast,
    interaction_term,
    linear_contrast_values,
    node_contrasts,
    resolvent_columns,
    source_positions,
    total_variance,
    true_contrast_values,
)
from uncertain_pagerank.expected_transition import expected_transition
from uncertain_pagerank.experiments.exp17_timescale_gate import build
from uncertain_pagerank.pagerank import pagerank, uniform_vector

ALPHA = 0.85
ER = {"topology": "er", "n": 10, "m": 10, "q": "beta", "q_mean": 0.6, "q_concentration": 3.0}
STAR = {"topology": "star", "n": 8, "m": 10, "q": "beta", "q_mean": 0.6, "q_concentration": 3.0}
PA = {"topology": "pa", "n": 8, "m_out": 1, "q": "beta", "q_mean": 0.5, "q_concentration": 2.0}
MIXED = {"topology": "er", "n": 10, "m": 13, "q": "mixed", "q_mean": 0.6,
         "q_concentration": 3.0, "certain_fraction": 0.25}


def _make_case(spec, seed=1, ks=(1, 2, 3)):
    """Build the graph, expected transition, p-average, contrast and all worlds."""
    g = build(spec, np.random.default_rng(seed))
    z = uniform_vector(g.n)
    et = expected_transition(g, method="dp")
    y0 = pagerank(et, alpha=ALPHA, z=z).scores
    cs = boundary_contrasts(y0, ks)
    ws = enumerate_world_scores(g, alpha=ALPHA, z=z)
    return g, z, et, y0, cs, ws


@pytest.mark.parametrize("spec", [ER, STAR, PA, MIXED])
def test_linear_surrogate_matches_analytic_attribution(spec):
    """The first gate: Psi_u^lin(C) equals Phi_hat_u(C) to machine precision.

    Without this, no later discrepancy can be told apart from an implementation
    error.
    """
    g, z, et, y0, cs, ws = _make_case(spec)
    zlin = linear_contrast_values(g, et, y0, cs, ws.bits, alpha=ALPHA, z=z)
    psi_lin = exact_attribution(ws.bits, ws.weights, zlin, ws.edge_source, g.n)
    phi = analytic_attribution(g, et, y0, cs, alpha=ALPHA, z=z)
    assert np.allclose(psi_lin, phi, rtol=1e-9, atol=1e-18)


@pytest.mark.parametrize("spec", [ER, STAR, PA, MIXED])
def test_linear_surrogate_sum_is_preserved(spec):
    """Within the first-order surrogate, the shares sum to the total."""
    g, z, et, y0, cs, ws = _make_case(spec)
    zlin = linear_contrast_values(g, et, y0, cs, ws.bits, alpha=ALPHA, z=z)
    psi_lin = exact_attribution(ws.bits, ws.weights, zlin, ws.edge_source, g.n)
    assert psi_lin.sum() == pytest.approx(total_variance(ws.weights, zlin), rel=1e-9)


@pytest.mark.parametrize("spec", [ER, STAR, PA, MIXED])
def test_law_of_total_variance_holds_per_source(spec):
    """Psi_u = Var(Z) - E[Var(Z | X_u)], checked by an independent computation."""
    g, z, et, y0, cs, ws = _make_case(spec)
    zt = true_contrast_values(ws.scores, cs)
    psi = exact_attribution(ws.bits, ws.weights, zt, ws.edge_source, g.n)
    var = total_variance(ws.weights, zt)
    for u in np.unique(ws.edge_source):
        pos = np.flatnonzero(ws.edge_source == u)
        code = (ws.bits[:, pos].astype(np.int64) << np.arange(pos.size)).sum(axis=1)
        ev = 0.0
        for c in np.unique(code):
            sel = code == c
            w = ws.weights[sel]
            p = w.sum()
            mu = (w @ zt[sel]) / p
            ev += float(w @ ((zt[sel] - mu) ** 2).sum(axis=1))
        assert psi[u] == pytest.approx(var - ev, rel=1e-9, abs=1e-18)


@pytest.mark.parametrize("spec", [ER, STAR, PA, MIXED])
def test_full_observation_recovers_total_variance(spec):
    """Observing every uncertain link makes Psi_S equal Var(Z).

    The conditional expectation becomes the realized value itself.
    """
    g, z, et, y0, cs, ws = _make_case(spec)
    zt = true_contrast_values(ws.scores, cs)
    allpos = np.arange(ws.bits.shape[1])
    got = exact_set_attribution(ws.bits, ws.weights, zt, allpos)
    assert got == pytest.approx(total_variance(ws.weights, zt), rel=1e-9)


def test_set_attribution_is_monotone_in_observations():
    """Psi_S never decreases as the observation set grows."""
    g, z, et, y0, cs, ws = _make_case(ER)
    zt = true_contrast_values(ws.scores, cs)
    srcs = list(np.unique(ws.edge_source))
    prev = 0.0
    for i in range(1, len(srcs) + 1):
        pos = source_positions(ws.edge_source, srcs[:i])
        if pos.size > 20:
            break
        cur = exact_set_attribution(ws.bits, ws.weights, zt, pos)
        assert cur >= prev - 1e-12
        prev = cur


@pytest.mark.parametrize("spec", [ER, STAR, PA, MIXED])
def test_attributions_are_nonnegative(spec):
    """Both Psi_u and Phi_hat_u are non-negative."""
    g, z, et, y0, cs, ws = _make_case(spec)
    zt = true_contrast_values(ws.scores, cs)
    psi = exact_attribution(ws.bits, ws.weights, zt, ws.edge_source, g.n)
    phi = analytic_attribution(g, et, y0, cs, alpha=ALPHA, z=z)
    assert (psi >= -1e-15).all()
    assert (phi >= -1e-15).all()


def test_sources_without_uncertain_edges_get_zero():
    """A source with no uncertain out-link contributes exactly zero."""
    g, z, et, y0, cs, ws = _make_case(MIXED)
    zt = true_contrast_values(ws.scores, cs)
    psi = exact_attribution(ws.bits, ws.weights, zt, ws.edge_source, g.n)
    phi = analytic_attribution(g, et, y0, cs, alpha=ALPHA, z=z)
    idle = np.setdiff1d(np.arange(g.n), np.unique(ws.edge_source))
    assert idle.size > 0, "this setting should contain a source with no uncertain link"
    assert np.all(psi[idle] == 0.0)


def test_identity_contrast_equals_per_node_variance_sum():
    """Psi_u(I) equals sum_v Var(E[y_v | X_u])."""
    g = build(ER, np.random.default_rng(1))
    z = uniform_vector(g.n)
    ws = enumerate_world_scores(g, alpha=ALPHA, z=z)
    cs = identity_contrast(g.n)
    zt = true_contrast_values(ws.scores, cs)
    assert np.allclose(zt, ws.scores)  # with C = I, Z is y itself
    psi = exact_attribution(ws.bits, ws.weights, zt, ws.edge_source, g.n)
    u = int(np.unique(ws.edge_source)[0])
    pos = np.flatnonzero(ws.edge_source == u)
    code = (ws.bits[:, pos].astype(np.int64) << np.arange(pos.size)).sum(axis=1)
    mean = ws.weights @ ws.scores
    acc = 0.0
    for c in np.unique(code):
        sel = code == c
        w = ws.weights[sel]
        p = w.sum()
        acc += p * float((((w @ ws.scores[sel]) / p - mean) ** 2).sum())
    assert psi[u] == pytest.approx(acc, rel=1e-9)


def test_interaction_term_is_zero_for_the_linear_surrogate():
    """I_inter is zero within the first-order surrogate; it is not for the true
    PageRank."""
    g, z, et, y0, cs, ws = _make_case(ER)
    zlin = linear_contrast_values(g, et, y0, cs, ws.bits, alpha=ALPHA, z=z)
    psi_lin = exact_attribution(ws.bits, ws.weights, zlin, ws.edge_source, g.n)
    assert interaction_term(ws.weights, zlin, psi_lin) == pytest.approx(0.0, abs=1e-14)

    zt = true_contrast_values(ws.scores, cs)
    psi = exact_attribution(ws.bits, ws.weights, zt, ws.edge_source, g.n)
    assert abs(interaction_term(ws.weights, zt, psi)) > 1e-12


def test_resolvent_solves_the_linear_system():
    """resolvent_columns satisfies (I - alpha Pbar) x = w."""
    g = build(ER, np.random.default_rng(1))
    z = uniform_vector(g.n)
    et = expected_transition(g, method="dp")
    rng = np.random.default_rng(0)
    w = rng.normal(size=(g.n, 3))
    x = resolvent_columns(et, w, alpha=ALPHA, z=z)
    lhs = x - ALPHA * (et.B @ x + np.outer(et.delta, z @ x))
    assert np.allclose(lhs, w, atol=1e-11)


def test_boundary_contrast_is_built_from_pavg_only():
    """The rank-boundary contrast follows from the p-average ranking alone and is
    normalised."""
    y0 = np.array([0.4, 0.1, 0.3, 0.2])
    cs = boundary_contrasts(y0, [1, 2])
    assert cs.r == 2
    assert np.allclose(np.linalg.norm(cs.W, axis=1), 1.0)
    # the margin between the first- and second-ranked nodes
    assert cs.W[0, 0] == pytest.approx(1 / np.sqrt(2))
    assert cs.W[0, 2] == pytest.approx(-1 / np.sqrt(2))
    assert cs.lam.sum() == pytest.approx(1.0)


def test_contrast_set_rejects_bad_input():
    with pytest.raises(ValueError):
        ContrastSet(W=np.zeros((2, 3)), lam=np.ones(3), labels=["a", "b"])
    with pytest.raises(ValueError):
        ContrastSet(W=np.zeros((2, 3)), lam=np.array([-1.0, 1.0]), labels=["a", "b"])
    with pytest.raises(ValueError):
        ContrastSet(W=np.zeros((2, 3)), lam=np.ones(2), labels=["a"])


def test_auxiliary_contrast_sets_are_well_formed():
    y0 = np.linspace(0.3, 0.01, 12)
    for cs in (node_contrasts([0, 3, 5], 12), gap_stratified_contrasts(y0, 2),
               identity_contrast(12)):
        assert cs.W.shape[1] == 12
        assert cs.lam.shape == (cs.r,)
        assert np.all(cs.lam >= 0)
        assert cs.C.shape == cs.W.shape


def test_batch_solver_matches_sequential_pagerank():
    """The batched solver agrees with the sequential one and returns distributions."""
    from uncertain_pagerank.sampling import WorldSampler
    from uncertain_pagerank.world_batch import batch_world_scores, sample_masks

    for spec in (ER, {"topology": "er", "n": 40, "m": 110, "q": "beta",
                      "q_mean": 0.6, "q_concentration": 3.0}):
        g = build(spec, np.random.default_rng(1))
        z = uniform_vector(g.n)
        masks = sample_masks(g, 25, np.random.default_rng(5))
        bat = batch_world_scores(g, masks, alpha=ALPHA, z=z, tol=1e-13)
        smp = WorldSampler(g, np.random.default_rng(0))
        seq = np.array([
            pagerank(smp.transition(masks[i]), alpha=ALPHA, z=z, tol=1e-13).scores
            for i in range(masks.shape[0])
        ])
        assert np.allclose(bat, seq, atol=1e-11)
        assert np.allclose(bat.sum(axis=1), 1.0)


def test_sample_masks_respects_certain_edges():
    """A link with q=1 is always connected and one with q=0 never is."""
    from uncertain_pagerank.world_batch import sample_masks

    g = build(MIXED, np.random.default_rng(1))
    masks = sample_masks(g, 200, np.random.default_rng(0))
    assert np.all(masks[:, g.probs >= 1.0] == 1.0)
    if np.any(g.probs <= 0.0):
        assert np.all(masks[:, g.probs <= 0.0] == 0.0)


def test_pickfreeze_approaches_the_exact_attribution():
    """The pick-freeze estimate approaches the exact value as the sample grows."""
    from uncertain_pagerank.attribution import pickfreeze_attribution

    g, z, et, y0, cs, ws = _make_case(ER)
    zt = true_contrast_values(ws.scores, cs)
    psi = exact_attribution(ws.bits, ws.weights, zt, ws.edge_source, g.n)
    active = np.unique(ws.edge_source)

    # The error of a single seed is not monotone in N, so average over seeds
    errs = []
    for n in (500, 8000):
        e = []
        for seed in (3, 4, 5):
            r = pickfreeze_attribution(g, cs, sources=active, n_samples=n, seed=seed,
                                       alpha=ALPHA, z=z)
            e.append(np.abs(r.psi - psi[active]).max() / psi[active].max())
            assert r.n_solves == n * (1 + active.size)
        errs.append(float(np.mean(e)))
    assert errs[1] < errs[0]
    assert errs[1] < 0.15


def test_pickfreeze_is_zero_without_uncertainty():
    """With no uncertain link, the estimate of Psi_u is zero."""
    from uncertain_pagerank.attribution import pickfreeze_attribution

    g = build(ER, np.random.default_rng(1))
    certain = g.with_probs(np.ones_like(g.probs))
    z = uniform_vector(g.n)
    et = expected_transition(certain, method="dp")
    y0 = pagerank(et, alpha=ALPHA, z=z).scores
    cs = boundary_contrasts(y0, [1, 2])
    r = pickfreeze_attribution(certain, cs, sources=np.arange(g.n), n_samples=64,
                               seed=0, alpha=ALPHA, z=z)
    assert np.allclose(r.psi, 0.0, atol=1e-15)
    assert r.var_z == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("spec", [ER, PA])
def test_shapley_effects_sum_to_total_variance(spec):
    """The Shapley effects sum to tr Cov(Z) exactly.

    The first-order Psi_u falls short by the interaction, which is what
    separates the two.
    """
    from uncertain_pagerank.attribution import shapley_effects

    g, z, et, y0, cs, ws = _make_case(spec)
    zt = true_contrast_values(ws.scores, cs)
    active = np.unique(ws.edge_source)
    sh = shapley_effects(ws.bits, ws.weights, zt, ws.edge_source, active)
    var = total_variance(ws.weights, zt)
    assert sh.sum() == pytest.approx(var, rel=1e-9)

    psi = exact_attribution(ws.bits, ws.weights, zt, ws.edge_source, g.n)
    assert psi[active].sum() < var  # the first order falls short by the interaction


def test_shapley_effects_are_nonnegative():
    """The characteristic function is monotone, so the Shapley effects are
    non-negative."""
    from uncertain_pagerank.attribution import shapley_effects

    g, z, et, y0, cs, ws = _make_case(ER)
    zt = true_contrast_values(ws.scores, cs)
    active = np.unique(ws.edge_source)
    sh = shapley_effects(ws.bits, ws.weights, zt, ws.edge_source, active)
    assert (sh >= -1e-15).all()


def test_shapley_equals_psi_for_the_linear_surrogate():
    """The first-order surrogate has no interaction, so Shapley equals Psi."""
    from uncertain_pagerank.attribution import shapley_effects

    g, z, et, y0, cs, ws = _make_case(ER)
    zlin = linear_contrast_values(g, et, y0, cs, ws.bits, alpha=ALPHA, z=z)
    active = np.unique(ws.edge_source)
    sh = shapley_effects(ws.bits, ws.weights, zlin, ws.edge_source, active)
    psi = exact_attribution(ws.bits, ws.weights, zlin, ws.edge_source, g.n)
    assert np.allclose(sh, psi[active], rtol=1e-8, atol=1e-18)


def test_selfloop_dangling_transition_is_row_stochastic():
    """The expected transition matrix under the stay-put convention is
    row-stochastic."""
    from uncertain_pagerank.attribution import selfloop_dangling_transition

    g = build(ER, np.random.default_rng(1))
    et = expected_transition(g, method="dp")
    mod = selfloop_dangling_transition(et)
    assert np.allclose(mod.delta, 0.0)
    assert np.allclose(np.asarray(mod.B.sum(axis=1)).ravel(), 1.0)
    # Rows with K_u>0 are unchanged: nodes of zero dangling probability
    live = et.delta == 0
    assert np.allclose(np.asarray(mod.B[live].todense()),
                       np.asarray(et.B[live].todense()))


@pytest.mark.parametrize("spec", [ER, PA, STAR])
def test_selfloop_dangling_attribution_matches_its_linear_surrogate(spec):
    """Psi_u^lin == Phi_hat_u holds under the other dangling convention too.

    Changing the convention turns the dangling branch of the variance from
    z^T x into x_u. Passing the first gate under the new convention is what
    confirms that substitution.
    """
    from uncertain_pagerank.attribution import (
        attribution_selfloop_dangling, selfloop_dangling_transition)
    from uncertain_pagerank.possible_worlds import _world_chunks

    g = build(spec, np.random.default_rng(1))
    z = uniform_vector(g.n)
    et = expected_transition(g, method="dp")
    mod = selfloop_dangling_transition(et)
    y0 = pagerank(mod, alpha=ALPHA, z=z, tol=1e-13).scores
    cs = boundary_contrasts(y0, [1, 2, 3])
    phi = attribution_selfloop_dangling(g, et, y0, cs, alpha=ALPHA, z=z)

    edges, probs = g.edge_list(), g.probs
    isc = probs >= 1.0
    isu = ~(isc | (probs <= 0.0))
    certain, uncertain, qu = edges[isc], edges[isu], probs[isu]
    x = resolvent_columns(mod, cs.W.T, alpha=ALPHA, z=z)
    sbar = et.B @ x + et.delta[:, None] * x
    base, scale = cs.W @ y0, np.sqrt(cs.lam)
    bits = next(_world_chunks(int(uncertain.shape[0]), 1 << int(uncertain.shape[0])))
    w = np.prod(np.where(bits > 0, qu, 1.0 - qu), axis=1)

    a = np.zeros((bits.shape[0], g.n, g.n))
    if certain.size:
        a[:, certain[:, 0], certain[:, 1]] = 1.0
    a[:, uncertain[:, 0], uncertain[:, 1]] = bits
    deg = a.sum(axis=2)
    live = deg > 0
    p = np.where(live[:, :, None], a / np.where(live, deg, 1.0)[:, :, None], 0.0)
    idx = np.arange(g.n)
    p[:, idx, idx] = np.where(live[:, idx], p[:, idx, idx], 1.0)
    assert np.allclose(p.sum(axis=2), 1.0)

    s = np.matmul(p, x)
    zlin = (base[None, :] + ALPHA * np.einsum("u,cur->cr", y0, s - sbar[None, :, :])) \
        * scale[None, :]
    psi = exact_attribution(bits.astype(np.int8), w, zlin, edges[isu][:, 0], g.n)
    assert np.allclose(psi, phi, rtol=1e-8, atol=1e-18)


def test_owen_estimator_is_consistent_with_exact_attribution():
    """Owen's Correlation 2 estimate approaches the exact value."""
    from uncertain_pagerank.attribution import pickfreeze_attribution_owen

    g, z, et, y0, cs, ws = _make_case(ER)
    zt = true_contrast_values(ws.scores, cs)
    psi = exact_attribution(ws.bits, ws.weights, zt, ws.edge_source, g.n)
    active = np.unique(ws.edge_source)
    errs = []
    for n in (500, 6000):
        e = []
        for seed in (5, 6, 7):
            r = pickfreeze_attribution_owen(g, cs, sources=active, n_samples=n,
                                            seed=seed, alpha=ALPHA, z=z)
            e.append(np.abs(r.psi - psi[active]).max() / psi[active].max())
            # two per sample, plus 2N for the shared part
            assert r.n_solves == 2 * n * (1 + active.size)
        errs.append(float(np.mean(e)))
    assert errs[1] < errs[0]
    assert errs[1] < 0.10


def test_tied_spearman_ignores_a_machine_precision_ordering():
    """A tie broken by rounding noise must not move the reported correlation.

    In six of experiment 24's 75 conditions a pair of sources is exchanged by
    an automorphism of the weighted graph, so those two effects are exactly
    equal in exact arithmetic and differ by 4e-18 to 1.3e-15 of the largest
    once computed. The closest gap that is not such a tie is 1.0e-5 of the
    largest, four orders above the tolerance, so collapsing them is safe.
    """
    from scipy.stats import spearmanr

    from uncertain_pagerank.experiments.exp24_shapley import tied_ranks, tied_spearman

    eps = 1e-15
    assert 1.0 + eps != 1.0, "the perturbation has to survive float64"

    # a near-tie collapses, a real gap of the same absolute size does not
    out = tied_ranks(np.array([1.0, 1.0 + eps, 2.0, 2.5]))
    assert out[0] == out[1]
    assert out[2] != out[3]

    # the tolerance is relative to the largest magnitude
    small = tied_ranks(np.array([1e-20, 2e-20, 1.0]))
    assert small[0] == small[1]

    # the same values with the noise swapped: plain Spearman is fooled
    a = np.array([1.0, 1.0 + eps, 3.0, 4.0])
    b = np.array([1.0 + eps, 1.0, 3.0, 4.0])
    assert spearmanr(a, b).statistic < 1.0
    assert tied_spearman(a, b) == 1.0

    # a genuine reordering is still reported
    assert tied_spearman(np.array([1.0, 2.0, 3.0, 4.0]),
                         np.array([2.0, 1.0, 3.0, 4.0])) < 1.0
