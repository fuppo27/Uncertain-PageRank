"""Per-source validation on real networks.

An earlier attempt could only validate per block of sources, because the naive
pick-freeze design gave a per-source reference whose split-half agreement was
0.076: unusable. Owen's Correlation 2 was found at medium scale to give a usable
reference at the same cost, so this checks whether a per-source reference can be
built on real data too.

The reference is two independent Owen runs, whose split-half agreement measures
the noise ceiling. The naive estimator is run alongside at **the same number of
solves**, to see whether the gap between estimators shows up on real data.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from uncertain_pagerank.attribution import (
    ContrastSet, analytic_attribution, boundary_contrasts,
    pickfreeze_attribution, pickfreeze_attribution_owen, resolvent_columns,
    trace_row_covariance)
from uncertain_pagerank.datasets import load_dlbl, load_edgelist
from uncertain_pagerank.expected_transition import expected_transition
from uncertain_pagerank.pagerank import pagerank, uniform_vector
from uncertain_pagerank.variance import variance_attribution

NETS = {"trackback": ("Trackback NW", "data/raw/original/trackback/blog34.dlbl", load_dlbl),
        "wikivote": ("wiki-Vote", "data/raw/snap/wiki-Vote.txt.gz", load_edgelist)}


class _Saved:
    """The Monte Carlo columns read from a saved CSV, shaped like an estimator's
    return value."""

    def __init__(self, psi, se, seconds=float("nan"), n_solves=0):
        self.psi = psi
        self.psi_lo = psi - 1.96 * se
        self.psi_hi = psi + 1.96 * se
        self.seconds = seconds
        self.n_solves = n_solves


def _load_mc(path, cand, stratum_of):
    """Read the Monte Carlo columns from a saved CSV, to rebuild only the
    analytic ones.

    Returns them only after checking that the candidate set and the strata match
    the saved ones; otherwise the reference cannot be reused and this stops.
    """
    import csv as _csv

    rows = {int(r["source"]): r for r in _csv.DictReader(path.open(encoding="utf-8"))}
    want = {int(u) for u in cand}
    if set(rows) != want:
        raise SystemExit(f"the candidate set differs from the saved one: "
                         f"only in saved {sorted(set(rows)-want)}, "
                         f"only in new {sorted(want-set(rows))}. "
                         f"--reuse-mc cannot be used")
    for u in cand:
        if rows[int(u)]["stratum"] != stratum_of.get(int(u), "?"):
            raise SystemExit(f"the stratum of source {int(u)} differs from the "
                             f"saved one. --reuse-mc cannot be used")
    col = lambda k: np.array([float(rows[int(u)][k]) for u in cand])
    se = col("psi_se")
    return (_Saved(col("psi_a"), se), _Saved(col("psi_b"), se),
            _Saved(col("mc_naive"), se), col("psi_ref"), se)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True, choices=sorted(NETS))
    ap.add_argument("--candidates", type=int, default=24)
    ap.add_argument("--n-owen", type=int, default=400)
    ap.add_argument("--strata", type=int, default=8)
    ap.add_argument("--reuse-mc", action="store_true",
                    help="reuse the Monte Carlo columns of the saved CSV "
                         "(psi_ref/psi_se/psi_a/psi_b/mc_naive) and rebuild only "
                         "the analytic ones. Checks first that the candidate set "
                         "is unchanged. The reference takes hours at 40,000 "
                         "solves twice over, so this suffices when only the "
                         "analytic side changed")
    a = ap.parse_args()

    name, path, loader = NETS[a.net]
    g0 = loader(REPO / path).graph
    g = g0.with_probs(np.random.default_rng(20260923).beta(7, 3, size=g0.m))
    z = uniform_vector(g.n)
    et = expected_transition(g, method="dp")
    y0 = pagerank(et, alpha=0.85, z=z, tol=1e-12).scores
    cs = boundary_contrasts(y0, [10, 100, 1000])

    src = np.repeat(np.arange(g.n), g.out_degrees())
    unc = np.unique(src[(g.probs > 0) & (g.probs < 1)])

    t = time.perf_counter()
    phi = analytic_attribution(g, et, y0, cs, alpha=0.85, z=z)
    t_phi = time.perf_counter() - t

    # Candidates are drawn from four strata. Validating only the proposal's own
    # top sources answers whether predicted-important sources are important, but
    # cannot reveal omissions; random sampling alone would rarely include an
    # important source. The strata are reported separately.
    rng = np.random.default_rng(42)
    per = max(1, a.candidates // 4)
    deg = g.out_degrees()
    rank_phi = {int(u): i for i, u in enumerate(unc[np.argsort(-phi[unc], kind="stable")])}
    rank_pr = {int(u): i for i, u in enumerate(unc[np.argsort(-y0[unc], kind="stable")])}
    hi_phi = unc[np.argsort(-phi[unc], kind="stable")][:per]
    # Sources that rank high by PageRank but not by the attribution
    pr_order = unc[np.argsort(-y0[unc], kind="stable")]
    gap = [u for u in pr_order if rank_phi[int(u)] > 10 * max(rank_pr[int(u)], 1)]
    hi_pr_lo_phi = np.array(gap[:per], dtype=np.int64)
    lo_deg = rng.choice(unc[deg[unc] <= max(1, int(np.median(deg[unc])))],
                        size=per, replace=False)
    strata = {"top Phi": hi_phi, "high PR low Phi": hi_pr_lo_phi, "low out-degree": lo_deg}
    chosen = np.unique(np.concatenate([v for v in strata.values() if v.size]))
    pool = np.setdiff1d(unc, chosen)
    rnd = rng.choice(pool, size=min(a.candidates - chosen.size, pool.size), replace=False)
    strata["random"] = rnd
    cand = np.unique(np.concatenate([chosen, rnd]))
    stratum_of = {}
    for lab, arr in strata.items():
        for u in np.atleast_1d(arr):
            stratum_of.setdefault(int(u), lab)
    print(f"=== {name}: n={g.n} m={g.m} sources={unc.size} "
          f"candidates={cand.size} ===", flush=True)
    for lab, arr in strata.items():
        print(f"    stratum {lab:18s} {np.atleast_1d(arr).size:3d}", flush=True)

    if a.reuse_mc:
        ra, rb, na, ref, se = _load_mc(REPO / "outputs" / f"real_per_source_{a.net}.csv",
                                       cand, stratum_of)
        n_naive = 0
        print("  reusing the saved Monte Carlo columns", flush=True)
    else:
        ra = pickfreeze_attribution_owen(g, cs, sources=cand, n_samples=a.n_owen, seed=1,
                                         alpha=0.85, z=z, tol=1e-8, chunk=256, n_boot=400)
        print(f"  reference A done, {ra.seconds:.0f}s, {ra.n_solves} solves", flush=True)
        rb = pickfreeze_attribution_owen(g, cs, sources=cand, n_samples=a.n_owen, seed=2,
                                         alpha=0.85, z=z, tol=1e-8, chunk=256)
        print(f"  reference B done, {rb.seconds:.0f}s", flush=True)
    ref = (ra.psi + rb.psi) / 2.0
    sh = float(spearmanr(ra.psi, rb.psi).statistic)
    scale = float(np.abs(ref).max())
    print(f"  Owen reference split-half = {sh:.4f}  relative CI width "
          f"{np.nanmean(ra.psi_hi-ra.psi_lo)/scale:.3f}",
          flush=True)

    # The naive estimator at the same number of solves
    if not a.reuse_mc:
        n_naive = max(1, (2 * a.n_owen * (1 + cand.size)) // (1 + cand.size))
        na = pickfreeze_attribution(g, cs, sources=cand, n_samples=n_naive, seed=1,
                                    alpha=0.85, z=z, tol=1e-8, chunk=256)
        print(f"  naive (N={n_naive}, {na.n_solves} solves) done, "
              f"{na.seconds:.0f}s", flush=True)

    # Ablations and the cheap baselines
    arng = np.random.default_rng(123)
    wr = arng.normal(size=cs.W.shape); wr /= np.linalg.norm(wr, axis=1, keepdims=True)
    phi_rand = analytic_attribution(g, et, y0, ContrastSet(W=wr, lam=cs.lam,
                                    labels=["r"] * cs.r), alpha=0.85, z=z)
    xk = resolvent_columns(et, cs.W.T, alpha=0.85, z=z)
    phi_nov = np.zeros(g.n)
    for k in range(cs.r):
        phi_nov += cs.lam[k] * variance_attribution(g, et, np.ones(g.n), xk[:, k],
                                                    alpha=0.85, z=z).per_source
    q = g.probs[(g.probs > 0) & (g.probs < 1)]
    su = src[(g.probs > 0) & (g.probs < 1)]
    qv = np.bincount(su, weights=q * (1 - q), minlength=g.n)
    trc = trace_row_covariance(g, et, z)

    # The breakdown by stratum. Sources the budget could not resolve are kept.
    if not a.reuse_mc:
        se = (ra.psi_hi - ra.psi_lo) / 3.92
    print("\n  reference by stratum (unresolved sources kept)")
    print(f"  {'stratum':18s} {'n':>3s} {'median |Psi|':>13s} {'median SE':>11s} "
          f"{'|Psi|>2SE':>10s}")
    for lab in strata:
        m = np.array([stratum_of.get(int(u)) == lab for u in cand])
        if not m.any():
            continue
        print(f"  {lab:18s} {int(m.sum()):4d} {np.median(np.abs(ref[m])):12.3e} "
              f"{np.median(se[m]):10.3e} {int((np.abs(ref[m]) > 2*se[m]).sum()):10d}", flush=True)
    print(f"  negative estimates, as evidence that none were clipped: "
          f"{int((ra.psi<0).sum())} of {cand.size} in reference A")

    # Save the raw values for the figures: reference, standard error, the score
    # of each method, and the stratum
    import csv
    outp = REPO / "outputs" / f"real_per_source_{a.net}.csv"
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", newline="", encoding="utf-8") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["source", "stratum", "psi_ref", "psi_se", "psi_a", "psi_b",
                      "phi_hat", "mc_naive", "pagerank_src", "y0sq_x_qvar",
                      "y0sq_x_trcov", "phi_no_y0sq", "phi_randdir", "out_degree"])
        for j, u in enumerate(cand):
            wtr.writerow([int(u), stratum_of.get(int(u), "?"), ref[j], se[j],
                          ra.psi[j], rb.psi[j], phi[u], na.psi[j], y0[u],
                          (y0**2 * qv)[u], (y0**2 * trc)[u], phi_nov[u],
                          phi_rand[u], int(g.out_degrees()[u])])
    print(f"  raw values saved to {outp}", flush=True)

    print(f"\n  {'method':16s} {'rho vs ref':>11s} {'sec':>9s} {'solves':>9s}")
    for nm, v, secs, nsv in [
        ("phi_hat", phi[cand], t_phi, 0),
        ("mc_naive (same cost)", na.psi, na.seconds, na.n_solves),
        ("y0sq_x_trcov", (y0**2 * trc)[cand], 0.0, 0),
        ("y0sq_x_qvar", (y0**2 * qv)[cand], 0.0, 0),
        ("phi_no_y0sq", phi_nov[cand], 0.0, 0),
        ("phi_randdir", phi_rand[cand], 0.0, 0),
        ("pagerank_src", y0[cand], 0.0, 0),
        ("q_variance", qv[cand], 0.0, 0),
    ]:
        print(f"  {nm:16s} {spearmanr(ref, v).statistic:12.4f} {secs:9.2f} {nsv:9d}",
              flush=True)


if __name__ == "__main__":
    main()
