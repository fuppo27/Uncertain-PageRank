"""Compare naive pick-freeze against Owen's "Correlation 2" at equal cost.

The setting is the one where the exp23 reference broke down: preferential
attachment at n=200, split-half 0.44. With no ground truth available, the
usability of a reference is measured by the rank correlation between two
independent runs of it (split-half). Owen costs twice as many solves per sample,
so its sample count is halved to match.
"""
import sys, time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from uncertain_pagerank.attribution import (
    boundary_contrasts, pickfreeze_attribution, pickfreeze_attribution_owen,
    analytic_attribution)
from uncertain_pagerank.expected_transition import expected_transition
from uncertain_pagerank.pagerank import pagerank, uniform_vector
from uncertain_pagerank.experiments.exp17_timescale_gate import build

SPECS = [
    ("Preferential attachment n=200",
     {"topology":"pa","n":200,"m_out":3,"q":"beta","q_mean":0.5,"q_concentration":2.0}),
    ("ER n=200", {"topology":"er","n":200,"m":600,"q":"beta","q_mean":0.6,"q_concentration":3.0}),
]
N_CAND, N_NAIVE = 40, 2000          # Owen matches the cost at half the samples

def main() -> None:
    for label, spec in SPECS:
        g = build(spec, np.random.default_rng(1))
        z = uniform_vector(g.n)
        et = expected_transition(g, method="dp")
        y0 = pagerank(et, alpha=0.85, z=z, tol=1e-12).scores
        cs = boundary_contrasts(y0, [1, 2, 3])
        src = np.unique(np.repeat(np.arange(g.n), g.out_degrees())[
            (g.probs > 0) & (g.probs < 1)])
        rng = np.random.default_rng(4242)
        cand = np.sort(rng.choice(src, size=min(N_CAND, src.size), replace=False))
        phi = analytic_attribution(g, et, y0, cs, alpha=0.85, z=z)[cand]
        print(f"=== {label} ({src.size} sources, {cand.size} candidates) ===", flush=True)
        print(f"{'estimator':>12s} {'N':>6s} {'split-half':>11s} {'solves':>9s} "
              f"{'sec':>7s} {'rho vs phi':>11s}")
        for name, fn, N in [("naive", pickfreeze_attribution, N_NAIVE),
                            ("Owen Corr2", pickfreeze_attribution_owen, N_NAIVE // 2)]:
            t = time.perf_counter()
            a = fn(g, cs, sources=cand, n_samples=N, seed=11, alpha=0.85, z=z, tol=1e-8)
            b = fn(g, cs, sources=cand, n_samples=N, seed=22, alpha=0.85, z=z, tol=1e-8)
            dt = time.perf_counter() - t
            sh = spearmanr(a.psi, b.psi).statistic
            ref = (a.psi + b.psi) / 2
            print(f"{name:>12s} {N:6d} {sh:11.4f} {a.n_solves + b.n_solves:9d} "
                  f"{dt:7.0f} {spearmanr(ref, phi).statistic:11.4f}", flush=True)
        print(flush=True)

if __name__ == "__main__":
    main()
