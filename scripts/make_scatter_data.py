"""Per-source data behind the exact-vs-analytic scatter plot.

The aggregate output of exp22 records one row per (condition, method) and so
carries rank correlations, not the per-source values the scatter needs. This
script re-runs the enumeration for a legible subset of the exp22 conditions and
writes one row per (condition, source), with both quantities normalised by the
total variance of the contrast.

The subset is five of the six exp22 generators and the first three of its five
seeds; the omitted generator (mixed certain and uncertain links) behaves like the
other two Erdos-Renyi ones. Nothing here is selected after seeing the result.

Usage:
    python scripts/make_scatter_data.py
"""
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from uncertain_pagerank.attribution import (
    analytic_attribution, enumerate_world_scores, exact_attribution,
    total_variance, true_contrast_values)
from uncertain_pagerank.expected_transition import expected_transition
from uncertain_pagerank.experiments.exp17_timescale_gate import build
from uncertain_pagerank.experiments.exp22_source_attribution import make_contrasts
from uncertain_pagerank.pagerank import pagerank, uniform_vector

# Short names used in the figure legend, keyed by the exp22 generator label.
# The omitted label is dropped from the scatter to keep it legible.
SHORT = {"ER n=16 Beta": "ER",
         "ER n=16 uniform 0.5": "ER-unif",
         "Preferential attachment n=16": "PA",
         "Cycle with chords": "Cycle",
         "Star": "Star"}
# The contrast families, named as the figure legend names them.
CONTRAST = {"boundary": "boundary", "node_top": "node",
            "identity": "identity", "gap_stratified": "gap"}
SEEDS = (1, 2, 3)


def main() -> None:
    cfg = yaml.safe_load(
        (REPO / "configs/experiment/exp22_source_attribution.yaml").read_text(
            encoding="utf-8"))
    alpha = float(cfg["alpha"])
    ks = [int(k) for k in cfg["boundary_ks"]]

    rows = []
    for spec in cfg["graphs"]:
        short = SHORT.get(spec["label"])
        if short is None:
            continue
        for seed in SEEDS:
            graph = build(spec, np.random.default_rng(int(seed)))
            z = uniform_vector(graph.n)
            et = expected_transition(graph, method="dp")
            y0 = pagerank(et, alpha=alpha, z=z).scores
            ws = enumerate_world_scores(graph, alpha=alpha, z=z)
            active = np.unique(ws.edge_source)

            for cname, legend in CONTRAST.items():
                cs = make_contrasts(cname, y0, ks, np.random.default_rng(int(seed) + 11))
                zt = true_contrast_values(ws.scores, cs)
                var = total_variance(ws.weights, zt)
                if var <= 0:
                    continue
                psi = exact_attribution(ws.bits, ws.weights, zt, ws.edge_source, graph.n)
                phi = analytic_attribution(graph, et, y0, cs, alpha=alpha, z=z)
                for u in active:
                    rows.append((short, int(seed), legend, int(u),
                                 float(psi[u] / var), float(phi[u] / var)))
            print(f"  {short:8s} seed={seed}  {active.size} active sources", flush=True)

    out = REPO / "outputs/fig_exact_vs_analytic.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["generator", "seed", "contrast", "source",
                    "psi_over_V", "phi_over_V"])
        w.writerows(rows)
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
