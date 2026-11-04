# Auditable PageRank on Uncertain Graphs

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23938199.svg)](https://doi.org/10.5281/zenodo.23938199)

Code and result data for *Auditable PageRank on Uncertain Graphs: Exact Row Moments and
Source-Level Attribution of Ranking Uncertainty*.

The method allocates the uncertainty of a fixed contrast of the PageRank vector (for
example a top-$k$ rank margin) over the source nodes whose uncertain out-links generate
it. See `DATA.md` for what data is and is not included here, and why.

## Install

```bash
python -m pip install -e ".[dev]"      # numpy, scipy, pandas, matplotlib, pyyaml
```

Python 3.12 with NumPy 2.5 and SciPy 1.18 was used for all reported runs. The package has
no GPU or deep-learning dependencies. `make help` lists the common targets.

## Layout

| Path | Contents |
|---|---|
| `src/uncertain_pagerank/` | Library: exact moments, attribution, estimators |
| `src/uncertain_pagerank/experiments/` | Experiment drivers (`expNN_*.py`) |
| `configs/experiment/` | One YAML per experiment, fixing seeds and settings |
| `scripts/` | Aggregation, figures, sensitivity analyses, data download |
| `outputs/` | Result CSVs for the tables and figures in the paper |
| `tests/unit/` | Regression tests, including the machine-precision gates |

## Reproducing the paper

Public data first:

```bash
python scripts/download_snap.py        # wiki-Vote, soc-Epinions1, ... (see DATA.md)
```

Then, by section of the paper:

| Paper | Command |
|---|---|
| Enumeration-scale validation | `python -m uncertain_pagerank.experiments.exp22_source_attribution --config configs/experiment/exp22_source_attribution.yaml` |
| Shapley effects | `... exp24_shapley --config configs/experiment/exp24_shapley.yaml` |
| Medium scale | `... exp23_medium_scale --config configs/experiment/exp23c_owen.yaml` |
| Real-network cost | `... exp25_real_networks --config configs/experiment/exp25_real_networks.yaml` |
| Per-source validation | `python scripts/real_per_source.py --net wikivote` |
| &nbsp;&nbsp;analytic columns only, reusing the saved reference | `python scripts/real_per_source.py --net wikivote --reuse-mc` |
| Budget of the reference | `python scripts/real_budget_strata.py --net wikivote --rep 1`, then `python scripts/aggregate_budget_curve.py` |
| Total-variance check | `python scripts/total_variance_check.py` |
| Gaussian-decoder table of the appendix | `python scripts/aggregate_exp20.py` |
| Same-footing per-source table and its paired bootstrap | `python scripts/aggregate_real_per_source.py` |
| Sensitivity to $q$ / dangling | `python scripts/q_sensitivity.py`, `python scripts/dangling_sensitivity.py` |
| Figures | `python scripts/make_figures.py` |

Aggregation scripts (`scripts/aggregate_*.py`) print the tables as they appear in the
paper from the CSVs in `outputs/`, so the tables can be checked without re-running the
experiments.

Runtimes on 48 cores: enumeration-scale experiments finish in under a minute; the
medium-scale rerun takes about 100 minutes; per-source validation on a real network takes
two to five hours, dominated by the Monte Carlo reference. The analytic attribution itself
takes 0.3 s on the smallest of the four real networks and 71 s on the largest.

## Tests

```bash
python -m pytest tests/ -q
```

The suite includes the identity of Lemma 1 (the analytic attribution must equal the
target evaluated on the linearized model, to machine precision), the law of total variance
per source, exactness under full observation, the agreement of the quadratic-form
recursion with exhaustive enumeration, and a guard that the row second moments are
computed by the exact dynamic program at every out-degree.

## Citing

Each release is archived on Zenodo. The concept DOI
[10.5281/zenodo.23938199](https://doi.org/10.5281/zenodo.23938199) always resolves to the
most recent version; every release also carries its own DOI, so a particular state of this
repository can be cited exactly. `CITATION.cff` holds the metadata, and GitHub's "Cite
this repository" button reads it.

## License

Code is MIT (`LICENSE`). Result CSVs under `outputs/` are CC BY 4.0.
