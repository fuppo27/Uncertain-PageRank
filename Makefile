SHELL := /bin/bash
PY := python
export PYTHONPATH := src

.DEFAULT_GOAL := help
.PHONY: help test figures snap exact medium real strata

help:  ## List the available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | sed 's/:.*## /	/'

test:  ## Run the regression tests, including the machine-precision gates
	$(PY) -m pytest tests/ -q

snap:  ## Fetch the public SNAP data from manifest.json and verify its SHA256
	$(PY) scripts/download_snap.py

exact:  ## Enumeration-scale validation (data behind Table IV)
	$(PY) -m uncertain_pagerank.experiments.exp22_source_attribution

medium:  ## Medium-scale validation (data behind Table VI)
	$(PY) -m uncertain_pagerank.experiments.exp23_medium_scale

real:  ## Real-network cost and per-source validation (data behind Tables VII-IX)
	$(PY) -m uncertain_pagerank.experiments.exp25_real_networks
	$(PY) scripts/real_per_source.py --net trackback
	$(PY) scripts/real_per_source.py --net wikivote

strata:  ## Budget curve, within stratum; each repetition uses an independent seed
	@for n in trackback wikivote; do for r in 1 2 3; do \
		$(PY) scripts/real_budget_strata.py --net $$n --rep $$r; done; done
	$(PY) scripts/aggregate_budget_curve.py

figures:  ## Build the figures in the paper
	$(PY) scripts/make_figures.py
