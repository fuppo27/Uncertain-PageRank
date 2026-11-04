# uncertain_pagerank

Implementation for the journal extension of IEEE BigData 2020, *"Efficient Computing of
PageRank Scores on Exact Expected Transition Matrix of Large Uncertain Graph"*.

No code from the conference version survived, so this implementation was rebuilt from the
equations in the paper.

## Terminology

| Name | Meaning | Implementation |
| --- | --- | --- |
| p-avg | `PR(E_X[P_X])`: one PageRank computation on the expected transition matrix | `expected_transition` + `pagerank` |
| s-avg | `E_X[PR(P_X)]`: the expected PageRank of a realized graph | `exact_savg` (exhaustive, exact) / `monte_carlo_savg` (sampling) |

The two are different models rather than approximations of one another.

## Modules

| Module | Contents | Conference paper |
| --- | --- | --- |
| `graph` | Directed graph with per-link connection probabilities (CSR) | Eq. (3) |
| `expected_transition` | Exact expected transition matrix | Eq. (6)-(10) |
| `pagerank` | Sparse, dense and direct solvers | Eq. (1), (2) |
| `possible_worlds` | Exact s-avg by enumerating all worlds | Eq. (4) |
| `sampling` | s-avg by sampling worlds | Eq. (11) |
| `metrics` | Score error, rank agreement, top-k guarantees | Eq. (12), (13) |
| `second_order`, `pair_moment` | Exact row second moments (new in the journal version) | --- |
| `variance`, `attribution` | Score covariance and source-level attribution (new) | --- |
| `world_batch` | Batched PageRank over sampled worlds | --- |
| `generators` | Synthetic graphs and schemes for assigning `q_e` | --- |

## Sparse form of the expected transition matrix

With `delta_u = P(D_u^X = 0) = prod_v (1 - q_uv)` the probability that `u` is dangling,

```
Pbar = B + delta z^T
```

where `B` is nonzero only on the original link set. Iteration therefore uses

```
Pbar^T y = B^T y + z (delta . y)
```

and never forms a dense `|V| x |V|` matrix: `O(|E| + |V|)` time per iteration and
`O(|E| + |V|)` memory.

## Numerical stability of Eq. (9)

Equation (9) of the conference paper recovers the connection-count distribution with link
`i` removed from the full distribution `gamma` by a **forward** deconvolution, whose error
amplification is `q_i / (1 - q_i)` per step. For `q_i > 1/2` that factor exceeds one, so
rounding error grows exponentially in the out-degree `D`.

Measured at `q_e = 0.9`, where the true value per link is `O(1/D)`:

| D | Eq. (9) as published (relative error) | This implementation |
| ---: | ---: | ---: |
| 20 | 1.6e-12 | 0 |
| 50 | 1.2e-04 | 4.4e-16 |
| 100 | 3.2e+07 | 0 |
| 200 | 3.3e+33 | 2.2e-16 |

Measured against the exact leave-one-out reference
(`expected_transition_loo_reference`). `test_published_eq9_error_matches_the_paper`
pins the two values the paper quotes.

This implementation selects the direction instead: forward when `q_i < 1/2`, backward
otherwise (amplification `(1 - q_i) / q_i <= 1`). The two recurrences are algebraically
equivalent and the cost stays `O(sum_u D_u^2)`.

The conference experiments used a uniform `q` throughout, for which the closed form of
Eq. (6) applies, so this path was never exercised there. With non-uniform `q_e`, which the
journal version is about, selecting the direction is necessary. The naive version is kept
as `expected_transition_naive_eq9` for comparison.

## Usage

```python
import numpy as np
from uncertain_pagerank import (
    UncertainGraph, expected_transition, pagerank, exact_savg, uniform_vector,
)

g = UncertainGraph.from_edges(3, [(0,1),(0,2),(1,0),(1,2),(2,0),(2,1)],
                              [0.7, 0.7, 0.9, 0.9, 0.5, 0.1])
z = uniform_vector(g.n)

p = pagerank(expected_transition(g), alpha=0.85, z=z).scores   # p-avg
s = exact_savg(g, alpha=0.85, z=z, topk=(1,))                  # s-avg, all 2^6 worlds

print(p)               # [0.366630 0.271180 0.362190] -> node 0 ranks first
print(s.scores)        # [0.363037 0.270970 0.365993] -> node 2 ranks first
print(s.topk_prob[1])  # [0.4955 0.1495 0.3551] probability of ranking first
```

On this three-node example p-avg and s-avg disagree about the top node. The conference
observation that the two agree closely on real data at the top of the ranking is an
empirical one, not a guarantee.

## Running an experiment

```bash
python -m uncertain_pagerank.experiments.exp22_source_attribution \
    --config configs/experiment/exp22_source_attribution.yaml
```

Results go to `outputs/<experiment name>/results.csv`, and the information needed to
reproduce them (git commit, seeds, environment) to `meta.json` in the same directory.
