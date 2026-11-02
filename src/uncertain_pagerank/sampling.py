"""Approximating the s-average by sampling possible worlds, Eq. (11).

    y_s(S) = (1/S) sum_{s=1}^{S} PRS(P_s; alpha)

Used as the comparison at sizes where enumeration is impossible, and to measure
how close the estimate gets to the true s-average as S grows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .expected_transition import ExpectedTransition
from .graph import UncertainGraph
from .pagerank import _resolve_z, pagerank


@dataclass
class MonteCarloResult:
    """The result of estimating the s-average by sampling."""

    scores: np.ndarray
    """The final sample mean."""

    n_samples: int
    checkpoints: dict[int, np.ndarray] = field(default_factory=dict)
    """S -> the sample mean up to that point."""

    mean_iterations: float = 0.0


class WorldSampler:
    """Sample possible worlds from an uncertain graph, one at a time.

    The sparsity pattern stays that of the original graph and only the data is
    rewritten; an absent link is kept as an explicit zero.
    """

    def __init__(self, graph: UncertainGraph, rng: np.random.Generator) -> None:
        self.graph = graph
        self.rng = rng
        self._csum = np.zeros(graph.m + 1, dtype=np.float64)
        self._b = sp.csr_matrix(
            (np.zeros(graph.m), graph.indices.copy(), graph.indptr.copy()),
            shape=(graph.n, graph.n),
        )

    def sample_mask(self) -> np.ndarray:
        """Draw the connection state x_e in {0,1} of each link."""
        return (self.rng.random(self.graph.m) < self.graph.probs).astype(np.float64)

    def transition(self, mask: np.ndarray) -> ExpectedTransition:
        """Build the transition matrix of the realized graph from the states."""
        g = self.graph
        np.cumsum(mask, out=self._csum[1:])
        deg = self._csum[g.indptr[1:]] - self._csum[g.indptr[:-1]]
        deg_per_edge = np.repeat(deg, g.out_degrees())
        data = np.divide(mask, deg_per_edge, out=np.zeros_like(mask), where=deg_per_edge > 0)
        self._b.data[:] = data
        return ExpectedTransition(B=self._b, delta=(deg == 0).astype(np.float64))


def monte_carlo_savg(
    graph: UncertainGraph,
    alpha: float = 0.85,
    z: np.ndarray | None = None,
    n_samples: int = 1000,
    seed: int = 0,
    checkpoints: tuple[int, ...] = (),
    tol: float = 1e-12,
) -> MonteCarloResult:
    """Approximate the s-average by sampling possible worlds.

    Args:
        graph: the uncertain graph.
        alpha: the damping factor.
        z: the personalization vector.
        n_samples: the sample size S.
        seed: the random seed.
        checkpoints: the sample counts at which to record the running mean,
            for example (1, 10, 100, 1000).
        tol: the PageRank convergence threshold within each sample.
    """
    z = _resolve_z(z, graph.n)
    sampler = WorldSampler(graph, np.random.default_rng(seed))
    marks = sorted({s for s in checkpoints if 0 < s <= n_samples})

    total = np.zeros(graph.n)
    iters = 0
    saved: dict[int, np.ndarray] = {}
    for s in range(1, n_samples + 1):
        res = pagerank(sampler.transition(sampler.sample_mask()), alpha=alpha, z=z, tol=tol)
        total += res.scores
        iters += res.iterations
        if marks and s == marks[0]:
            saved[s] = total / s
            marks.pop(0)

    return MonteCarloResult(
        scores=total / n_samples,
        n_samples=n_samples,
        checkpoints=saved,
        mean_iterations=iters / n_samples,
    )
