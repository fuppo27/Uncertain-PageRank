"""A minimal representation of an uncertain directed graph.

Each directed link e=(u,v) carries a connection probability q_e in [0,1], and
the trials are independent Bernoulli ones, Eq. (3) of the conference paper.

The internal representation is CSR: out-links grouped by source.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class UncertainGraph:
    """A directed graph with per-link connection probabilities.

    Attributes:
        n: the node count |V|. Nodes are the integers 0..n-1.
        indptr: length n+1. The out-links of u are
            indices[indptr[u]:indptr[u+1]].
        indices: length m, the targets of the out-links.
        probs: length m, the connection probability q_e of each link.
    """

    n: int
    indptr: np.ndarray
    indices: np.ndarray
    probs: np.ndarray

    def __post_init__(self) -> None:
        if self.indptr.shape != (self.n + 1,):
            raise ValueError(f"indptr must have length n+1={self.n + 1}")
        if self.indices.shape != self.probs.shape:
            raise ValueError("indices and probs have different lengths")
        if self.indptr[-1] != self.indices.shape[0]:
            raise ValueError("the last entry of indptr does not match the link count")
        if self.indices.size and (self.indices.min() < 0 or self.indices.max() >= self.n):
            raise ValueError("indices contains a value outside the node range")
        if self.probs.size and (self.probs.min() < 0.0 or self.probs.max() > 1.0):
            raise ValueError("probs must lie in [0,1]")

    @property
    def m(self) -> int:
        """The link count |E|."""
        return int(self.indices.shape[0])

    def out_degrees(self) -> np.ndarray:
        """The out-degree D_u in the original graph, ignoring the probabilities."""
        return np.diff(self.indptr)

    def sum_out_degree_sq(self) -> int:
        """sum_u D_u^2: the cost of building the expected transition matrix."""
        d = self.out_degrees().astype(np.int64)
        return int((d * d).sum())

    def neighbors(self, u: int) -> tuple[np.ndarray, np.ndarray]:
        """The (out-link targets, connection probabilities) of node u."""
        s, e = self.indptr[u], self.indptr[u + 1]
        return self.indices[s:e], self.probs[s:e]

    def edge_list(self) -> np.ndarray:
        """An (m, 2) array of [source, target]."""
        src = np.repeat(np.arange(self.n), self.out_degrees())
        return np.stack([src, self.indices], axis=1)

    def with_probs(self, probs: np.ndarray) -> "UncertainGraph":
        """The same topology with the connection probabilities replaced."""
        probs = np.asarray(probs, dtype=np.float64)
        if probs.shape != self.probs.shape:
            raise ValueError("len(probs) does not match the link count")
        return UncertainGraph(self.n, self.indptr, self.indices, probs)

    def to_dense_adjacency(self) -> np.ndarray:
        """The adjacency matrix A: one on every link, probabilities ignored. For
        small-scale checks."""
        a = np.zeros((self.n, self.n), dtype=np.float64)
        e = self.edge_list()
        if e.size:
            a[e[:, 0], e[:, 1]] = 1.0
        return a

    @classmethod
    def from_edges(
        cls,
        n: int,
        edges: np.ndarray | list[tuple[int, int]],
        probs: np.ndarray | float,
    ) -> "UncertainGraph":
        """Build from an edge list.

        Parallel edges are rejected: a repeated (u,v) raises ValueError.
        Self-loops are allowed.

        Args:
            n: the node count.
            edges: (m, 2) of [source, target].
            probs: length m, or a scalar shared by every link.
        """
        edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        m = edges.shape[0]
        probs = np.full(m, float(probs)) if np.isscalar(probs) else np.asarray(probs, np.float64)
        if probs.shape != (m,):
            raise ValueError("len(probs) does not match the link count")

        order = np.lexsort((edges[:, 1], edges[:, 0]))
        edges, probs = edges[order], probs[order]
        if m > 1 and np.any(np.all(edges[1:] == edges[:-1], axis=1)):
            raise ValueError("the edge list contains a parallel edge")

        counts = np.bincount(edges[:, 0], minlength=n)
        indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(counts, out=indptr[1:])
        return cls(n, indptr, np.ascontiguousarray(edges[:, 1]), np.ascontiguousarray(probs))
