"""Synthetic graphs, and schemes for assigning the link probabilities q_e.

Four assignment schemes are provided. The conference version evaluated only
the uniform one, with a single q shared by every link.
"""

from __future__ import annotations

import numpy as np

from .graph import UncertainGraph


def erdos_renyi_digraph(n: int, m: int, rng: np.random.Generator) -> UncertainGraph:
    """A random digraph: m directed edges drawn uniformly, no self-loops."""
    max_edges = n * (n - 1)
    if m > max_edges:
        raise ValueError(f"{m} edges exceeds the maximum {max_edges} for {n} nodes")
    codes = rng.choice(max_edges, size=m, replace=False)
    src, off = np.divmod(codes, n - 1)
    tgt = off + (off >= src)  # skip self-loops
    return UncertainGraph.from_edges(n, np.stack([src, tgt], axis=1), 1.0)


def scale_free_digraph(n: int, m_out: int, rng: np.random.Generator) -> UncertainGraph:
    """A preferential-attachment digraph, with a heavy-tailed in-degree.

    Each new node adds m_out out-links to existing nodes, chosen with
    probability proportional to (in-degree + 1).
    """
    if n <= m_out:
        raise ValueError("the node count must exceed m_out")
    edges: list[tuple[int, int]] = []
    indeg = np.ones(n, dtype=np.float64)
    for u in range(1, m_out + 1):  # a small complete seed: each node links to
        #                              every node before it
        edges.extend((u, j) for j in range(u))
        indeg[:u] += 1.0
    for u in range(m_out + 1, n):
        w = indeg[:u] / indeg[:u].sum()
        targets = rng.choice(u, size=m_out, replace=False, p=w)
        edges.extend((u, int(v)) for v in targets)
        indeg[targets] += 1.0
    return UncertainGraph.from_edges(n, np.asarray(edges), 1.0)


def sbm_digraph(
    n: int, n_blocks: int, p_in: float, p_out: float, rng: np.random.Generator
) -> tuple[UncertainGraph, np.ndarray]:
    """A directed stochastic block model: p_in within a block, p_out across."""
    block = np.repeat(np.arange(n_blocks), int(np.ceil(n / n_blocks)))[:n]
    same = block[:, None] == block[None, :]
    prob = np.where(same, p_in, p_out)
    np.fill_diagonal(prob, 0.0)
    adj = rng.random((n, n)) < prob
    src, tgt = np.nonzero(adj)
    return UncertainGraph.from_edges(n, np.stack([src, tgt], axis=1), 1.0), block


def uniform_probs(graph: UncertainGraph, q: float) -> UncertainGraph:
    """One connection probability shared by every link, as in the conference version."""
    return graph.with_probs(np.full(graph.m, float(q)))


def beta_probs(
    graph: UncertainGraph, mean: float, concentration: float, rng: np.random.Generator
) -> UncertainGraph:
    """q_e ~ Beta(a, b), with concentration varying the spread at a fixed mean.

    a = mean * concentration, b = (1 - mean) * concentration.
    A smaller concentration makes the uncertainty more heterogeneous.
    """
    a, b = mean * concentration, (1.0 - mean) * concentration
    return graph.with_probs(rng.beta(a, b, size=graph.m))


def source_dependent_probs(
    graph: UncertainGraph, mean: float, spread: float, rng: np.random.Generator
) -> UncertainGraph:
    """A per-node reliability r_u, with q_uv = r_u.

    Models the case where every link out of a node shares its reliability, as
    when a communication node itself fails.
    """
    r = np.clip(rng.normal(mean, spread, size=graph.n), 1e-6, 1.0 - 1e-6)
    return graph.with_probs(np.repeat(r, graph.out_degrees()))


def topology_dependent_probs(
    graph: UncertainGraph, beta0: float, beta1: float, beta2: float
) -> UncertainGraph:
    """A degree-dependent q_uv = sigmoid(b0 + b1 log d_u + b2 log d_v).

    Lets the neighbourhood of a hub be stable (b1, b2 > 0) or unstable (< 0).
    Here d_u is the out-degree and d_v the in-degree.
    """
    out_deg = graph.out_degrees().astype(np.float64)
    in_deg = np.bincount(graph.indices, minlength=graph.n).astype(np.float64)
    src = graph.edge_list()[:, 0]
    logit = beta0 + beta1 * np.log1p(out_deg[src]) + beta2 * np.log1p(in_deg[graph.indices])
    return graph.with_probs(1.0 / (1.0 + np.exp(-logit)))
