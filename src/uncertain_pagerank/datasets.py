"""Loading real networks, and estimating link probabilities from temporal data.

Reads the SNAP format: comment lines starting with `#`, then two or three
whitespace-separated integers per line. Node ids are relabelled to 0..n-1 and
the original ids are returned alongside.

From temporal data the link probability is estimated as

    q_uv = (time windows in which (u,v) appears) / (all time windows)

which is the probability of observing that interaction in an arbitrary time
window, **not** a probability that the link fails. It must be described as a
temporal activity probability wherever it is reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .graph import UncertainGraph

DAY = 86_400
WEEK = 7 * DAY


@dataclass
class LoadedGraph:
    """A loaded graph, with the original node ids and related information."""

    graph: UncertainGraph
    node_ids: np.ndarray
    """New id -> original node id."""

    n_raw_rows: int
    n_self_loops: int
    n_duplicates: int
    source: str
    link_weights: np.ndarray | None = None
    """Link weights from the source data, if any. Unused by the PageRank here."""

    def summary(self) -> dict:
        """A summary of size and degree. sum_u D_u^2 indicates the cost of building
        the expected transition matrix."""
        g = self.graph
        d = g.out_degrees()
        indeg = np.bincount(g.indices, minlength=g.n)
        return {
            "name": self.source,
            "n": g.n,
            "m": g.m,
            "sum_deg_sq": g.sum_out_degree_sq(),
            "max_out_degree": int(d.max()) if d.size else 0,
            "mean_out_degree": float(d.mean()),
            "out_degree_var": float(d.var()),
            "max_in_degree": int(indeg.max()) if indeg.size else 0,
            "n_dangling": int((d == 0).sum()),
            "self_loops_removed": self.n_self_loops,
            "duplicates_removed": self.n_duplicates,
        }


def _read_columns(path: Path, n_cols: int) -> np.ndarray:
    """Read whitespace-separated integers, skipping lines that start with `#`."""
    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        usecols=list(range(n_cols)),
        dtype=np.int64,
        engine="c",
    )
    return df.to_numpy()


def _relabel(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Relabel the node ids to 0..n-1."""
    ids = np.unique(np.concatenate([src, dst]))
    return np.searchsorted(ids, src), np.searchsorted(ids, dst), ids


def load_edgelist(path: str | Path, drop_self_loops: bool = True) -> LoadedGraph:
    """Read a SNAP directed edge list, with every probability set to 1.0.

    Args:
        path: a `.txt` or `.txt.gz` file.
        drop_self_loops: whether to drop self-loops. On by default, since a
            self-loop inflates a PageRank score unnaturally.
    """
    path = Path(path)
    raw = _read_columns(path, 2)
    n_rows = int(raw.shape[0])

    # Relabel first, so that a node with only a self-loop is not removed: the
    # node count then matches the source data whether or not self-loops are
    # dropped.
    src, dst, ids = _relabel(raw[:, 0], raw[:, 1])
    n = int(ids.shape[0])

    n_self = 0
    if drop_self_loops:
        keep = src != dst
        n_self = int((~keep).sum())
        src, dst = src[keep], dst[keep]

    code = np.unique(src.astype(np.int64) * n + dst.astype(np.int64))
    n_dup = int(src.shape[0] - code.shape[0])
    src, dst = np.divmod(code, n)

    graph = UncertainGraph.from_edges(n, np.stack([src, dst], axis=1), 1.0)
    return LoadedGraph(graph, ids, n_rows, n_self, n_dup, path.name.split(".")[0])


def load_temporal_edgelist(
    path: str | Path,
    window: int = WEEK,
    drop_self_loops: bool = True,
    min_probability: float = 0.0,
) -> tuple[LoadedGraph, dict]:
    """Estimate link probabilities from a temporal edge list (SRC DST UNIXTS).

    Args:
        path: a `.txt` or `.txt.gz` file whose third column is a Unix time.
        window: the width of a time window in seconds. One week by default.
        drop_self_loops: whether to drop self-loops.
        min_probability: drop links whose q_e falls below this. Zero keeps every
            link.
    Returns:
        (LoadedGraph, information about the windows). The estimated q_e are in
        graph.probs.
    """
    path = Path(path)
    raw = _read_columns(path, 3)
    n_rows = int(raw.shape[0])

    ts = raw[:, 2]
    src, dst, ids = _relabel(raw[:, 0], raw[:, 1])  # relabel before dropping loops
    n = int(ids.shape[0])

    n_self = 0
    if drop_self_loops:
        keep = src != dst
        n_self = int((~keep).sum())
        src, dst, ts = src[keep], dst[keep], ts[keep]

    win = (ts - ts.min()) // window
    n_windows = int(win.max()) + 1

    code = src.astype(np.int64) * n + dst.astype(np.int64)
    # Deduplicate (link, window) pairs, then count the windows per link
    pair = np.unique(code * n_windows + win)
    edge_code, counts = np.unique(pair // n_windows, return_counts=True)
    probs = counts / n_windows

    if min_probability > 0.0:
        keep = probs >= min_probability
        edge_code, probs = edge_code[keep], probs[keep]

    e_src, e_dst = np.divmod(edge_code, n)
    # A node that ends up unused keeps its id and becomes isolated
    graph = UncertainGraph.from_edges(n, np.stack([e_src, e_dst], axis=1), probs)
    loaded = LoadedGraph(graph, ids, n_rows, n_self, 0, path.name.split(".")[0])
    info = {
        "window_seconds": window,
        "n_windows": n_windows,
        "span_days": float((ts.max() - ts.min()) / DAY),
        "q_mean": float(probs.mean()),
        "q_median": float(np.median(probs)),
        "q_max": float(probs.max()),
        "frac_q_below_0.05": float((probs < 0.05).mean()),
    }
    return loaded, info



def load_dlbl(path: str | Path, drop_self_loops: bool = True) -> LoadedGraph:
    """Read a directed graph in the laboratory `.dlbl` format.

    The format of Follow NW (Ameba Blog) and Trackback NW (goo Blog), the two
    networks used in the conference version.

    Layout::

        <node count> <node count> <?>         line 1, the header
        <out-degree> <id:weight> ... 1 1      one line per node thereafter

    - Line order corresponds to nodes: line k, counting from 1 after the
      header, is node k.
    - Target ids are **1-based** and are converted to 0..n-1 internally.
    - The trailing `1 1` on each line is padding and is skipped.
    - A node of out-degree zero has the line `0 1 1`.

    Args:
        path: the `.dlbl` file.
        drop_self_loops: whether to drop self-loops.
    """
    path = Path(path)
    lines = path.read_text().splitlines()
    n = int(lines[0].split()[0])
    if len(lines) < n + 1:
        raise ValueError(f"the header declares {n} nodes but the body has only "
                         f"{len(lines) - 1} lines")

    src: list[int] = []
    dst: list[int] = []
    weight: list[float] = []
    for i, line in enumerate(lines[1 : n + 1]):
        tok = line.split()
        deg = int(tok[0])
        if len(tok) != deg + 3:  # out-degree + links + two padding tokens
            raise ValueError(f"{path.name} line {i + 2}: out-degree {deg} does not match "
                             f"the token count")
        for t in tok[1 : deg + 1]:
            node_id, _, w = t.partition(":")
            src.append(i)  # line position = 0-based node index
            dst.append(int(node_id) - 1)  # target ids are 1-based
            weight.append(float(w) if w else 1.0)

    src_a = np.asarray(src, dtype=np.int64)
    dst_a = np.asarray(dst, dtype=np.int64)
    w_a = np.asarray(weight, dtype=np.float64)
    n_rows = int(src_a.shape[0])

    if dst_a.size and (dst_a.min() < 0 or dst_a.max() >= n):
        raise ValueError(f"{path.name}: a target id falls outside the node range 1..{n}")

    n_self = 0
    if drop_self_loops:
        keep = src_a != dst_a
        n_self = int((~keep).sum())
        src_a, dst_a, w_a = src_a[keep], dst_a[keep], w_a[keep]

    code = src_a * n + dst_a
    uniq, first = np.unique(code, return_index=True)
    n_dup = int(src_a.shape[0] - uniq.shape[0])
    src_a, dst_a, w_a = src_a[first], dst_a[first], w_a[first]

    graph = UncertainGraph.from_edges(n, np.stack([src_a, dst_a], axis=1), 1.0)
    # from_edges reorders the edges, so reorder the weights the same way
    order = np.lexsort((dst_a, src_a))
    return LoadedGraph(
        graph, np.arange(1, n + 1), n_rows, n_self, n_dup, path.stem, w_a[order]
    )


def main() -> None:
    """List the size and degree of each downloaded dataset."""
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="where to write the CSV")
    args = parser.parse_args()

    rows = []
    for p in args.paths:
        print(f"loading: {p.name} ...", flush=True)
        loader = load_dlbl if p.suffix == ".dlbl" else load_edgelist
        rows.append(loader(p).summary())
    df = pd.DataFrame(rows).sort_values("m")
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()

