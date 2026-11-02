"""Checks on the data readers."""

import numpy as np

from uncertain_pagerank.datasets import DAY, load_dlbl, load_edgelist, load_temporal_edgelist

# Four nodes; node 3 has no out-link. Ids are 1-based and the trailing "1 1" is padding.
DLBL = """4 4 1
2 2:1 3:1 1 1
1 4:1 1 1
0 1 1
2 1:1 2:1 1 1
"""


def test_load_dlbl_parses_degrees_and_targets(tmp_path):
    p = tmp_path / "toy.dlbl"
    p.write_text(DLBL)
    lg = load_dlbl(p)
    g = lg.graph
    assert g.n == 4
    assert g.m == 5
    assert list(g.out_degrees()) == [2, 1, 0, 2]
    assert list(g.neighbors(0)[0]) == [1, 2]  # 1-based 2,3 -> 0-based 1,2
    assert list(g.neighbors(1)[0]) == [3]
    assert list(g.neighbors(3)[0]) == [0, 1]
    assert np.allclose(lg.link_weights, 1.0)
    assert list(lg.node_ids) == [1, 2, 3, 4]


def test_load_dlbl_drops_self_loops(tmp_path):
    p = tmp_path / "loop.dlbl"
    p.write_text("2 2 1\n2 1:1 2:1 1 1\n1 1:1 1 1\n")
    assert load_dlbl(p, drop_self_loops=True).graph.m == 2
    kept = load_dlbl(p, drop_self_loops=False)
    assert kept.graph.m == 3
    assert kept.n_self_loops == 0


def test_load_dlbl_rejects_inconsistent_degree(tmp_path):
    p = tmp_path / "bad.dlbl"
    p.write_text("2 2 1\n3 2:1 1 1\n0 1 1\n")
    try:
        load_dlbl(p)
    except ValueError as exc:
        assert "out-degree" in str(exc)
    else:
        raise AssertionError("an inconsistent out-degree went undetected")


def test_load_edgelist_keeps_node_count_when_dropping_self_loops(tmp_path):
    """A node whose only link is a self-loop must not disappear."""
    p = tmp_path / "e.txt"
    p.write_text("# comment\n10\t20\n20\t30\n40\t40\n")
    lg = load_edgelist(p, drop_self_loops=True)
    assert lg.graph.n == 4  # 10, 20, 30, 40
    assert lg.graph.m == 2
    assert lg.n_self_loops == 1
    assert list(lg.node_ids) == [10, 20, 30, 40]


def test_load_edgelist_removes_duplicates(tmp_path):
    p = tmp_path / "d.txt"
    p.write_text("1 2\n1 2\n2 1\n")
    lg = load_edgelist(p)
    assert lg.graph.m == 2
    assert lg.n_duplicates == 1


def test_load_temporal_estimates_activity_probability(tmp_path):
    """q_uv = (windows in which the link appears) / (all windows)."""
    p = tmp_path / "t.txt"
    t0 = 1_000_000_000
    rows = [
        (1, 2, t0),                  # window 0
        (1, 2, t0 + 100),            # window 0 again: counted once
        (1, 2, t0 + 1 * DAY),        # window 1
        (2, 3, t0 + 3 * DAY),        # window 3
    ]
    p.write_text("".join(f"{a} {b} {c}\n" for a, b, c in rows))
    lg, info = load_temporal_edgelist(p, window=DAY)
    assert info["n_windows"] == 4
    g = lg.graph
    assert g.m == 2
    q = dict(zip(map(tuple, g.edge_list().tolist()), g.probs.tolist()))
    assert np.isclose(q[(0, 1)], 2 / 4)  # (1,2) appears in windows 0 and 1
    assert np.isclose(q[(1, 2)], 1 / 4)  # (2,3) only in window 3
