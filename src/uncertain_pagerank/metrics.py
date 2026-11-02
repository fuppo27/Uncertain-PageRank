"""Metrics for score error and rank agreement."""

from __future__ import annotations

import numpy as np
from scipy import stats


def l1(a: np.ndarray, b: np.ndarray) -> float:
    """||a - b||_1。"""
    return float(np.abs(a - b).sum())


def linf(a: np.ndarray, b: np.ndarray) -> float:
    """||a - b||_inf: the eps in the top-k guarantee condition g_k > 2 eps."""
    return float(np.abs(a - b).max())


def topk(y: np.ndarray, k: int) -> np.ndarray:
    """The indices of the top k nodes, descending, ties broken by index."""
    return np.argsort(-y, kind="stable")[:k]


def topk_indicator(y: np.ndarray, k: int, rtol: float = 1e-12) -> np.ndarray:
    """The indicator of the top-k set, splitting weight evenly across ties.

    Used when summing P(v in top-k) over possible worlds. Breaking ties by index
    would make the answer depend on how PageRank was solved (by iteration or
    directly), so ties share their weight instead.

    Args:
        y: the scores, (n,) or (batch, n).
        k: the size of the top set.
        rtol: the relative tolerance within which scores count as tied.
    Returns:
        The same shape as y, each entry in [0, 1], summing to min(k, n).
    """
    single = y.ndim == 1
    yy = y[None, :] if single else y
    n = yy.shape[1]
    if k >= n:
        out = np.ones_like(yy)
        return out[0] if single else out
    threshold = np.partition(yy, n - k, axis=1)[:, n - k][:, None]
    tol = rtol * np.maximum(np.abs(threshold), 1.0)
    above = yy > threshold + tol
    tied = np.abs(yy - threshold) <= tol
    slots = np.maximum(k - above.sum(axis=1, keepdims=True), 0)
    n_tied = tied.sum(axis=1, keepdims=True)
    share = np.divide(slots, n_tied, out=np.zeros_like(threshold), where=n_tied > 0)
    out = above.astype(np.float64) + tied * share
    return out[0] if single else out


def topk_overlap(y_ref: np.ndarray, y: np.ndarray, k: int) -> float:
    """The top-k overlap |A_ref(k) and A(k)| / k, Eq. (13)."""
    return len(set(topk(y_ref, k).tolist()) & set(topk(y, k).tolist())) / k


def rank_gap(y: np.ndarray, k: int) -> float:
    """The margin between ranks k and k+1, g_k = y_(k) - y_(k+1).

    If ||y_ref - y||_inf <= eps and g_k > 2 eps, the two top-k sets agree.
    """
    s = np.sort(y)[::-1]
    if k >= s.size:
        return float("inf")
    return float(s[k - 1] - s[k])


def certified_topk(y: np.ndarray, eps: float, k: int) -> bool:
    """Whether the top-k sets are guaranteed to agree, i.e. g_k > 2 eps.

    `eps` is meant to be a computable upper bound. Passing the true
    ||y_ref - y||_inf instead turns this into an oracle check: whether the
    guarantee would hold if the answer were known.
    """
    return rank_gap(y, k) > 2.0 * eps


def kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    """Kendall's rank correlation (tau-b)."""
    return float(stats.kendalltau(a, b).statistic)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman's rank correlation."""
    return float(stats.spearmanr(a, b).statistic)


def ndcg_at_k(y_ref: np.ndarray, y: np.ndarray, k: int) -> float:
    """NDCG@k, collecting the scores of y_ref as gains in the order given by y."""
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(y_ref[topk(y, k)] @ disc)
    idcg = float(np.sort(y_ref)[::-1][:k] @ disc)
    return dcg / idcg if idcg > 0 else 1.0


def ranking_report(
    y_ref: np.ndarray,
    y: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10, 20, 50),
    kendall_max_n: int = 100_000,
) -> dict:
    """Compute the score error and the rank agreement together.

    Args:
        y_ref: the reference scores, for example the exact s-average.
        y: the scores compared against it, for example the p-average.
        ks: the values of k for the top-k metrics.
        kendall_max_n: skip Kendall tau above this node count. It is
            O(n log n) but still costly.
    Returns:
        A mapping from metric name to value. Note that certified_oracle@k uses
        the true Linf error, so it is an oracle check rather than a guarantee
        from a computable bound.
    """
    eps = linf(y_ref, y)
    report: dict[str, float] = {
        "l1": l1(y_ref, y),
        "linf": eps,
        "kendall_tau": kendall_tau(y_ref, y) if y.size <= kendall_max_n else float("nan"),
        "spearman": spearman(y_ref, y),
    }
    for k in ks:
        if k > y.size:
            continue
        report[f"overlap@{k}"] = topk_overlap(y_ref, y, k)
        report[f"ndcg@{k}"] = ndcg_at_k(y_ref, y, k)
        report[f"gap@{k}"] = rank_gap(y, k)
        # An oracle check, using the true Linf error as eps. A usable
        # guarantee would need a computable upper bound on eps.
        report[f"certified_oracle@{k}"] = float(rank_gap(y, k) > 2.0 * eps)
    return report
