"""Build the figures for the paper, at IEEE single-column width (3.4in).

Legible in greyscale and under colour-vision deficiency. The palette is five
Okabe-Ito colours, checked: every pair is at least 16 dE apart in normal vision,
and at least 8 under dichromacy except green/pink, which is 7.9. Every series is
therefore **also** encoded by marker and line style, so nothing depends on
colour alone.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "papers/journal/figures"
FIG.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, GREEN, PINK, SKY = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9"
INK, MUTED = "#1a1a1a", "#6b6b6b"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.labelcolor": INK, "grid.color": "#dddddd", "grid.linewidth": 0.5,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def fig_exact_vs_analytic() -> None:
    """Exact Psi_u against analytic Phi_hat_u: the value error and the preserved
    ordering in one view."""
    d = pd.read_csv(REPO / "outputs/fig_exact_vs_analytic.csv")
    d = d[(d.psi_over_V > 0) & (d.phi_over_V > 0)]
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    styles = {"boundary": (BLUE, "o"), "node": (ORANGE, "s"),
              "identity": (GREEN, "^"), "gap": (PINK, "D")}
    lo = min(d.psi_over_V.min(), d.phi_over_V.min()) * 0.5
    hi = max(d.psi_over_V.max(), d.phi_over_V.max()) * 2
    ax.plot([lo, hi], [lo, hi], "-", color=MUTED, lw=0.8, zorder=1)
    ax.annotate("$y=x$", (hi, lo), textcoords="offset points", xytext=(-26, 14),
                fontsize=6, color=MUTED, ha="right")
    for name, (c, mk) in styles.items():
        s = d[d.contrast == name]
        ax.plot(s.psi_over_V, s.phi_over_V, mk, color=c, ms=2.6, alpha=0.75,
                mew=0, ls="none", label=name, zorder=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(r"exact $\Psi_u(C)\,/\,\mathrm{tr}\,\mathrm{Cov}(Z)$")
    ax.set_ylabel(r"analytic $\widehat\Phi_u(C)\,/\,\mathrm{tr}\,\mathrm{Cov}(Z)$")
    ax.grid(True, ls=":", lw=0.5, zorder=0); ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.legend(loc="upper left", frameon=False, markerscale=2.2,
              handletextpad=0.3, borderpad=0.2, labelspacing=0.25)
    fig.savefig(FIG / "exact_vs_analytic.pdf"); fig.savefig(FIG / "exact_vs_analytic.png", dpi=220)
    plt.close(fig)
    print(f"exact_vs_analytic.pdf ({len(d)} points)")


def fig_real_per_source() -> None:
    """Per-source validation on real data: the reference Psi_u with error bars,
    ordered by phi_hat.

    Values can be negative or minute, so a plain log axis will not do: the
    vertical axis is symlog, and sources the budget could not resolve are kept
    rather than dropped. The strata are distinguished by colour and marker.
    """
    labels = {"top Phi": ("top $\\widehat\\Phi$", BLUE, "o"),
              "high PR, low Phi": ("high PR, low $\\widehat\\Phi$", ORANGE, "s"),
              "low out-degree": ("low out-degree", GREEN, "^"),
              "random": ("random", PINK, "D")}
    nets = [("trackback", "Trackback NW", 0.944, 0.960),
            ("wikivote", "wiki-Vote", 0.973, 0.978)]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=False)
    for ax, (key, title, sh, rho) in zip(axes, nets):
        d = pd.read_csv(REPO / f"outputs/real_per_source_{key}.csv")
        d = d.sort_values("phi_hat", ascending=False).reset_index(drop=True)
        d["rank"] = np.arange(1, len(d) + 1)
        for jp, (lab, c, mk) in labels.items():
            s = d[d.stratum == jp]
            if s.empty:
                continue
            ax.errorbar(s["rank"], s.psi_ref, yerr=1.96 * s.psi_se, fmt=mk,
                        color=c, ms=3.6, mew=0, lw=0, elinewidth=0.7,
                        ecolor=c, capsize=1.4, alpha=0.95, label=lab, zorder=3)
        ax.set_yscale("symlog", linthresh=1e-19)
        ax.set_xlabel(r"source rank by $\widehat\Phi_u$ (1 = highest)")
        ax.set_title(f"{title}\n" r"$\rho$=" f"{rho:.3f}  (reference split-half {sh:.3f})",
                     fontsize=7, pad=4)
        ax.grid(True, ls=":", lw=0.5, zorder=0); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=6)
    axes[0].set_ylabel(r"reference $\Psi_u(C)$  (95% CI)")
    axes[0].legend(loc="lower left", frameon=False, fontsize=5.8,
                   handletextpad=0.3, borderpad=0.15, labelspacing=0.25)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIG / "real_per_source.pdf")
    fig.savefig(FIG / "real_per_source.png", dpi=220)
    plt.close(fig)
    print("real_per_source.pdf")


if __name__ == "__main__":
    # Two figures remain in the paper: the scatter and the per-source panel.
    # accuracy_cost and sensitivity_dangling duplicated tables and were cut.
    fig_exact_vs_analytic()
    fig_real_per_source()
