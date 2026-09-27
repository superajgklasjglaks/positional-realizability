"""Generate paper figures from results/study_rows.json (run after run_study.py)."""

import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = os.path.join(os.path.dirname(__file__), "..")
ROWS = json.load(open(os.path.join(ROOT, "results", "study_rows.json")))
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)


def _tiers():
    return [t for t in ["1L1H", "2L4H", "4L4H"] if any(r["tier"] == t for r in ROWS)]


def _cell_means(rows):
    """aggregate rows to one (alpha, lg) point per (pe,task) cell (avoids pseudoreplication)."""
    d = {}
    for r in rows:
        d.setdefault((r["pe"], r["task"]), []).append(r)
    a = np.array([np.mean([x["alpha"] for x in v]) for v in d.values()])
    g = np.array([np.mean([x["acc_lg"] for x in v]) for v in d.values()])
    return a, g


def fig_scatter():
    """Cell-level alpha_L vs empirical LG (one point per (PE,task) cell), per tier."""
    tiers = _tiers()
    fig, axes = plt.subplots(1, len(tiers), figsize=(5 * len(tiers), 4.2))
    if len(tiers) == 1:
        axes = [axes]
    for ax, tier in zip(axes, tiers):
        rs = [r for r in ROWS if r["tier"] == tier]
        a_all, lg_all = _cell_means(rs)
        a_l, lg_l = _cell_means([r for r in rs if r["acc_train"] > 0.8])
        ax.scatter(a_all, lg_all, c="lightgray", s=34, label="all cells")
        ax.scatter(a_l, lg_l, c="C0", s=40, label="learnable (train>0.8)")
        if len(a_l) > 2 and np.std(a_l) > 0:
            rho = spearmanr(a_l, lg_l)[0]
            ax.set_title(f"{tier}: learnable cell-level $\\rho$={rho:.2f} ($n$={len(a_l)} cells)")
        ax.set_xlabel(r"training-free $\alpha_L$ (cell mean)"); ax.set_ylabel("empirical LG (cell mean)")
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "scatter.pdf")); plt.close()
    print("wrote figures/scatter.pdf")


def fig_length_sweep():
    """LG accuracy and alpha_L vs test length L, for representative (pe,task) in 1L1H."""
    picks = [("rope", "prev"), ("rope", "skip5"), ("rope", "copy"),
             ("sinusoidal", "first"), ("sinusoidal", "prev"), ("learned-abs", "first")]
    lens = sorted({r["L"] for r in ROWS})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for pe, task in picks:
        lg = [np.mean([r["acc_lg"] for r in ROWS
                       if r["tier"] == "1L1H" and r["pe"] == pe and r["task"] == task and r["L"] == L])
              for L in lens]
        al = [np.mean([r["alpha"] for r in ROWS
                       if r["tier"] == "1L1H" and r["pe"] == pe and r["task"] == task and r["L"] == L])
              for L in lens]
        line, = ax.plot(lens, lg, marker="o", label=f"{pe}/{task}")
        ax.plot(lens, al, marker="x", ls="--", color=line.get_color(), alpha=0.6)
    ax.set_xlabel("test length $L$ (train $L_0=20$)"); ax.set_ylabel("accuracy")
    ax.set_title(r"solid = empirical LG, dashed = training-free $\alpha_L$")
    ax.set_ylim(-0.08, 1.18)        # headroom so the in-plot legend clears the top cluster
    # legend inside, in the clear band between the top (generalizing) and bottom (failing) curves
    ax.legend(fontsize=7, ncol=3, loc="center", bbox_to_anchor=(0.5, 0.60),
              framealpha=0.92, columnspacing=1.0, handlelength=1.6)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "length_sweep.pdf")); plt.close()
    print("wrote figures/length_sweep.pdf")


def fig_heatmaps():
    """alpha_L and LG heatmaps over (PE, task) at L=60, 1L1H."""
    pes = sorted({r["pe"] for r in ROWS}); tasks = sorted({r["task"] for r in ROWS})
    A = np.full((len(pes), len(tasks)), np.nan); G = np.full_like(A, np.nan)
    for i, pe in enumerate(pes):
        for j, task in enumerate(tasks):
            cell = [r for r in ROWS if r["tier"] == "1L1H" and r["pe"] == pe and r["task"] == task and r["L"] == 60]
            if cell:
                A[i, j] = np.mean([r["alpha"] for r in cell])
                G[i, j] = np.mean([r["acc_lg"] for r in cell])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, M, ttl in zip(axes, [A, G], [r"training-free $\alpha_{60}$", "empirical LG@60"]):
        im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(tasks))); ax.set_xticklabels(tasks, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(pes))); ax.set_yticklabels(pes, fontsize=8)
        ax.set_title(ttl)
        for i in range(len(pes)):
            for j in range(len(tasks)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                            color="white" if M[i, j] < 0.6 else "black", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "heatmaps.pdf")); plt.close()
    print("wrote figures/heatmaps.pdf")


def fig_depth_trend():
    """H3: learnable-cell Spearman rho and sufficiency precision vs depth."""
    tiers = _tiers()
    depth = {"1L1H": 1, "2L4H": 2, "4L4H": 4}
    xs, rhos, precs = [], [], []
    for tier in tiers:
        learn = [r for r in ROWS if r["tier"] == tier and r["acc_train"] > 0.8]
        if len(learn) < 3:
            continue
        a, lg = _cell_means(learn)              # cell-level, matching Table 3 (avoids pseudoreplication)
        rho = spearmanr(a, lg)[0]
        hi = [r for r in learn if r["alpha"] >= 0.5]
        prec = np.mean([r["acc_lg"] >= 0.7 for r in hi]) if hi else np.nan
        xs.append(depth[tier]); rhos.append(rho); precs.append(prec)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(xs, rhos, "o-", label=r"learnable Spearman $\rho(\alpha_L,\mathrm{LG})$")
    ax.plot(xs, precs, "s--", label=r"sufficiency precision")
    ax.set_xticks(xs); ax.set_xlabel("model depth (layers)"); ax.set_ylabel("value")
    ax.set_ylim(0, 1.05); ax.set_title(r"H3: $\alpha_L$ stays sufficient as depth grows")
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "depth_trend.pdf")); plt.close()
    print("wrote figures/depth_trend.pdf")


if __name__ == "__main__":
    fig_scatter()
    fig_length_sweep()
    fig_heatmaps()
    fig_depth_trend()
    print("done")
