"""M1 ablation: does the inductive bias of the kernel fit drive the alpha_L<->LG correlation?

Computes alpha_L three ways on the 1L1H learnable cells and correlates each with empirical LG:
  (i)   min-norm least-squares interpolant to the one-hot target (Theorem 1's W*)
  (ii)  cross-entropy fit with weight decay 1e-4 (the reported alpha_L)
  (iii) cross-entropy fit with wd=0, long horizon (max-margin direction, Soudry et al.)
"""
import os, sys, json
import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import spectra as S

L0, TEST_LENS = 20, [30, 40, 50, 60, 80]
rows = json.load(open(os.path.join(os.path.dirname(__file__), "..", "results", "study_rows.json")))

# 1L1H learnable cells (train_acc>0.8), averaged LG over seeds, per (pe,task,L)
def lg_at(pe, task, L):
    v = [r["acc_lg"] for r in rows if r["tier"] == "1L1H" and r["pe"] == pe and r["task"] == task and r["L"] == L]
    return float(np.mean(v)) if v else None

def is_learnable(pe, task):
    v = [r["acc_train"] for r in rows if r["tier"] == "1L1H" and r["pe"] == pe and r["task"] == task]
    return v and np.mean(v) > 0.8


def alpha_minnorm(pe, task, L):
    """min-norm least-squares fit of kernel coefficients to the one-hot target."""
    tk = S.TASKS[task]; B = S.PES[pe](L, L0=L0)
    cons = S._causal_constraints(tk, L0)
    A = np.array([[Bk[i, j] for Bk in B] for (i, j, _) in cons])
    b = np.array([v for (_, _, v) in cons])
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)            # min-norm LS solution
    P = sum(c * Bk for c, Bk in zip(coef, B))
    return _argmax_acc(P, tk, L)


def alpha_ce(pe, task, L, wd, steps):
    tk = S.TASKS[task]; B = S.PES[pe](L, L0=L0)
    Bstack = torch.tensor(np.stack(B), dtype=torch.float32)
    rows_L0 = [(i, tk["target"](L0, i), i + 1) for i in tk["qregion"](L0)
               if 0 <= tk["target"](L0, i) <= i]
    coef = S._fit_coefficients(Bstack, rows_L0, steps=steps, wd=wd,
                               neg_idx=S.NEG_IDX.get(pe, ()))
    P = torch.einsum("k,kij->ij", coef, Bstack).numpy()
    return _argmax_acc(P, tk, L)


def _argmax_acc(P, tk, L, eps=1e-4):
    correct = rows_ = 0
    for i in tk["qregion"](L):
        if i < L0:
            continue
        t = tk["target"](L, i)
        if t < 0 or t > i:
            continue
        logits = P[i, : i + 1]; srt = np.sort(logits)[::-1]
        ru = srt[1] if len(srt) > 1 else srt[0] - 1.0
        rows_ += 1
        correct += int(logits[t] >= srt[0] - 1e-9 and (logits[t] - ru) > eps)
    return correct / rows_ if rows_ else float("nan")


cells = [(pe, t) for pe in S.PES for t in S.TASKS if is_learnable(pe, t)]
print(f"learnable 1L1H cells: {len(cells)} -> {sorted(cells)}\n")

variants = {"min-norm (LS)": lambda pe, t, L: alpha_minnorm(pe, t, L),
            "CE wd=1e-4 (reported)": lambda pe, t, L: alpha_ce(pe, t, L, 1e-4, 400),
            "CE wd=0 (max-margin)": lambda pe, t, L: alpha_ce(pe, t, L, 0.0, 2000)}

lg = []
A = {k: [] for k in variants}
for pe, t in cells:
    for L in TEST_LENS:
        g = lg_at(pe, t, L)
        if g is None:
            continue
        lg.append(g)
        for k, fn in variants.items():
            A[k].append(fn(pe, t, L))
lg = np.array(lg)
print(f"correlation of alpha_L (each fit) with empirical LG, over {len(lg)} learnable (pe,task,L) rows:")
for k in variants:
    a = np.array(A[k])
    rho = spearmanr(a, lg)[0]; pear = np.corrcoef(a, lg)[0, 1]
    print(f"  {k:24s}: Spearman={rho:.3f}  Pearson={pear:.3f}")
out = {k: {"alpha": A[k]} for k in variants}
out["lg"] = lg.tolist(); out["cells"] = cells
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", "ablation_fit.json"), "w"))
print("\nsaved results/ablation_fit.json")
