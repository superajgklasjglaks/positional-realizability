"""
Simulation Spectra: training-free length-generalization predictor.

For each (task, positional-encoding) pair we:
  1. realize the PE's positional-kernel family as a linear basis {B_k} over an n x n grid,
  2. fit the kernel coefficients so a linear-attention head attends correctly at train length L0
     (cross-entropy on the causal softmax -- exactly how an attention head is trained),
  3. re-evaluate the *same* coefficients at test length L  (this is the operator T_L),
  4. score strict-argmax accuracy on the extrapolation rows  ->  alpha_L  in [0, 1].

alpha_L is computed with NO model, NO data, NO gradient steps on real data -- only the PE
feature map and the task's target function. See theory/theory.md for the formal statements.
"""

import numpy as np
import torch


# --------------------------------------------------------------------------------------
# Tasks: each defines a query region and a target key position t_n(i), at length n.
# "growing" tasks need an offset ~ n (never seen at L0); "bounded" need a fixed small offset;
# "absolute" need a fixed absolute position. All are position-essential (repeated-token regime).
# --------------------------------------------------------------------------------------

def _half(n):
    return (n + 1) // 2


TASKS = {
    # relative, growing offset: copy aligned input token from the first half
    "copy":      dict(kind="rel-grow", qregion=lambda n: range(_half(n), n),
                      target=lambda n, i: i - _half(n)),
    # relative, growing & varying offset: reverse the first half
    "reverse":   dict(kind="rel-grow", qregion=lambda n: range(_half(n), n),
                      target=lambda n, i: (_half(n) - 1) - (i - _half(n))),
    # relative, bounded offset 1: attend to previous token
    "prev":      dict(kind="rel-bound", qregion=lambda n: range(1, n),
                      target=lambda n, i: i - 1),
    # relative, bounded offset 2 / 3 / 5
    "skip2":     dict(kind="rel-bound", qregion=lambda n: range(2, n),
                      target=lambda n, i: i - 2),
    "skip3":     dict(kind="rel-bound", qregion=lambda n: range(3, n),
                      target=lambda n, i: i - 3),
    "skip5":     dict(kind="rel-bound", qregion=lambda n: range(5, n),
                      target=lambda n, i: i - 5),
    # absolute, fixed position: attend to the very first token
    "first":     dict(kind="abs-fixed", qregion=lambda n: range(1, n),
                      target=lambda n, i: 0),
    # absolute, fixed INTERIOR positions (position 0 is the special causal boundary;
    # these check the absolute regime away from that boundary)
    "pos2":      dict(kind="abs-fixed", qregion=lambda n: range(3, n),
                      target=lambda n, i: 2),
    "pos5":      dict(kind="abs-fixed", qregion=lambda n: range(6, n),
                      target=lambda n, i: 5),
}


# --------------------------------------------------------------------------------------
# Positional encodings: each returns a list of n x n basis matrices spanning its
# realizable positional-kernel space K_n  (kernel = sum_k c_k B_k).
# --------------------------------------------------------------------------------------

def _freqs(num, base=10000.0):
    # standard transformer geometric frequencies (base = RoPE/sinusoidal theta base)
    return np.array([base ** (-2.0 * k / (2 * num)) for k in range(num)])


def basis_nope(n, **kw):
    # constant kernel only -> rank 1
    return [np.ones((n, n))]


def basis_learned_abs(n, L0=None, **kw):
    # Learned absolute PE: feature phi(i) = one-hot of position i for i < L0.
    # Faithful to training: positions i >= L0 are UNTRAINED; the model clamps them to the last
    # trained position L0-1 (as in src/model.py). So phi(i) = e_{min(i, L0-1)}.
    # Kernel = W[clamp(i), clamp(j)]: in-range absolute keys (e.g. position 0) stay addressable,
    # but distinct new query positions collapse to one feature (cannot resolve new relative offsets).
    lim = n if L0 is None else min(L0, n)
    clamp = np.minimum(np.arange(n), lim - 1)
    B = []
    for a in range(lim):
        ra = (clamp == a).astype(float)          # rows whose clamped feature is e_a
        for b in range(lim):
            cb = (clamp == b).astype(float)
            B.append(np.outer(ra, cb))           # B[i,j] = 1 iff clamp(i)==a and clamp(j)==b
    return B


def basis_sinusoidal(n, num=16, base=10000.0, **kw):
    # absolute sinusoidal features phi(i); full bilinear kernel phi(i)^T W phi(j)
    th = _freqs(num, base)
    idx = np.arange(n)
    feats = np.concatenate([np.sin(np.outer(idx, th)), np.cos(np.outer(idx, th))], axis=1)  # n x 2num
    m = feats.shape[1]
    B = []
    for a in range(m):
        for b in range(m):
            B.append(np.outer(feats[:, a], feats[:, b]))
    return B


def _rope_basis(n, th):
    ii = np.arange(n)
    diff = ii[:, None] - ii[None, :]  # i - j
    B = [np.ones((n, n))]
    for t in th:
        B.append(np.cos(diff * t))
        B.append(np.sin(diff * t))
    return B


def basis_rope(n, num=16, base=10000.0, **kw):
    # RoPE realizes RELATIVE kernels kappa(i-j): rotation-commuting structure.
    # span: { cos((i-j)theta_k), sin((i-j)theta_k) }_k  plus a constant.
    return _rope_basis(n, _freqs(num, base))


def basis_rope_lowfreq(n, num=16, scale=0.08, **kw):
    # Same relative family, frequencies shrunk by `scale` (matches model.ROPE_SCALE): the kernel is
    # too smooth to sharply resolve small offsets, so it should FAIL bounded-offset LG even though it
    # is still a "relative" PE -- a case the relative/absolute taxonomy cannot distinguish but alpha_L can.
    return _rope_basis(n, scale * _freqs(num))


def basis_alibi(n, **kw):
    # ALiBi: fixed monotone recency bias, learnable slope only.
    # span: { constant, (i - j) }   (causal: i - j >= 0)
    ii = np.arange(n)
    diff = (ii[:, None] - ii[None, :]).astype(float)
    return [np.ones((n, n)), diff]


PES = {
    "learned-abs":  basis_learned_abs,
    "nope":         basis_nope,
    "sinusoidal":   basis_sinusoidal,
    "rope":         basis_rope,
    "rope-lowfreq": basis_rope_lowfreq,
    "alibi":        basis_alibi,
}


# --------------------------------------------------------------------------------------
# The predictor.
# --------------------------------------------------------------------------------------

def _causal_constraints(task, n):
    """Return list of (i, j, value) target constraints over the query region, j <= i."""
    cons = []
    for i in task["qregion"](n):
        t = task["target"](n, i)
        if t < 0 or t > i:
            continue
        for j in range(i + 1):  # causal
            cons.append((i, j, 1.0 if j == t else 0.0))
    return cons


def _fit_coefficients(Bstack, rows_L0, steps=400, lr=0.1, wd=1e-4, neg_idx=()):
    """Fit kernel coefficients by cross-entropy so the head attends to the target at L0.

    Bstack: (K, n, n) basis tensor.  rows_L0: list of (i, target, n_candidates).
    neg_idx: indices whose coefficient is constrained <= 0 (e.g. canonical ALiBi recency slope,
             which is FIXED-direction, not freely learnable). Parameterized as -softplus(raw).
    Returns coef (K,).  Convex in the logits; sharpens like a real attention head.
    """
    K = Bstack.shape[0]
    raw = torch.zeros(K, requires_grad=True)
    neg = torch.zeros(K, dtype=torch.bool)
    for j in neg_idx:
        neg[j] = True
    opt = torch.optim.Adam([raw], lr=lr, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad()
        coef = torch.where(neg, -torch.nn.functional.softplus(raw), raw)
        P = torch.einsum("k,kij->ij", coef, Bstack)
        loss = 0.0
        for (i, t, ncand) in rows_L0:
            loss = loss + torch.nn.functional.cross_entropy(
                P[i, :ncand].unsqueeze(0), torch.tensor([t]))
        loss = loss / max(len(rows_L0), 1)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return torch.where(neg, -torch.nn.functional.softplus(raw), raw).detach()


# coefficients whose sign is fixed by the PE design (not freely learnable):
# canonical ALiBi's recency slope on the (i-j) term must be <= 0 (monotone recency bias).
NEG_IDX = {"alibi": (1,)}


def simulate(pe_name, task_name, L0, L, eps=1e-4, **pe_kw):
    """Fit the PE kernel to the task at length L0, transfer to L, return alpha_L and margin.

    Extra keyword args (e.g. num=, scale=, base=) are forwarded to the PE basis builder; this is
    used by the frequency/dimension sensitivity study (App. freq) and is backward compatible.
    """
    task = TASKS[task_name]
    B = PES[pe_name](L, L0=L0, **pe_kw)              # basis on the full L grid
    Bstack = torch.tensor(np.stack(B), dtype=torch.float32)

    # rows used to FIT: query positions present at L0
    rows_L0 = []
    for i in task["qregion"](L0):
        t = task["target"](L0, i)
        if 0 <= t <= i:
            rows_L0.append((i, t, i + 1))            # candidates j = 0..i (causal)
    coef = _fit_coefficients(Bstack, rows_L0, neg_idx=NEG_IDX.get(pe_name, ()))

    # transfer: SAME coefficients, length L
    P = torch.einsum("k,kij->ij", coef, Bstack).numpy()

    # score strict-unique-argmax accuracy + margin on extrapolation rows i in [L0, L)
    correct, margins, rows = 0, [], 0
    for i in task["qregion"](L):
        if i < L0:
            continue
        t = task["target"](L, i)
        if t < 0 or t > i:
            continue
        logits = P[i, : i + 1]
        rows += 1
        srt = np.sort(logits)[::-1]
        runnerup = srt[1] if len(srt) > 1 else srt[0] - 1.0
        # strict unique argmax: target must beat every competitor by > eps (kills tie artifacts)
        correct += int(logits[t] >= srt[0] - 1e-9 and (logits[t] - runnerup) > eps)
        margins.append(float(logits[t] - runnerup))
    alpha = correct / rows if rows else float("nan")
    margin = float(np.mean(margins)) if margins else float("nan")
    return dict(alpha=alpha, margin=margin, rows=rows)


if __name__ == "__main__":
    L0, L = 20, 60
    pes = ["learned-abs", "nope", "alibi", "sinusoidal", "rope"]
    tasks = ["prev", "skip3", "copy", "reverse", "first"]
    print(f"\nSimulation alignment  alpha_L  (train L0={L0}  ->  test L={L}),  higher = predicts LG\n")
    header = "PE \\ task".ljust(14) + "".join(t.ljust(10) for t in tasks)
    print(header)
    print("-" * len(header))
    table = {}
    for pe in pes:
        row = pe.ljust(14)
        for t in tasks:
            r = simulate(pe, t, L0, L)
            table[(pe, t)] = r["alpha"]
            row += f"{r['alpha']:.2f}".ljust(10)
        print(row)
    print("\n(rows = PEs, cols = tasks; copy/reverse are growing-offset, prev/skip3 bounded, first absolute)")
