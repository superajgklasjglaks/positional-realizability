"""DECISIVE test: is the intersection cone C_{L0} cap C_L EMPTY in the under-determined regime?

The conjecture is about C_{L0} cap C_L = empty: a SINGLE W that solves L0 AND L.
A 'rescue' that abandons the L0 fit is NOT a counterexample to LG (it wouldn't be learned at L0).

Construction respecting BOTH:
  Pin the seen block A = W*|_V (forced by L0 interpolation within V).  Then SEARCH the FREE
  blocks (B,C,D touching V^perp) to satisfy the NEW-row argmax constraints, WITHOUT disturbing
  old rows.  Old rows i<L0: keys j<=i<L0 are all in the training block; their logit is
  u_i^T A u_j + (Vperp terms). If old-row features have a Vperp component, B,C,D leak into old
  rows too -- so we must keep old-row argmax intact as a constraint.

We solve a feasibility LP/least-squares for the free blocks subject to:
  new-row margins:  a_{i,t} - a_{i,j} >= 1  for all competitors j != t, new rows i
  old-row margins:  preserve correct old-row argmax with margin >= 0.1
and report whether a feasible W exists (=> cone non-empty => conjecture FALSE for existence)
or not (=> cone empty => conjecture TRUE).

Then separately: even if non-empty, can a TRAINED-ROWS-ONLY rule select it? (already shown: no.)
"""
import numpy as np
from scipy.optimize import linprog


def phi_poly(n, m):
    i = np.arange(n)[:, None] / max(n - 1, 1)
    return np.concatenate([i ** k for k in range(m)], axis=1)


def phi_sinusoidal(n, m):
    pos = np.arange(n)[:, None]; half = m // 2
    div = np.exp(np.arange(half) * (-np.log(10000.0) / max(half - 1, 1)))
    ang = pos * div[None, :]
    out = np.empty((n, m)); out[:, 0::2] = np.sin(ang); out[:, 1::2] = np.cos(ang)
    return out


def tgt(task, i, n_in):
    return {"prev": i - 1, "skip2": i - 2, "first": 0, "copy": i - n_in}[task]


def cone_feasible(Phi, L0, L, m, task, n_in):
    """LP over W (m*m vars, vectorized) for joint L0+L argmax margins.
    a_{ij} = phi_i^T W phi_j = (phi_i kron phi_j) vec(W).  Linear in vec(W)."""
    rows = list(range(L))
    A_ub = []; b_ub = []
    margin = 1.0
    for i in rows:
        t = tgt(task, i, n_in)
        if not (0 <= t <= i):
            continue
        pi = Phi[i]
        feat_t = np.kron(pi, Phi[t])
        for j in range(i + 1):
            if j == t:
                continue
            feat_j = np.kron(pi, Phi[j])
            # want a_t - a_j >= margin  ->  -(a_t - a_j) <= -margin
            A_ub.append(feat_j - feat_t)
            b_ub.append(-margin)
    A_ub = np.array(A_ub); b_ub = np.array(b_ub)
    # feasibility: minimize 0 s.t. A_ub w <= b_ub, w bounded
    n = m * m
    res = linprog(c=np.zeros(n), A_ub=A_ub, b_ub=b_ub,
                  bounds=[(-50, 50)] * n, method="highs")
    return res.success, res


def main():
    cases = [
        ("poly", phi_poly, "prev", 8, 24, 12),
        ("poly", phi_poly, "copy", 8, 24, 12),
        ("poly", phi_poly, "first", 8, 24, 12),
        ("sinusoidal", phi_sinusoidal, "prev", 12, 36, 48),
        ("sinusoidal", phi_sinusoidal, "copy", 12, 36, 48),
        ("sinusoidal", phi_sinusoidal, "first", 12, 36, 48),
    ]
    print("JOINT cone feasibility  C_{L0} cap C_L  (single W solving BOTH lengths)")
    print("=" * 72)
    print(f"{'PE':11s} {'task':6s} {'m':>3s} {'L0':>3s} {'L':>3s} {'rk':>3s} "
          f"{'underdet':>8s}  {'C_L0 ∩ C_L':>11s}")
    for name, fn, task, L0, L, m in cases:
        Phi = fn(L, m)
        rk = int(np.linalg.matrix_rank(Phi[:L0], tol=1e-8))
        ok, res = cone_feasible(Phi, L0, L, m, task, L0)
        verdict = "NON-EMPTY" if ok else "EMPTY"
        print(f"{name:11s} {task:6s} {m:3d} {L0:3d} {L:3d} {rk:3d} "
              f"{str(rk<m):>8s}  {verdict:>11s}")
    print()
    print("NON-EMPTY  => a single W solves L0 AND L => conjecture IMPOSSIBILITY is FALSE")
    print("            for existence; LG fails only because SELECTION can't pick it.")
    print("EMPTY      => no W solves both => conjecture-style impossibility holds outright.")


if __name__ == "__main__":
    main()
