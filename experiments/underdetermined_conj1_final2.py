"""Corrected final test. The subtlety: 'solving L0' must use the L0-appropriate target.
For copy, the input length n_in differs by sequence length. We use a clean MIXED-LENGTH
training protocol so old rows are genuinely constrained:

  Training distribution: lengths in {L0_min..L0}. At each training length n, copy target
  is i - n_in(n) with n_in(n)=n//2 (first half input). The head must solve ALL training
  lengths with ONE W (that's what 'solves L0' means for a length-varying task).
  Test: length L, copy with n_in(L)=L//2, score new output rows.

For a FIXED-OFFSET task (prev: t=i-1) there is no such subtlety; old rows i>=1 are
constrained at L0 directly. We test prev too.

Cone C_train cap C_L feasibility via LP over vec(W); then min-norm selection accuracy.
"""
import numpy as np
from scipy.optimize import linprog


def phi_sinusoidal(n, m):
    pos = np.arange(n)[:, None]; half = m // 2
    div = np.exp(np.arange(half) * (-np.log(10000.0) / max(half - 1, 1)))
    ang = pos * div[None, :]
    out = np.empty((n, m)); out[:, 0::2] = np.sin(ang); out[:, 1::2] = np.cos(ang)
    return out


def phi_poly(n, m):
    i = np.arange(n)[:, None] / max(n - 1, 1)
    return np.concatenate([i ** k for k in range(m)], axis=1)


def copy_tgt(i, n):  # copy at sequence length n: output row i reads input row i - n//2
    return i - n // 2


def constraints_copy(Phi, n, m, out_rows, margin):
    """argmax margin constraints for copy at length n, over given output rows."""
    A, b = [], []
    for i in out_rows:
        t = copy_tgt(i, n)
        if not (0 <= t <= i):
            continue
        ft = np.kron(Phi[i], Phi[t])
        for j in range(i + 1):
            if j == t:
                continue
            A.append(np.kron(Phi[i], Phi[j]) - ft); b.append(-margin)
    return A, b


def main():
    m_sin, m_poly = 48, 12
    margin = 1.0
    # ----- COPY: mixed-length training {8,10,12} ; test L=36 -----
    for name, fn, m in [("sinusoidal", phi_sinusoidal, m_sin),
                        ("poly", phi_poly, m_poly)]:
        train_lengths = [8, 10, 12]; L0 = max(train_lengths); L = 36
        Phi = fn(L, m)
        rk = int(np.linalg.matrix_rank(Phi[:L0], tol=1e-8))
        A, b = [], []
        for n in train_lengths:                   # all training lengths, all output rows
            out_rows = range(n)
            Ai, bi = constraints_copy(Phi[:n], n, m, out_rows, margin)
            A += Ai; b += bi
        # test rows at length L: the NEW output rows beyond L0
        At, bt = constraints_copy(Phi, L, m, range(L0, L), margin)
        A_train = np.array(A); b_train = np.array(b)
        A_all = np.array(A + At); b_all = np.array(b + bt)
        res_tr = linprog(np.zeros(m * m), A_ub=A_train, b_ub=b_train,
                         bounds=[(-50, 50)] * (m * m), method="highs")
        res_all = linprog(np.zeros(m * m), A_ub=A_all, b_ub=b_all,
                          bounds=[(-50, 50)] * (m * m), method="highs")
        print(f"[copy] {name:10s} m={m} rank(Phi_L0)={rk} underdet={rk<m}")
        print(f"   C_train feasible          = {res_tr.success}")
        print(f"   C_train ∩ C_L feasible    = {res_all.success}  "
              f"-> {'NON-EMPTY: conj FALSE (existence)' if res_all.success else 'EMPTY: conj TRUE'}")
        if res_all.success:
            W = res_all.x.reshape(m, m)
            # independent argmax verification at a training length and at L
            def chk(n, rows):
                ok = tot = 0
                for i in rows:
                    t = copy_tgt(i, n); c = np.arange(i + 1)
                    if not (0 <= t <= i) or len(c) < 2:
                        continue
                    lg = (Phi[c] @ W.T) @ Phi[i]
                    ok += int(int(c[int(np.argmax(lg))]) == t); tot += 1
                return ok / max(tot, 1)
            print(f"   verify: train(n=12) acc={chk(12, range(12)):.2f}, "
                  f"L=36 new acc={chk(36, range(L0, L)):.2f}")

    # ----- PREV: fixed offset, train L0=12 ; test L=36 -----
    for name, fn, m in [("sinusoidal", phi_sinusoidal, m_sin),
                        ("poly", phi_poly, m_poly)]:
        L0, L = 12, 36
        Phi = fn(L, m)
        rk = int(np.linalg.matrix_rank(Phi[:L0], tol=1e-8))

        def prev_constr(rows, n):
            A, b = [], []
            for i in rows:
                t = i - 1
                if t < 1:
                    continue
                ft = np.kron(Phi[i], Phi[t])
                for j in range(i + 1):
                    if j == t:
                        continue
                    A.append(np.kron(Phi[i], Phi[j]) - ft); b.append(-margin)
            return A, b
        At, bt = prev_constr(range(1, L0), L0)
        An, bn = prev_constr(range(L0, L), L)
        ra = linprog(np.zeros(m * m), A_ub=np.array(At + An), b_ub=np.array(bt + bn),
                     bounds=[(-50, 50)] * (m * m), method="highs")
        print(f"[prev] {name:10s} m={m} rank(Phi_L0)={rk} underdet={rk<m}  "
              f"C_L0 ∩ C_L feasible={ra.success} -> "
              f"{'NON-EMPTY conj FALSE' if ra.success else 'EMPTY conj TRUE'}")


if __name__ == "__main__":
    main()
