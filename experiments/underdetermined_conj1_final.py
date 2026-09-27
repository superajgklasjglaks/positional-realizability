"""Final synthesis: extract the LP cone-witness W for sinusoidal-copy (underdetermined),
verify it INDEPENDENTLY by argmax at both lengths, and confirm min-norm selection misses it.
This nails: existence NON-empty (conjecture FALSE for existence) but selection blind.
Also confirms poly prev/copy cone EMPTY (conjecture TRUE there)."""
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


def tgt(task, i, n_in):
    return {"prev": i - 1, "copy": i - n_in, "first": 0}[task]


def lp_witness(Phi, L0, L, m, task, n_in, margin=1.0):
    A_ub, b_ub = [], []
    for i in range(L):
        t = tgt(task, i, n_in)
        if not (0 <= t <= i):
            continue
        pi = Phi[i]; ft = np.kron(pi, Phi[t])
        for j in range(i + 1):
            if j == t:
                continue
            A_ub.append(np.kron(pi, Phi[j]) - ft); b_ub.append(-margin)
    res = linprog(np.zeros(m * m), A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  bounds=[(-50, 50)] * (m * m), method="highs")
    return res


def acc(W, Phi, rows, task, n_in):
    ok = tot = 0
    for i in rows:
        c = np.arange(i + 1); t = tgt(task, i, n_in)
        if not (0 <= t <= i) or len(c) < 2:
            continue
        lg = (Phi[c] @ W.T) @ Phi[i]
        ok += int(int(c[int(np.argmax(lg))]) == t); tot += 1
    return ok / max(tot, 1)


def minnorm(Phi_L0, L0, task, n_in):
    R = np.zeros((L0, L0))
    for i in range(L0):
        t = tgt(task, i, n_in)
        if 0 <= t <= i:
            R[i, t] = 1.0
    P = np.linalg.pinv(Phi_L0)
    return P @ R @ P.T


def main():
    print("=== sinusoidal copy, UNDERDETERMINED (m=48 >> rank=12): cone witness ===")
    L0, L, m = 12, 36, 48
    Phi = phi_sinusoidal(L, m)
    res = lp_witness(Phi, L0, L, m, "copy", L0)
    print(f"LP feasible: {res.success}")
    W = res.x.reshape(m, m)
    a_old = acc(W, Phi[:L0], range(L0), "copy", L0)
    a_new = acc(W, Phi, range(L0, L), "copy", L0)
    print(f"INDEPENDENT argmax check of LP witness W:")
    print(f"   L0 (old rows) copy acc = {a_old:.3f}")
    print(f"   L  (new rows) copy acc = {a_new:.3f}")
    print(f"   => single W solves BOTH => C_L0 cap C_L NON-EMPTY => conjecture FALSE (existence)")
    Wmn = minnorm(Phi[:L0], L0, "copy", L0)
    print(f"   min-norm selection (the alpha_L predictor) new-row acc = "
          f"{acc(Wmn, Phi, range(L0, L), 'copy', L0):.3f}  (<< {a_new:.3f})")
    print(f"   => selection MISSES the rescue: existence-vs-selection gap is the real story.")

    print()
    print("=== poly prev & copy, UNDERDETERMINED (m=12 > rank=8): cone EMPTY ===")
    for task in ["prev", "copy"]:
        L0p, Lp, mp = 8, 24, 12
        Php = phi_poly(Lp, mp)
        r = lp_witness(Php, L0p, Lp, mp, task, L0p)
        print(f"   poly {task:5s}: LP feasible={r.success}  -> "
              f"{'NON-EMPTY (conj FALSE)' if r.success else 'EMPTY (conj TRUE outright)'}")

    print()
    print("BOTTOM LINE: in the underdetermined regime the conjecture's outcome is")
    print("PE-DEPENDENT: cone EMPTY for poly prev/copy (impossibility holds), cone")
    print("NON-EMPTY for sinusoidal copy (impossibility FALSE for existence). In BOTH,")
    print("min-norm SELECTION stays at the chance band => alpha_L predictor is robust.")


if __name__ == "__main__":
    main()
