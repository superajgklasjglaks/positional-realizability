"""Conjecture 1 hard case: does UNSEEN-DIRECTION FREEDOM rescue LG, and can a
trained-rows-only SELECTION RULE pick the rescue?

Refined claim being tested (the proposed REFINED NECESSARY CONDITION):

  Decompose phi(i) = u_i + w_i,  u_i in V=rowspan(Phi_{L0}), w_i in V^perp.
  Realized new-row logit a_{ij} = phi(i)^T W phi(j).  Split W into blocks w.r.t. V (+) V^perp:
      W = [[A, B],[C, D]]   (A on V(x)V, B on V(x)Vperp, C,D touch Vperp).
  a_{ij} = u_i^T A u_j  +  u_i^T B w_j  +  w_i^T C u_j  +  w_i^T D w_j.

  * The seen block A is PINNED by interpolation at L0 (A acts on V where Phi_{L0} has full
    coords): A = W*|_V is forced (up to the L0 nullspace, which is empty within V).
  * Blocks B,C,D are FREE (the under-determined freedom).

  NECESSARY CONDITION FOR ANY SELECTION RULE THAT IS A FUNCTION OF TRAINED ROWS ONLY:
  the free blocks B,C,D are chosen WITHOUT seeing the new-row target t_L(i). The new keys
  j>=L0 carry features whose Vperp parts {w_j} are (for a generic/structured-but-unaligned
  PE) NOT alignable with t_L by a target-blind rule. So:

     If the correct new-row decision REQUIRES the free blocks (i.e. the pinned block A
     alone gives the WRONG argmax), then a target-blind selection rule rescues only by
     luck -> E[alpha_L] does not exceed what A|_V predicts.  =>  alpha_L (min-norm, B=C=D=0)
     is the honest predictor, and the conjecture-style impossibility holds for the SELECTION
     problem even though the EXISTENCE problem (cone) is non-empty.

We test three things per (PE, task):
  (E) EXISTENCE: a rescue W (any blocks) making new rows correct -- least-squares + checks.
  (S0) min-norm selection accuracy (B=C=D=0): the alpha_L predictor.
  (S1) BEST target-BLIND selection: pick B,C,D using ONLY trained rows (random / fixed
       symmetric / ridge), average accuracy. Does it beat S0 / reach E?
"""
import numpy as np


def phi_sinusoidal(n, m):
    pos = np.arange(n)[:, None]; half = m // 2
    div = np.exp(np.arange(half) * (-np.log(10000.0) / max(half - 1, 1)))
    ang = pos * div[None, :]
    out = np.empty((n, m)); out[:, 0::2] = np.sin(ang); out[:, 1::2] = np.cos(ang)
    return out


def phi_poly(n, m):
    i = np.arange(n)[:, None] / max(n - 1, 1)
    return np.concatenate([i ** k for k in range(m)], axis=1)


def target(task, i, n_in):
    return {"prev": i - 1, "skip2": i - 2, "first": 0}[task] if task != "copy" else i - n_in


def build_R(Phi, L0, task, n_in, rows):
    L = Phi.shape[0]; R = np.zeros((L, L))
    for i in rows:
        t = target(task, i, n_in)
        if 0 <= t <= i:
            R[i, t] = 1.0
    return R


def argmax_acc(W, Phi, rows, task, n_in):
    ok = tot = 0
    for i in rows:
        c = np.arange(i + 1)
        if len(c) < 2:
            continue
        t = target(task, i, n_in)
        if not (0 <= t <= i):
            continue
        lg = (Phi[c] @ W.T) @ Phi[i]
        ok += int(int(c[int(np.argmax(lg))]) == t); tot += 1
    return ok / max(tot, 1)


def main():
    rng = np.random.default_rng(7)
    cases = [
        ("sinusoidal", phi_sinusoidal, "prev", 20, 60, 64),
        ("sinusoidal", phi_sinusoidal, "skip2", 20, 60, 64),
        ("sinusoidal", phi_sinusoidal, "first", 20, 60, 64),
        ("poly", phi_poly, "prev", 8, 40, 12),
        ("poly", phi_poly, "first", 8, 40, 12),
        ("poly", phi_poly, "copy", 8, 40, 12),
    ]
    print(f"{'PE':11s} {'task':6s} {'m':>3s} {'rk':>3s}  "
          f"{'A_only(S0/aL)':>13s} {'best_blind':>11s} {'EXISTS(E)':>10s}  verdict")
    print("-" * 78)
    for name, fn, task, L0, L, m in cases:
        Phi = fn(L, m); Phi_L0 = Phi[:L0]
        rk = np.linalg.matrix_rank(Phi_L0, tol=1e-8)
        new_rows = list(range(L0, L))
        n_in = L0
        R_L0 = build_R(Phi_L0, L0, task, n_in, range(L0))
        P0 = np.linalg.pinv(Phi_L0)
        Wmn = P0 @ R_L0 @ P0.T                      # A-only, B=C=D=0
        s0 = argmax_acc(Wmn, Phi, new_rows, task, n_in)

        # EXISTENCE: full-rank LS fit to old+new pattern (uses all blocks)
        Rfull = build_R(Phi, L0, task, n_in, range(L))
        Pf = np.linalg.pinv(Phi)
        We = Pf @ Rfull @ Pf.T
        ex = argmax_acc(We, Phi, new_rows, task, n_in)

        # BEST target-blind selection: keep A pinned, draw B,C,D from trained-rows-only
        # rules; we add a random free perturbation supported off V(x)V and take the BEST
        # over many draws (an UPPER BOUND on any blind rule's typical accuracy; if even
        # the best random blind draw can't beat A_only systematically, selection is blind).
        U, S, Vt = np.linalg.svd(Phi_L0, full_matrices=True)
        Vb = Vt[:rk].T; Pi_V = Vb @ Vb.T; Pi_perp = np.eye(m) - Pi_V
        best_blind = s0; accs = []
        for _ in range(400):
            G = rng.standard_normal((m, m))
            # free part: anything touching V^perp (B,C,D), leave A = Wmn|_V intact
            Delta = Pi_V @ G @ Pi_perp + Pi_perp @ G @ Pi_V + Pi_perp @ G @ Pi_perp
            Wtry = Wmn + 3.0 * Delta
            a = argmax_acc(Wtry, Phi, new_rows, task, n_in)
            accs.append(a); best_blind = max(best_blind, a)
        mean_blind = float(np.mean(accs))

        verdict = ("MIRAGE (no Vperp use)" if abs(ex - s0) < 1e-6 else
                   "FREEDOM REAL; selection blind" if mean_blind <= s0 + 0.02 and ex > s0 + 0.05
                   else "freedom helps blind rule?")
        print(f"{name:11s} {task:6s} {m:3d} {rk:3d}  {s0:13.3f} "
              f"{mean_blind:6.3f}/{best_blind:4.3f} {ex:10.3f}  {verdict}")
    print()
    print("S0 = alpha_L (min-norm, free blocks 0).  best_blind = best of 400 random")
    print("target-BLIND free perturbations.  E = existence (full LS fit, sees target).")


if __name__ == "__main__":
    main()
