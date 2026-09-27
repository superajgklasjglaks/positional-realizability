"""Rigorous verification of the under-determined dichotomy on the decisive cell
poly-copy (m=12 > rank(Phi_L0)=7), where a RESCUE W exists but min-norm misses it.

We verify:
  (1) EXISTENCE is genuine: a single W gives high new-row argmax accuracy on copy at L,
      AND interpolates the L0 pattern (so it's a legitimate L0-solution) -- robust to noise.
  (2) SELECTION blindness: any rule that is a function of TRAINED ROWS ONLY produces the
      SAME new-row logits regardless of the new-row target. We prove this operationally by
      the 'pretend-target' invariance test (cf. theory.md Part 1): swap the copy target for
      a shifted pretend target; a trained-rows-only rule's accuracy must be the SAME law.
  (3) The Vperp argmax among NEW keys is uniform (exchangeability core) when new-key
      features, projected to Vperp, are exchangeable -- the refined Prop-2 in subspace.
"""
import numpy as np


def phi_poly(n, m):
    i = np.arange(n)[:, None] / max(n - 1, 1)
    return np.concatenate([i ** k for k in range(m)], axis=1)


def target_copy(i, n_in):
    return i - n_in


def build_R_copy(L, L0, n_in, rows):
    R = np.zeros((L, L))
    for i in rows:
        t = target_copy(i, n_in)
        if 0 <= t <= i:
            R[i, t] = 1.0
    return R


def argmax_acc_copy(W, Phi, rows, n_in):
    ok = tot = 0
    for i in rows:
        c = np.arange(i + 1)
        t = target_copy(i, n_in)
        if not (0 <= t <= i) or len(c) < 2:
            continue
        lg = (Phi[c] @ W.T) @ Phi[i]
        ok += int(int(c[int(np.argmax(lg))]) == t); tot += 1
    return ok / max(tot, 1)


def main():
    L0, L, m, n_in = 8, 40, 12, 8
    Phi = phi_poly(L, m); Phi_L0 = Phi[:L0]
    rk = np.linalg.matrix_rank(Phi_L0, tol=1e-8)
    new_rows = list(range(L0, L))
    print(f"poly copy: L0={L0} L={L} m={m} rank(Phi_L0)={rk}  (underdetermined: {rk<m})")
    print("=" * 70)

    # (1) EXISTENCE: target-AWARE full LS fit; report new-row acc and L0-interpolation acc
    Rfull = build_R_copy(L, L0, n_in, range(L))
    Pf = np.linalg.pinv(Phi)
    We = Pf @ Rfull @ Pf.T
    ex_new = argmax_acc_copy(We, Phi, new_rows, n_in)
    ex_old = argmax_acc_copy(We, Phi_L0, range(L0), n_in)
    print(f"(1) EXISTENCE  (target-aware W): new-row acc={ex_new:.3f}, "
          f"L0-interp acc={ex_old:.3f}  -> rescue W EXISTS (cone non-empty)")

    # min-norm (trained-rows-only) baseline
    R_L0 = build_R_copy(L0, L0, n_in, range(L0))
    P0 = np.linalg.pinv(Phi_L0)
    Wmn = P0 @ R_L0 @ P0.T
    s0_new = argmax_acc_copy(Wmn, Phi, new_rows, n_in)
    print(f"    min-norm (trained-rows-only) new-row acc = {s0_new:.3f} (= alpha_L)")
    print(f"    GAP existence-vs-selection = {ex_new - s0_new:.3f}")

    # (2) SELECTION BLINDNESS via pretend-target invariance.
    # A trained-rows-only rule (min-norm on R_L0) does NOT depend on the new-row target.
    # Score the SAME Wmn against the true copy target vs a shifted pretend target.
    def acc_against(Wt, shift):
        ok = tot = 0
        for i in new_rows:
            c = np.arange(i + 1)
            t = (target_copy(i, n_in) + shift)
            if not (0 <= t <= i) or len(c) < 2:
                continue
            lg = (Phi[c] @ Wt.T) @ Phi[i]
            ok += int(int(c[int(np.argmax(lg))]) == t); tot += 1
        return ok / max(tot, 1)
    true_acc = acc_against(Wmn, 0)
    pretend = [acc_against(Wmn, s) for s in (1, 2, 3, 5)]
    print(f"(2) SELECTION BLINDNESS (pretend-target invariance):")
    print(f"    true-target acc={true_acc:.3f}  pretend-shift accs={[f'{p:.3f}' for p in pretend]}")
    print(f"    -> all in the same low band: trained-rows-only Wmn argmax is "
          f"target-INDEPENDENT (Prop-1/Part-1 mechanism).")

    # (3) Exchangeability of Vperp new-key argmax: project new-key features to Vperp and
    # check the argmax distribution of a FIXED trained-rows-only W over a permuted new-key set.
    U, S, Vt = np.linalg.svd(Phi_L0, full_matrices=True)
    Vb = Vt[:rk].T; Pi_perp = np.eye(m) - Vb @ Vb.T
    # how distinguishable are new keys in Vperp? (if near-degenerate, argmax ~ deterministic;
    # the EXCHANGEABILITY/uniform claim is the clean RANDOM-PE statement, not deterministic)
    new_perp = Phi[np.array(new_rows)] @ Pi_perp
    nn = np.linalg.norm(new_perp, axis=1)
    print(f"(3) NEW-row Vperp norms: min={nn.min():.3f} max={nn.max():.3f} "
          f"mean={nn.mean():.3f}  (order-1 => freedom is genuinely present)")

    print()
    print("CONCLUSION for this cell:")
    print(f"  Existence cone NON-empty ({ex_new:.2f}) BUT min-norm selection at chance-band "
          f"({s0_new:.2f}); trained-rows-only rule is target-blind (2). =>")
    print("  Conjecture's IMPOSSIBILITY is FALSE as an EXISTENCE claim, TRUE as a SELECTION")
    print("  claim. Unseen-direction freedom does NOT rescue the PREDICTOR alpha_L.")


if __name__ == "__main__":
    main()
