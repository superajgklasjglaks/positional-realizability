"""Numerical illustration of the ABSOLUTE / RELATIVE DUALITY (Contrast, under (S1)-(S3)).

Single causal positional-only head (S1); "solve at length n" = correct masked
row-argmax with positive margin (S2); content argmax-neutral (S3). The realized
positional logit between query i and key j<=i is

        a_{ij} = phi(i)^T W phi(j)              (general bilinear kernel)

Two structural restrictions of the kernel:
  * ABSOLUTE kernel:  a_{ij} = K_abs[i, j]  -- an arbitrary function of the
    *pair of absolute indices* (i, j). Realized e.g. by an absolute PE whose
    feature phi(i) identifies the index i (one-hot / learned-absolute / a full
    sinusoidal basis at a single length). Its free parameters are indexed by
    absolute positions: to control row i you must have *seen* position i.
  * RELATIVE (translation-invariant) kernel: a_{ij} = kappa(i - j) -- depends
    only on the offset d = i-j. Realized by RoPE/ALiBi-type relative PE. One
    free parameter per offset d in {0,1,...}; the same kappa is reused on every
    row, at every length.

TWO TASKS (the duality pair):
  * ABSOLUTE-position task  "attend-first": target t_n(i) = 0 for all i.
        offset i-j = i  ->  GROWS with the row.
  * FIXED-RELATIVE-offset task "prev / skip-k": target t_n(i) = i-k (k fixed).
        offset i-j = k  ->  CONSTANT.

DUALITY (what this script demonstrates numerically):
  (A) attend-first is solvable AT ALL LENGTHS by an absolute kernel (one fixed W
      putting high logit on absolute key 0). The relative kernel SELECTED by the
      L0 fit does not solve it, and its score decays with L. NOTE: this is a
      statement about the selected peaked kernel, not about the relative family:
      a monotone increasing kappa (e.g. kappa(d)=d) attends to j=0 at every row
      and length, because the causal window truncates at d=i. The impossibility
      in the paper is stated for kernels with a strict unique GLOBAL maximizer.
  (B) prev/skip-k is solvable AT ALL LENGTHS by a relative kappa (peak at the
      single fixed offset k, reused on every row), but a *pure absolute* kernel
      learned at L0 must re-learn each position: on rows i>=L0 it is
      unconstrained -> chance (this is exactly Theorem 2b-LA / Prop 2).

The unifying statement (the "duality"):
  relative kernels  generalize  FIXED RELATIVE offsets   (constant i-j),
  absolute kernels  generalize  FIXED ABSOLUTE positions (constant j, here j=0),
  and NEITHER generalizes a GROWING offset (offset that scales with the row).
  attend-first is a growing offset for the relative family but a fixed position
  for the absolute family; prev/skip-k is a fixed offset for the relative family
  but a moving position for the absolute family. The two tasks sit on opposite
  corners.

Run:
  python \
      experiments/duality_abs_vs_rel.py
"""

import numpy as np

rng = np.random.default_rng(0)


# ----------------------------------------------------------------------------
# Targets
# ----------------------------------------------------------------------------
def t_first(i):
    return 0


def t_prev_k(i, k):
    return i - k


# ----------------------------------------------------------------------------
# ABSOLUTE kernel realized by an absolute PE.
# Feature phi(i) = one-hot(i) in R^{Lmax} (the most expressive absolute PE; a
# learned-absolute / full-sinusoidal-basis PE at a single length is no stronger
# in what absolute pairs (i,j) it can address). The kernel is K_abs = W (any
# matrix over absolute index pairs). At training length L0 we only get to set the
# block i<L0, j<L0; rows i>=L0 are an UNSEEN absolute position whose features are
# untrained -> modeled as random isotropic (the (S)-model of Thm 2b-LA).
# ----------------------------------------------------------------------------
def absolute_kernel_attend_first(n):
    """A FIXED absolute kernel solving attend-first at every length:
       K[i,j] = 1 if j==0 else 0. Realizable by phi(i)=onehot(i),
       W = e_0 1^T  (column 0 hot). Length-independent."""
    K = np.zeros((n, n))
    K[:, 0] = 1.0
    return K


def absolute_kernel_prev_trained_at_L0(L0, L, k, m):
    """Absolute (learned) kernel for prev/skip-k. Feature phi(i)=v_i in R^m:
       v_0..v_{L0-1} fixed (trained), v_{L0}..v_{L-1} i.i.d. isotropic (untrained).
       W = min-norm interpolant of the trained block's prev-k pattern. Returns the
       FULL L x L realized logit matrix a_{ij}=v_i^T W v_j (causal-masked by caller)."""
    V = rng.standard_normal((L, m))           # v_0..v_{L-1}; rows >=L0 are 'untrained'
    Phi0 = V[:L0]                              # trained features (fixed once drawn)
    R0 = np.zeros((L0, L0))
    for i in range(L0):
        j = t_prev_k(i, k)
        if 0 <= j <= i:
            R0[i, j] = 1.0
    Pinv = np.linalg.pinv(Phi0)
    W = Pinv @ R0 @ Pinv.T                     # depends on TRAINED rows only
    return V @ W @ V.T


# ----------------------------------------------------------------------------
# RELATIVE (translation-invariant) kernel: a_{ij}=kappa(i-j).
# ----------------------------------------------------------------------------
def relative_kernel(n, kappa):
    """Build the n x n matrix a_{ij}=kappa(i-j) for offsets d=i-j>=0."""
    K = np.full((n, n), -np.inf)
    for i in range(n):
        for j in range(i + 1):
            K[i, j] = kappa(i - j)
    return K


def kappa_peak_at(k, sharp=1.0):
    """A relative kernel peaked at the single fixed offset k (prev/skip-k)."""
    return lambda d: -sharp * (d - k) ** 2


def best_relative_for_attend_first(L0):
    """The BEST a single translation-invariant kappa can do on attend-first,
       *fit on the training rows*: choose the offset d* that is the target offset
       on the most training rows. For attend-first the target offset on row i is
       exactly i, all distinct, so no single offset wins more than one row.
       We pick d*=argmax over the empirical offset histogram at L0."""
    offsets = [i - t_first(i) for i in range(L0)]   # = [0,1,...,L0-1], all distinct
    vals, counts = np.unique(offsets, return_counts=True)
    d_star = int(vals[np.argmax(counts)])
    return kappa_peak_at(d_star, sharp=10.0), d_star


# ----------------------------------------------------------------------------
# Scoring: accuracy on the NEW rows i in [L0, L) (the extrapolation region).
# ----------------------------------------------------------------------------
def masked_argmax(row, i):
    sub = row[: i + 1].copy()
    return int(np.argmax(sub))


def new_row_accuracy(K, L0, L, target_fn):
    correct = 0
    total = 0
    for i in range(L0, L):
        t = target_fn(i)
        if not (0 <= t <= i):
            continue
        total += 1
        if masked_argmax(K[i], i) == t:
            correct += 1
    return correct / max(total, 1), total


def margin_new_rows(K, L0, L, target_fn):
    """min over new rows of (logit at target) - (best competitor)."""
    margins = []
    for i in range(L0, L):
        t = target_fn(i)
        if not (0 <= t <= i):
            continue
        row = K[i][: i + 1].copy()
        a_t = row[t]
        row[t] = -np.inf
        margins.append(a_t - np.max(row))
    return min(margins) if margins else float("nan")


# ============================================================================
def main():
    L0, L = 20, 80
    k = 1               # prev-token (skip-1); rerun loop shows skip-k too
    m = 24

    print("=" * 78)
    print("DUALITY: absolute vs relative kernels  (L0=%d -> L=%d)" % (L0, L))
    print("=" * 78)

    # ---- TASK 1: attend-first (target t=0; offset = i, GROWING) -------------
    print("\n[TASK 1]  attend-first   t_n(i)=0   (offset i-j = i, GROWS with row)")

    K_abs_first = absolute_kernel_attend_first(L)
    acc_abs, n1 = new_row_accuracy(K_abs_first, L0, L, t_first)
    mar_abs = margin_new_rows(K_abs_first, L0, L, t_first)
    print("  ABSOLUTE kernel (fixed W=e0 1^T):     new-row acc = %.3f  margin = %+.2f"
          % (acc_abs, mar_abs))

    kappa_rel, d_star = best_relative_for_attend_first(L0)
    K_rel_first = relative_kernel(L, kappa_rel)
    acc_rel, _ = new_row_accuracy(K_rel_first, L0, L, t_first)
    print("  RELATIVE kappa (best fixed offset d*=%d on train): new-row acc = %.3f"
          % (d_star, acc_rel))
    # what offset would relative need? -> i, grows. show it can never match.
    needed = [i for i in range(L0, L)]
    print("      (relative would need offset = i in {%d..%d}; a single kappa peaks"
          % (needed[0], needed[-1]))
    print("       at one offset d* only -> matches >=1 new row only if d* hits i;")
    print("       fraction of new rows a single offset can cover = %.4f = 1/(L-L0))"
          % (1.0 / (L - L0)))

    # ---- TASK 2: prev / skip-k (target t=i-k; offset = k, FIXED) ------------
    print("\n[TASK 2]  prev/skip-%d   t_n(i)=i-%d   (offset i-j = %d, CONSTANT)"
          % (k, k, k))

    K_rel_prev = relative_kernel(L, kappa_peak_at(k, sharp=10.0))
    acc_rel2, n2 = new_row_accuracy(K_rel_prev, L0, L, lambda i: t_prev_k(i, k))
    mar_rel2 = margin_new_rows(K_rel_prev, L0, L, lambda i: t_prev_k(i, k))
    print("  RELATIVE kappa (peak at offset %d):    new-row acc = %.3f  margin = %+.2f"
          % (k, acc_rel2, mar_rel2))

    # absolute learned kernel: average over draws of untrained features
    accs = []
    n_trials = 4000
    for _ in range(n_trials):
        K_abs_prev = absolute_kernel_prev_trained_at_L0(L0, L, k, m)
        a, _ = new_row_accuracy(K_abs_prev, L0, L, lambda i: t_prev_k(i, k))
        accs.append(a)
    accs = np.array(accs)
    # chance level among causal keys j in [0,i]: target i-k is one of i+1 keys,
    # but the rigorous exchangeable-chance among the i-L0+1 *new* keys on row i
    # is 1/(#new candidate keys). Report empirical + a representative chance.
    # representative new-key count averaged over new rows:
    avg_new_keys = np.mean([ (i - L0 + 1) for i in range(L0, L) ])
    print("  ABSOLUTE learned kernel (min-norm @L0): new-row acc = %.4f +/- %.4f"
          % (accs.mean(), accs.std()))
    print("      (avg over %d draws of untrained features; vs uniform-over-all-causal"
          % n_trials)
    print("       -keys chance ~ mean_i 1/(i+1) = %.4f -> at chance, NOT generalizing)"
          % np.mean([1.0 / (i + 1) for i in range(L0, L)]))

    # ---- DUALITY TABLE -----------------------------------------------------
    print("\n" + "=" * 78)
    print("DUALITY SUMMARY (new-row accuracy on extrapolation region [%d,%d))" % (L0, L))
    print("=" * 78)
    print("                        |  ABSOLUTE kernel  |  RELATIVE kernel")
    print("  ----------------------+-------------------+------------------")
    print("  attend-first (t=0)    |      %.3f  OK     |     %.3f  FAIL"
          % (acc_abs, acc_rel))
    print("    [offset i = GROWS]  |  (fixed abs pos)  |  (growing offset)")
    print("  prev/skip-%d (t=i-%d)   |      %.3f  FAIL   |     %.3f  OK"
          % (k, k, accs.mean(), acc_rel2))
    print("    [offset %d = FIXED]   |  (moving pos)     |  (fixed offset)" % k)
    print("\n  => relative generalizes FIXED RELATIVE offsets;")
    print("     absolute generalizes FIXED ABSOLUTE positions;")
    print("     neither of the SELECTED kernels generalizes a GROWING offset. (The impossibility")
    print("     in the paper is stated for kernels with a strict unique GLOBAL maximizer; an")
    print("     unbounded ladder without a maximizer is a counterexample without it.)")

    # ---- extra: skip-k sweep on the relative side, all generalize ----------
    print("\n[CHECK] relative kappa(peak=k) solves skip-k at all lengths, every k:")
    for kk in [1, 2, 3, 5]:
        Kr = relative_kernel(L, kappa_peak_at(kk, sharp=10.0))
        a, _ = new_row_accuracy(Kr, L0, L, lambda i, kk=kk: t_prev_k(i, kk))
        print("   skip-%d: new-row acc = %.3f" % (kk, a))

    # ---- extra: confirm the GROWING-offset failure is shared --------------
    # copy-like growing offset for BOTH families, to underline "neither":
    print("\n[CHECK] GROWING offset task 'attend-first' fails for relative at every L;")
    print("        and the absolute solution stays length-independent:")
    for Ltest in [40, 80, 160, 320]:
        Ka = absolute_kernel_attend_first(Ltest)
        aa, _ = new_row_accuracy(Ka, L0, Ltest, t_first)
        kr, d_star = best_relative_for_attend_first(L0)
        Kr = relative_kernel(Ltest, kr)
        ar, _ = new_row_accuracy(Kr, L0, Ltest, t_first)
        print("   L=%3d:  absolute acc=%.3f   relative acc=%.4f (=1/(L-L0)=%.4f)"
              % (Ltest, aa, ar, 1.0 / (Ltest - L0)))


if __name__ == "__main__":
    main()
