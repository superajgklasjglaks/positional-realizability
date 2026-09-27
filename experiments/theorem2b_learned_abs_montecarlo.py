"""Monte-Carlo confirmation of Theorem 2b (learned-absolute case).

PROBABILISTIC MODEL
-------------------
Learned-absolute PE: phi(i) = v_i in R^m.
  * Trained (FIXED) rows:    v_0, ..., v_{L0-1}   -- constrained by training at L0.
  * Untrained (RANDOM) rows: v_{L0}, ..., v_{L-1} -- i.i.d. draws from a continuous,
    rotation-invariant law mu (standard Gaussian here). "Untrained" is modeled as
    "unconstrained by the L0 signal, isotropic prior".

The head selects an interaction matrix W by ANY rule that is a function of the
TRAINED rows ONLY (invariant to the untrained coordinates). Canonical example:
the min-norm interpolant W* = Phi_{L0}^+ R_{L0} (Phi_{L0}^+)^T (the GD/ridgeless
limit). On a NEW query row i >= L0 the realized logits are

        a_{i, .} = v_i^T W phi(.) = v_i^T g(.),    g(j) := W phi(j),

with v_i a FRESH isotropic draw INDEPENDENT of W, of g, and -- crucially -- of the
correct target index t_L(i).

WHAT IS RANDOM / WHAT IS FIXED
  random : the untrained features v_{L0..L-1} (incl. the query v_i).
  fixed  : trained features v_{0..L0-1}, the trained pattern R_{L0}, the selection
           rule, and the target function t_L(.).

CLAIM
  (Core, fully general) The new-row argmax  argmax_{j<=i} a_{ij}  is INDEPENDENT of
  the correct target t_L(i): its law is a fixed distribution q over candidate keys
  determined by W and the feature geometry, with no dependence on which key is the
  truth. Hence P[correct on row i] = q_{t_L(i)} cannot track the truth.

  (Chance, under exchangeability) When the competitor keys are exchangeable -- e.g.
  the new keys j>=L0 all carry i.i.d. isotropic features -- the argmax among them is
  UNIFORM, so on a row whose target is a new key, P[correct] = 1/#newcandidates =
  chance, and ties have probability 0 (continuity of mu). Therefore E[alpha_L] =
  chance.

THIS IS UNDERDETERMINATION, NOT CONE-EMPTINESS
  A W that makes EVERY new row correct exists just as readily as one that fails
  (we exhibit one per trial). The target pattern is realizable; what fails is that
  the SELECTION RULE cannot see the new-row target, so the realized argmax is
  target-independent. (The false 'target lies outside colspan(Phi_{L0})' argument is
  a non-sequitur and is NOT what we prove.)

Run:
  python \
      experiments/theorem2b_learned_abs_montecarlo.py
"""

import numpy as np


def min_norm_W(Phi_L0, R_L0):
    """W* = Phi_L0^+ R_L0 (Phi_L0^+)^T -- depends ONLY on trained rows."""
    Pinv = np.linalg.pinv(Phi_L0)
    return Pinv @ R_L0 @ Pinv.T


def target(task, i, L0, n_in):
    if task == "prev":   # prev-token: t = i-1  (a NEW key once i >= L0+1)
        return i - 1
    if task == "copy":   # copy: t = i - n_in  (an OLD key when n_in >= L0-... )
        return i - n_in
    raise ValueError(task)


def run(task, L0=20, L=60, m=24, n_in=20, trials=8000, selection="min_norm",
        rng=None):
    """Monte-Carlo over the untrained features.

    Reports, on the new rows i in [L0, L):
      A. realized accuracy of the SELECTED W's argmax vs. the true target.
      B. the same with a DIFFERENT (shifted) 'pretend target' -> tests
         target-independence: distribution must be unchanged.
      C. accuracy restricted to NEW candidate keys j>=L0 vs. uniform chance
         1/#newcands (exchangeability => uniform).
      D. fraction of trials in which a RESCUING W exists (=> not cone-empty).
    """
    rng = rng or np.random.default_rng(0)
    new_rows = list(range(L0, L))

    # ----- FIXED trained block (any full-rank features + any L0-realizable R) -----
    # m >= L0 so the trained pattern is exactly interpolated (Theorem 1 hyp (i)).
    Phi_L0 = rng.standard_normal((L0, m))
    R_L0 = np.zeros((L0, L0))
    for i in range(1, L0):
        R_L0[i, i - 1] = 1.0           # a prev-token-like trained pattern (arbitrary)

    acc_true = []        # A: argmax == true target
    acc_pretend = []     # B: argmax == shifted 'pretend' target (independence control)
    acc_new = []         # C: among NEW keys only, argmax == true target
    chance_new = []      # C baseline: 1/#newcands averaged over scored rows
    rescuable = []       # D

    for _ in range(trials):
        Phi = np.empty((L, m))
        Phi[:L0] = Phi_L0
        Phi[L0:] = rng.standard_normal((L - L0, m))   # untrained: i.i.d. isotropic

        if selection == "min_norm":
            W = min_norm_W(Phi_L0, R_L0)              # trained-rows-only
        elif selection == "fixed_symmetric":          # any other invariant rule
            S = rng.standard_normal((m, m)); W = S + S.T
        else:
            raise ValueError(selection)

        G = Phi @ W.T            # g(j)^T stacked: row j = (W phi(j))^T   (L x m)

        ct = cp = 0; nt = 0
        cn = 0; ndenom = 0
        for i in new_rows:
            vi = Phi[i]
            cands = np.arange(i + 1)                  # causal keys j <= i
            logits = G[cands] @ vi
            am = int(cands[int(np.argmax(logits))])
            t = target(task, i, L0, n_in)
            ct += int(am == t)
            # pretend the truth is a different LEGAL key (shift within candidates)
            t_pre = int(cands[(int(np.where(cands == t)[0][0]) + 7) % len(cands)])
            cp += int(am == t_pre)
            nt += 1
            # restrict to the OFF-DIAGONAL new keys N_i = {L0 <= j < i}: the self-key j=i is
            # NOT exchangeable with them (it contributes the quadratic v_i^T W v_i), so it is excluded.
            new_c = cands[(cands >= L0) & (cands < i)]
            if t >= L0 and len(new_c) >= 2:
                amn = int(new_c[int(np.argmax(G[new_c] @ vi))])
                cn += int(amn == t)
                chance_new.append(1.0 / len(new_c))
                ndenom += 1

        acc_true.append(ct / nt)
        acc_pretend.append(cp / nt)
        if ndenom:
            acc_new.append(cn / ndenom)

        # ----- D: a rescuing W EXISTS (not cone-emptiness) -----
        R_new = np.zeros((len(new_rows), L))
        for r, i in enumerate(new_rows):
            R_new[r, target(task, i, L0, n_in)] = 1.0
        Phi_new = Phi[np.array(new_rows)]
        W_rescue = np.linalg.pinv(Phi_new) @ R_new @ np.linalg.pinv(Phi).T
        ok = True
        for i in new_rows:
            cands = np.arange(i + 1)
            lg = (Phi[cands] @ W_rescue.T) @ Phi[i]
            if int(cands[int(np.argmax(lg))]) != target(task, i, L0, n_in):
                ok = False; break
        rescuable.append(ok)

    acc_true = np.array(acc_true); acc_pretend = np.array(acc_pretend)
    acc_new = np.array(acc_new)
    return dict(
        task=task, selection=selection, trials=trials,
        acc_true=float(acc_true.mean()), acc_true_se=float(acc_true.std()/np.sqrt(trials)),
        acc_pretend=float(acc_pretend.mean()), acc_pretend_se=float(acc_pretend.std()/np.sqrt(trials)),
        acc_new_keys=float(acc_new.mean()) if len(acc_new) else float("nan"),
        chance_new_keys=float(np.mean(chance_new)) if chance_new else float("nan"),
        frac_rescuable=float(np.mean(rescuable)),
    )


def main():
    rng = np.random.default_rng(20260624)
    print("Theorem 2b (learned-absolute) -- Monte-Carlo confirmation")
    print("=" * 72)
    all_ok = True
    for selection in ["min_norm", "fixed_symmetric"]:
        print(f"\n=== selection rule (uses TRAINED rows only): {selection} ===")
        for task in ["prev", "copy"]:
            r = run(task, selection=selection, rng=rng)
            print(f"\n  task = {task}")
            print(f"   A. realized new-row acc (vs TRUE target)   = "
                  f"{r['acc_true']:.4f} ± {r['acc_true_se']:.4f}")
            print(f"   B. acc vs SHIFTED 'pretend' target          = "
                  f"{r['acc_pretend']:.4f}   "
                  f"(~= A  =>  argmax INDEPENDENT of target)")
            print(f"   C. acc among NEW keys (exchangeable set)    = "
                  f"{r['acc_new_keys']:.4f}   "
                  f"vs uniform chance 1/#newcands = {r['chance_new_keys']:.4f}")
            print(f"   D. frac trials a RESCUING W exists          = "
                  f"{r['frac_rescuable']:.3f}   "
                  f"({'>0 => NOT cone-empty' if r['frac_rescuable']>0 else 'cone-empty?'})")
            # Assertions: (C) realized new-key acc == uniform chance;
            # (D) rescuing W exists.
            if not (abs(r['acc_new_keys'] - r['chance_new_keys'])
                    < 0.01 + 3 * np.sqrt(r['chance_new_keys'] * (1 - r['chance_new_keys'])
                                         / (r['trials']))):
                all_ok = False; print("      [WARN] new-key acc not at chance")
            if r['frac_rescuable'] <= 0:
                all_ok = False; print("      [WARN] no rescuing W found")
    print("\n" + "=" * 72)
    print("RESULT:", "all checks passed." if all_ok else "SOME CHECKS FAILED (see [WARN] above).")
    print("  Checks attempted:")
    print("  - (A vs B is reported for information only: equal hit rates on two labels is NOT a")
    print("     valid test of target-independence, so no assertion is made on it),")
    print("  - among exchangeable new keys the argmax is uniform = chance (C),")
    print("  - a rescuing W exists (D)  =>  underdetermination, NOT cone-emptiness.")
    if not all_ok:
        sys.exit(1)


import sys

if __name__ == "__main__":
    main()
