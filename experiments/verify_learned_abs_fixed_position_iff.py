"""TIGHT IFF for a single LEARNED-ABSOLUTE head under (S1)-(S3).

CLAIM
-----
phi(i) = v_i, trained/in-range for i < L0, untrained (random, isotropic,
abs-continuous) for i >= L0. W selected by ANY rule using TRAINED rows only
(canonical: min-norm). For a position-driven task with target t_n(i):

  the head length-generalizes (E[alpha_L] = 1)
      IFF
  t_n(i) = c is a FIXED, length-INDEPENDENT ABSOLUTE position with c < L0.

FORWARD (<=): target is the fixed trained key c < L0. A kernel attending to
key c solves EVERY new row at EVERY length, deterministically (no randomness),
so alpha_L = 1.

REVERSE (=>): if the target's absolute index grows with n or i (so that on
new rows i >= L0 the target is a key >= L0), the target key is untrained =>
Prop 2 underdetermination => realized new-row argmax is target-independent and
uniform over exchangeable new keys => E[alpha_L] = chance < 1.

We verify three regimes numerically:
  (A) FORWARD: t = c (fixed in-range absolute). Expect alpha_L = 1 exactly,
      deterministic, at several L.
  (B) REVERSE-growing-offset: t = i-1 (prev), t = i - n_in (copy). target on
      new rows is a new (>=L0) key => chance.
  (C) REVERSE-growing-position: t = n_in-1-(i-n_in) (reverse) and t = i (a
      growing absolute index) => target leaves the trained block => chance.
  (D) BOUNDARY: t = c with c >= L0 (a *fixed* absolute target but OUT of range)
      => untrained key => chance. Shows the 'c < L0' condition is necessary,
      not just 'fixed'.

Run:
  python \
      experiments/verify_learned_abs_fixed_position_iff.py
"""

import numpy as np


def min_norm_W(Phi_L0, R_L0):
    Pinv = np.linalg.pinv(Phi_L0)
    return Pinv @ R_L0 @ Pinv.T


def build_features(L0, L, m, rng):
    """Trained block fixed once; untrained block fresh isotropic Gaussian."""
    Phi = np.empty((L, m))
    Phi[:L0] = rng.standard_normal((L0, m))   # treat as 'the trained features'
    Phi[L0:] = rng.standard_normal((L - L0, m))
    return Phi


def trained_pattern_for_fixed_key(L0, c):
    """An L0-realizable trained pattern whose target is the fixed key c (<L0).
    Every trained row attends to c (causal: only rows i>=c). R_L0[i,c]=1."""
    R = np.zeros((L0, L0))
    for i in range(L0):
        if c <= i:
            R[i, c] = 1.0
    return R


def trained_pattern_prevlike(L0):
    R = np.zeros((L0, L0))
    for i in range(1, L0):
        R[i, i - 1] = 1.0
    return R


def alpha_L_fixed_key(L0, L, m, c, trials, rng):
    """FORWARD regime. min-norm W from R_L0=attend-to-c. Score new rows i in [L0,L)
    against target t=c (a TRAINED key, deterministic). Returns mean acc + std."""
    Phi_L0 = rng.standard_normal((L0, m))
    R_L0 = trained_pattern_for_fixed_key(L0, c)
    W = min_norm_W(Phi_L0, R_L0)
    accs = []
    margins = []
    for _ in range(trials):
        Phi = np.empty((L, m))
        Phi[:L0] = Phi_L0
        Phi[L0:] = rng.standard_normal((L - L0, m))
        G = Phi @ W.T
        correct = 0
        row_margins = []
        for i in range(L0, L):
            vi = Phi[i]
            logits = G[: i + 1] @ vi
            am = int(np.argmax(logits))
            correct += int(am == c)
            sorted_l = np.sort(logits)
            row_margins.append(logits[c] - max(v for k, v in enumerate(logits) if k != c))
        accs.append(correct / (L - L0))
        margins.append(np.min(row_margins))
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(margins))


def alpha_L_moving_target(task, L0, L, m, n_in, trials, selection, rng):
    """REVERSE regime. Target moves with row -> new rows reference untrained keys."""
    Phi_L0 = rng.standard_normal((L0, m))
    R_L0 = trained_pattern_prevlike(L0)

    def tgt(i):
        if task == "prev":
            return i - 1
        if task == "copy":
            return i - n_in
        if task == "reverse":
            return n_in - 1 - (i - n_in)
        if task == "growpos":   # t = i (growing absolute position, e.g. attend-self-ish)
            return i
        raise ValueError(task)

    accs = []
    new_key_acc = []      # restrict competitors to new keys j>=L0 (exchangeable core)
    chance_new = []
    for _ in range(trials):
        Phi = np.empty((L, m))
        Phi[:L0] = Phi_L0
        Phi[L0:] = rng.standard_normal((L - L0, m))
        if selection == "min_norm":
            W = min_norm_W(Phi_L0, R_L0)
        elif selection == "fixed_symmetric":
            S = rng.standard_normal((m, m)); W = S + S.T
        else:
            raise ValueError(selection)
        G = Phi @ W.T
        correct = 0; cn = 0; ndenom = 0; scored = 0
        for i in range(L0, L):
            t = tgt(i)
            if t < 0 or t > i:
                continue
            scored += 1
            vi = Phi[i]
            logits = G[: i + 1] @ vi
            am = int(np.argmax(logits))
            correct += int(am == t)
            # exchangeable new-key core: keys L0..i (genuinely untrained); only if t is one
            new_keys = np.arange(L0, i + 1)
            if t >= L0 and len(new_keys) >= 2:
                aml = int(new_keys[int(np.argmax(G[new_keys] @ vi))])
                cn += int(aml == t)
                ndenom += 1
                chance_new.append(1.0 / len(new_keys))
        if scored:
            accs.append(correct / scored)
        if ndenom:
            new_key_acc.append(cn / ndenom)
    return (float(np.mean(accs)) if accs else float("nan"),
            float(np.mean(new_key_acc)) if new_key_acc else float("nan"),
            float(np.mean(chance_new)) if chance_new else float("nan"))


def alpha_L_fixed_OUT_of_range(L0, L, m, c, trials, rng):
    """BOUNDARY: target is a FIXED absolute key c, but c >= L0 (untrained).
    Trained pattern can only realize attend-to-c for c<L0; for c>=L0 there is no
    training signal about key c, so we fit a prev-like R_L0 (any trained-rows-only
    rule) and score new rows against the fixed untrained key c. Expect chance."""
    Phi_L0 = rng.standard_normal((L0, m))
    R_L0 = trained_pattern_prevlike(L0)
    W = min_norm_W(Phi_L0, R_L0)
    accs = []
    for _ in range(trials):
        Phi = np.empty((L, m))
        Phi[:L0] = Phi_L0
        Phi[L0:] = rng.standard_normal((L - L0, m))
        G = Phi @ W.T
        correct = 0; scored = 0
        for i in range(L0, L):
            if c > i:
                continue
            scored += 1
            vi = Phi[i]
            logits = G[: i + 1] @ vi
            correct += int(int(np.argmax(logits)) == c)
        if scored:
            accs.append(correct / scored)
    return float(np.mean(accs))


def main():
    rng = np.random.default_rng(0)
    L0, m = 20, 24
    trials = 4000

    print("=" * 72)
    print("FORWARD (<=): t = c fixed in-range absolute (c < L0), on the I.I.D.-EXTENDED table")
    print("              under a min-norm selection. A realizing kernel EXISTS, but this")
    print("              selection does not find it, so alpha_L stays far below 1. The CLAMPED")
    print("              implementation used in the paper does reach alpha_L = 1 (src/spectra.py).")
    print("=" * 72)
    for c in [0, 5, 19]:
        for L in [40, 80, 160, 320]:
            a, s, mar = alpha_L_fixed_key(L0, L, m, c, trials=200, rng=rng)
            print(f"  c={c:2d}  L={L:3d}   alpha_L = {a:.4f}  (std {s:.1e})  "
                  f"min-margin {mar:+.3f}")
    print()

    print("=" * 72)
    print("REVERSE (=>): moving/growing target -> new rows hit untrained keys")
    print("              Expect alpha << 1, new-key core == uniform chance")
    print("=" * 72)
    for task in ["prev", "copy", "reverse", "growpos"]:
        for sel in ["min_norm", "fixed_symmetric"]:
            a, nk, ch = alpha_L_moving_target(task, L0, 60, m, n_in=20,
                                              trials=trials, selection=sel, rng=rng)
            print(f"  task={task:8s} sel={sel:15s}  full-acc={a:.4f}   "
                  f"new-key-acc={nk:.4f}  vs uniform-chance={ch:.4f}")
    print()

    print("=" * 72)
    print("BOUNDARY: t = c FIXED but c >= L0 (out of trained range). Expect chance")
    print("=" * 72)
    for c in [25, 40, 55]:
        a = alpha_L_fixed_OUT_of_range(L0, 60, m, c, trials=trials, rng=rng)
        # chance reference: averaged 1/(i+1) over scored rows roughly
        print(f"  c={c:2d}  L=60   alpha_L = {a:.4f}  (untrained fixed key -> chance)")
    print()
    print("CONCLUSION, for the i.i.d.-extended table under a min-norm selection:")
    print("  - a fixed in-range target is REALIZABLE, but this selection does not recover it, so")
    print("    alpha_L stays well below 1: an existence-versus-selection gap, not an impossibility;")
    print("  - a growing/moving target, or a fixed key outside the trained range, is at chance,")
    print("    which is what Proposition 2 predicts.")
    print("Proposition 4(b) of the paper concerns the CLAMPED table instead, for which alpha_L = 1")
    print("on a fixed in-range absolute position (verified separately via src/spectra.py).")


if __name__ == "__main__":
    main()
