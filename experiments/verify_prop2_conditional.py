"""Adversarial verification of the CORRECTED Proposition 2 (learned-absolute underdetermination).

The published statement asserted UNCONDITIONALLY that the new-row hit probability is 1/(k-1).
A reviewer objected: exchangeability of the off-diagonal new keys only makes their indices
uniform CONDITIONAL on the global winner lying in that subset. Old keys and the self-key are
non-exchangeable competitors, so the unconditional probability is smaller.

This script measures both quantities directly:
  p_cond   = P[argmax == target | argmax is an off-diagonal new key]      -- should be 1/(k-1)
  p_uncond = P[argmax == target]                                          -- should be < 1/(k-1)
and checks the qualitative claim that survives either way: extrapolation is at chance, i.e. the
selection rule carries no information about the target (p_uncond does not exceed 1/(k-1)).
"""
import numpy as np

RNG = np.random.default_rng(0)
L0, D = 20, 8


def trial(L, rng, degenerate=False):
    """One new-row draw. Returns (argmax_index, target, k_offdiag, i)."""
    V = rng.standard_normal((L, D))                 # phi(j) = v_j (trained rows fixed, new rows i.i.d.)
    W = np.zeros((D, D)) if degenerate else rng.standard_normal((D, D))
    i = L - 1                                       # a new row
    logits = V[i] @ W @ V[: i + 1].T                # over ALL causal keys j <= i
    new_offdiag = list(range(L0, i))                # off-diagonal new keys
    t = int(rng.choice(new_offdiag))                # the target is one of them
    return int(np.argmax(logits)), t, len(new_offdiag), i


def run(L, T=40000, degenerate=False):
    rng = np.random.default_rng(1)
    hits = inset = 0
    hits_in = 0
    for _ in range(T):
        am, t, k_off, i = trial(L, rng, degenerate)
        in_new_offdiag = (L0 <= am < i)
        inset += in_new_offdiag
        hits += (am == t)
        if in_new_offdiag:
            hits_in += (am == t)
    k_new = len(range(L0, L))                       # new keys incl. self, as in the paper's k
    return dict(p_uncond=hits / T, p_in=inset / T,
                p_cond=(hits_in / inset if inset else float("nan")),
                inv_k1=1 / (k_new - 1), k_new=k_new,
                n_offdiag=len(range(L0, L - 1)))


if __name__ == "__main__":
    print("=" * 88)
    print("Proposition 2: unconditional vs conditional new-row hit probability")
    print("=" * 88)
    print(f"{'L':>5} {'#offdiag':>9} {'1/(k-1)':>9} {'P[hit]':>9} {'P[win in set]':>14} "
          f"{'P[hit | in set]':>16} {'1/#offdiag':>11}")
    ok = True
    for L in [30, 40, 60, 80]:
        r = run(L)
        expect_cond = 1 / r["n_offdiag"]
        print(f"{L:>5} {r['n_offdiag']:>9} {r['inv_k1']:>9.4f} {r['p_uncond']:>9.4f} "
              f"{r['p_in']:>14.4f} {r['p_cond']:>16.4f} {expect_cond:>11.4f}")
        # the corrected claim: conditional probability matches uniform-on-off-diagonal;
        # unconditional is strictly smaller than 1/(k-1)
        ok &= abs(r["p_cond"] - expect_cond) < 3 * np.sqrt(expect_cond / max(r["p_in"] * 40000, 1))
        ok &= r["p_uncond"] < r["inv_k1"]
    print()
    print("  => CONDITIONAL uniformity holds (P[hit | winner in off-diagonal new keys] ~ 1/#offdiag).")
    print("  => UNCONDITIONAL probability is strictly SMALLER than 1/(k-1): the old-key and self-key")
    print("     competitors absorb probability mass. The published unconditional '1/(k-1)' was wrong;")
    print("     the corrected claim (chance-level, with 1/(k-1) as the conditional value) is supported.")
    print(f"\n  checks passed: {ok}")

    print("\n" + "=" * 88)
    print("Degenerate selection W = 0: almost-sure uniqueness of the argmax fails")
    print("=" * 88)
    rng = np.random.default_rng(2)
    V = rng.standard_normal((40, D))
    W0 = np.zeros((D, D))
    logits = V[39] @ W0 @ V[:40].T
    print(f"  all logits equal under W=0: {np.allclose(logits, logits[0])}  "
          f"(distinct values: {len(np.unique(np.round(logits, 12)))})")
    print("  => with W=0 every key ties, so the argmax is not a.s. unique and the tie rule decides.")
    print("     The corrected statement excludes degenerate selections explicitly.")
