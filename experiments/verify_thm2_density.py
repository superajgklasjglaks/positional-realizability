"""Adversarial verification of the CORRECTED Theorem 2.

The published statement was wrong on two counts (found by peer review):
  (E1) necessity: alpha_L -> 1 does NOT force the required offset to be eventually constant on
       every deep row; a density-zero exceptional set is compatible with alpha_L -> 1
       (counterexample: delta(i)=2 on perfect squares, 1 elsewhere).
  (E2) sufficiency: the proof exhibited kappa(d) = -|d - delta*| without checking that this
       kernel lies in the PE-INDUCED family. For canonical ALiBi (kappa(d)=c0+s*d, s<=0) the
       strict maximizer is always offset 0, so delta*=1 is NOT attainable there.

CORRECTED STATEMENT (what this script tries to falsify):
  Fix a relative family K (translation-invariant kernels kappa with a strict unique maximizer
  c(kappa)). Let D_L = set of deep rows in [L0,L). Then
      some kappa in K achieves alpha_L -> 1
  <=> there exists delta* such that
      (i)  DENSITY: #{i in D_L : delta_L(i) != delta*} / #D_L -> 0, and
      (ii) ATTAINABILITY: delta* = c(kappa) for some kappa in K.
  Separately: alpha_L = 1 for all large L  <=>  delta_L(i) = delta* on EVERY deep row (exact).

This script fuzzes BOTH the kernel family AND the task target function (the previous
verification only fuzzed kernels, which is exactly why (E1) was missed).
"""
import numpy as np

RNG = np.random.default_rng(0)
L0 = 20


# ---------------------------------------------------------------- kernels / families
def realized_offset(kappa, i):
    """Masked argmax offset of a relative kernel at row i (causal window d=0..i)."""
    d = np.arange(0, i + 1)
    return int(np.argmax(kappa[d])) if len(kappa) > i else int(np.argmax(kappa[: i + 1]))


def peak_kernel(delta, D=40):
    """kappa(d) = -|d - delta| : strict unique maximizer at offset delta."""
    d = np.arange(D)
    return -np.abs(d - delta).astype(float)


def alibi_kernel(slope, D=40):
    """Canonical ALiBi: kappa(d) = c0 + s*d with s <= 0 (monotone recency)."""
    d = np.arange(D)
    return -abs(slope) * d


def alpha_L(kappa, target_offset, L, deep_from=0):
    """Fraction of scored rows in [L0,L) whose masked argmax offset equals the required offset."""
    rows = [i for i in range(L0, L) if i >= deep_from]
    if not rows:
        return float("nan")
    ok = sum(1 for i in rows if realized_offset(kappa, i) == target_offset(i))
    return ok / len(rows)


# ---------------------------------------------------------------- task generators (FUZZED)
def make_task(kind, rng, delta_star=1):
    """Return (offset_fn, description, density_constant?, exact_constant?)."""
    if kind == "constant":
        return (lambda i: delta_star), f"constant {delta_star}", True, True
    if kind == "squares":                       # the reviewer counterexample: density 0 exceptions
        return (lambda i: 2 if int(np.sqrt(i)) ** 2 == i else 1), "2 on squares else 1", True, False
    if kind == "powers2":                       # a variant: i = 2^k
        pw = {2 ** k for k in range(1, 20)}
        return (lambda i: 2 if i in pw else 1), "2 on powers of 2 else 1", True, False
    if kind == "sparse_random":                 # genuinely density-0 exception set: cubes
        return (lambda i: 2 if round(i ** (1 / 3)) ** 3 == i else 1), "2 on cubes else 1", True, False
    if kind == "parity":                        # POSITIVE density exceptions -> must NOT survive
        return (lambda i: 1 if i % 2 == 0 else 2), "parity 1/2 (density 1/2)", False, False
    if kind == "growing":                       # copy-like
        return (lambda i: (i + 1) // 2), "growing ~i/2", False, False
    if kind == "drift":                          # slowly growing -> not density-constant
        return (lambda i: 1 + i // 50), "drifts upward", False, False
    if kind == "blocks":                         # alternating long blocks: no density limit
        return (lambda i: 1 if (int(np.log2(i + 2)) % 2 == 0) else 2), "log-blocks 1/2", False, False
    raise ValueError(kind)


def density_nonconst(target_offset, delta_star, L):
    rows = range(L0, L)
    bad = sum(1 for i in rows if target_offset(i) != delta_star)
    return bad / len(list(rows))


# ---------------------------------------------------------------- tests
def test_direction_density():
    """(<=) If offsets are density-1 constant at delta* AND delta* attainable, alpha_L -> 1.
       (=>) If NOT density-1 constant at any attainable delta*, alpha_L does NOT -> 1."""
    print("=" * 78)
    print("TEST 1: density characterization (fuzzing the TASK, kernel = best peak kernel)")
    print("=" * 78)
    kinds = ["constant", "squares", "powers2", "sparse_random", "parity", "growing", "drift", "blocks"]
    ok = True
    # Judge by CONVERGENCE TREND, not a fixed threshold: density-zero exception sets converge
    # slowly (e.g. squares give 1-alpha = O(L^-1/2)), so alpha at any finite L can be < 0.99
    # while still tending to 1. The theorem is asymptotic, so the test must be too.
    LS = (200, 1000, 5000, 50000)
    for kind in kinds:
        f, desc, dens_const, exact_const = make_task(kind, RNG)
        k = peak_kernel(1)                      # peak at the majority offset (delta*=1 here)
        a = [alpha_L(k, f, L) for L in LS]
        dn = density_nonconst(f, 1, LS[-1])
        # tends to 1: either already exactly 1 throughout, or the shortfall shrinks by >2x
        tends_to_1 = (a[-1] == 1.0 and a[0] == 1.0) or (
            a[-1] > a[0] and (1 - a[-1]) < 0.5 * (1 - a[0]) and a[-1] > 0.98)
        pred = "-> 1" if dens_const else "bounded away"
        match = tends_to_1 == dens_const
        ok &= match
        print(f"  {desc:24s} density1const={str(dens_const):5s} "
              f"alpha(0.2k/1k/5k/50k)={a[0]:.3f}/{a[1]:.3f}/{a[2]:.3f}/{a[3]:.3f}  "
              f"bad-dens={dn:.4f} pred={pred:13s} {'OK' if match else '** MISMATCH **'}")
    print(f"\n  => density characterization {'HOLDS on all fuzzed tasks' if ok else 'FAILED'}")
    return ok


def test_exact_vs_density():
    """alpha_L = 1 for all large L requires EXACT constancy, not just density."""
    print("\n" + "=" * 78)
    print("TEST 2: exact alpha_L=1 vs alpha_L->1  (the two-level distinction)")
    print("=" * 78)
    ok = True
    for kind in ["constant", "squares", "powers2"]:
        f, desc, dens_const, exact_const = make_task(kind, RNG)
        k = peak_kernel(1)
        exact_at = [alpha_L(k, f, L) == 1.0 for L in (200, 1000, 5000)]
        # "tends to 1" judged by a shrinking shortfall, not a fixed cutoff at finite L
        a_small, a_big = alpha_L(k, f, 2000), alpha_L(k, f, 200000)
        limit1 = (1 - a_big) < 0.25 * (1 - a_small) or a_big == 1.0
        match = (all(exact_at) == exact_const) and (limit1 == dens_const)
        ok &= match
        print(f"  {desc:24s} exact_const={str(exact_const):5s} alpha==1 at all L: {str(all(exact_at)):5s}"
              f" alpha(2k)={a_small:.4f} alpha(200k)={a_big:.5f} tends_to_1={limit1}"
              f"  {'OK' if match else '** MISMATCH **'}")
    print(f"\n  => two-level distinction {'HOLDS' if ok else 'FAILED'}")
    return ok


def test_family_attainability():
    """(E2) sufficiency needs delta* to be an ATTAINABLE strict maximizer in the PE family."""
    print("\n" + "=" * 78)
    print("TEST 3: family membership (ALiBi cannot peak at any interior offset)")
    print("=" * 78)
    f, desc, _, _ = make_task("constant", RNG, delta_star=1)
    print("  task: constant offset 1 (prev-token) -- density-1 constant, so the OLD statement")
    print("        would wrongly promise a realizing kernel exists in EVERY family.")
    ok = True
    for slope in [0.0, 0.1, 1.0, 5.0]:
        k = alibi_kernel(slope)
        peak = int(np.argmax(k[:10]))
        a = alpha_L(k, f, 1000)
        print(f"    ALiBi slope={slope:4.1f}: strict maximizer at offset {peak}, alpha_1000={a:.3f}")
        ok &= (peak == 0 and a < 0.05)
    a_peak = alpha_L(peak_kernel(1), f, 1000)
    print(f"    peak-kernel family (delta*=1 attainable): alpha_1000={a_peak:.3f}")
    ok &= a_peak > 0.99
    print(f"\n  => attainability is NECESSARY: density-1 constancy alone does NOT imply realizability")
    print(f"     {'CONFIRMED' if ok else 'FAILED'} (ALiBi: offset 1 unattainable -> alpha ~ 0)")
    return ok


def test_corollaries():
    """The paper's headline corollaries must survive the correction."""
    print("\n" + "=" * 78)
    print("TEST 4: headline corollaries survive (prev/skip-k yes; copy/reverse no)")
    print("=" * 78)
    ok = True
    for k_off in [1, 2, 3, 5]:
        f = lambda i, k=k_off: k
        a = alpha_L(peak_kernel(k_off), f, 2000)
        ok &= a > 0.99
        print(f"  fixed offset {k_off}: alpha_2000={a:.3f}  (expect ~1)")
    # copy: growing offset -- NO kernel in the family works
    fcopy = lambda i: (i + 1) // 2
    worst = max(alpha_L(peak_kernel(c), fcopy, 2000) for c in range(0, 40))
    ok &= worst < 0.05
    print(f"  copy (growing): best alpha over 40 candidate peaks = {worst:.3f}  (expect ~0)")
    print(f"\n  => corollaries {'SURVIVE' if ok else 'BROKEN'}")
    return ok


if __name__ == "__main__":
    r = [test_direction_density(), test_exact_vs_density(),
         test_family_attainability(), test_corollaries()]
    print("\n" + "=" * 78)
    print("OVERALL:", "ALL CHECKS PASS" if all(r) else "SOME CHECKS FAILED")
    print("=" * 78)
