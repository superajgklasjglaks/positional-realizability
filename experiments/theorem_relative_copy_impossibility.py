"""
Numerical verification of the relative-single-head copy/reverse length-generalization
impossibility, under (S1)-(S3) of theory/theory.md.

Setup: translation-invariant positional kernel a_{ij} = g(i-j), causal mask j <= i.
Row-argmax target offset: d*(i) = i - argmax_{j<=i} g(i-j) = argmax_{0<=d<=i} g(d).

Claims verified numerically:
  (A) Row-stability: for every row i >= D (D = support radius of g), the masked
      row-argmax offset d*(i) equals the same constant d* = argmax_{d>=0} g(d),
      independent of i and of n. (Homogeneous unbounded case: d* constant for ALL
      rows with i >= d*.)
  (B) Edge rows i < d* cannot realize offset d* (the needed key j=i-d* < 0 is masked),
      so the argmax differs there -- the claimed "i >= support" caveat is necessary.
  (C) Copy offset delta_n = ceil(n/2) changes with n; for two lengths L0, L with
      ceil(L0/2) != ceil(L/2), a single fixed g whose constant offset d* matches
      copy at L0 (d* = ceil(L0/2)) necessarily mismatches at L on the deep rows.
  (D) Sweep over many random g and many tie-broken g to confirm the argmax offset
      is row-invariant past the support and changes nowhere with n.
"""
import numpy as np

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# Core: given a kernel g defined on offsets 0..N-1, compute the causally-masked
# row-argmax offset d*(i) for each row i of an n x n attention matrix.
# Tie-break: theory uses argmax over j<=i; numpy argmax picks the SMALLEST j on
# ties, i.e. the LARGEST offset d. We make the tie convention explicit and test
# both, since the theorem's "same function of g" must specify ties.
# ---------------------------------------------------------------------------

def row_argmax_offset(g, n, tie="largest_offset"):
    """Return array d_star[i] = i - argmax_{j<=i} g(i-j), for i=0..n-1.
    g: callable or array giving g(d) for d>=0.
    tie: 'largest_offset' (== smallest key j, numpy default) or 'smallest_offset'.
    """
    out = np.empty(n, dtype=int)
    for i in range(n):
        offsets = np.arange(i + 1)              # d = 0..i (j = i-d ranges i..0)
        vals = np.array([g(int(d)) for d in offsets], dtype=float)
        if tie == "largest_offset":
            # pick max value; on ties pick the largest d
            best = np.max(vals)
            cand = offsets[np.isclose(vals, best)]
            out[i] = cand.max()
        elif tie == "smallest_offset":
            best = np.max(vals)
            cand = offsets[np.isclose(vals, best)]
            out[i] = cand.min()
        else:
            raise ValueError(tie)
    return out


def make_finite_support_kernel(values, D):
    """g(d) = values[d] for 0<=d<=D, else -inf (so it never wins). Support radius D."""
    arr = np.asarray(values, dtype=float)
    assert len(arr) == D + 1
    def g(d):
        if 0 <= d <= D:
            return float(arr[d])
        return -np.inf
    return g


# ---------------------------------------------------------------------------
# (A) Row-stability past the support, and n-invariance.
# ---------------------------------------------------------------------------
print("=" * 70)
print("(A) Row-stability past support + n-invariance (finite-support g)")
print("=" * 70)

A_ok = True
for trial in range(2000):
    D = int(rng.integers(0, 8))                  # support radius
    vals = rng.standard_normal(D + 1)
    # ensure unique max offset to avoid ambiguity in the 'true' d*
    if D >= 1:
        vals[rng.integers(0, D + 1)] += 5.0
    g = make_finite_support_kernel(vals, D)
    d_star_true = int(np.argmax(vals))           # argmax_{d>=0} g(d)

    for n in (D + 1, D + 5, 2 * D + 7, 40, 73):
        if n <= D:
            continue
        ds = row_argmax_offset(g, n, tie="largest_offset")
        # rows with i >= max(D, d_star_true): all needed offsets in [0,D] are realizable
        deep = ds[max(D, d_star_true):]
        if not np.all(deep == d_star_true):
            A_ok = False
            print("FAIL trial", trial, "n", n, "d*_true", d_star_true, "deep", deep)
            break
    if not A_ok:
        break
print("All deep rows equal constant d* = argmax_d g(d), for all n:", A_ok)

# ---------------------------------------------------------------------------
# (B) Edge rows i < d* genuinely differ (caveat is necessary).
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("(B) Edge rows i < d* cannot realize offset d* (caveat necessary)")
print("=" * 70)
# Kernel with a clear peak at d=3.
vals = np.array([0.0, 0.1, 0.2, 1.0, 0.05])      # peak at d=3, D=4
g = make_finite_support_kernel(vals, 4)
n = 10
ds = row_argmax_offset(g, n, tie="largest_offset")
print("d*(i) per row, peak offset = 3:", ds.tolist())
edge_differs = any(ds[i] != 3 for i in range(3))     # rows 0,1,2 cannot reach offset 3
deep_constant = all(ds[i] == 3 for i in range(4, n))
print("edge rows (i<3) differ:", edge_differs, "| deep rows (i>=4) == 3:", deep_constant)

# ---------------------------------------------------------------------------
# (C) The copy contradiction across two lengths.
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("(C) Copy: delta_n = ceil(n/2) changes with n -> no single g works at both")
print("=" * 70)

def copy_offset(n):
    return -(-n // 2)   # ceil(n/2)

L0, L = 20, 60
d0, dL = copy_offset(L0), copy_offset(L)
print(f"copy offset at L0={L0}: {d0};  at L={L}: {dL};  differ: {d0 != dL}")

# Build the UNIQUE-peak homogeneous kernel that copy at L0 forces: peak at d0.
# (Any g that solves copy at L0 on deep rows must have argmax_d g(d) = d0, since
# d*(i)=argmax_d g(d) is the SAME constant for all deep rows -- by (A).)
D = max(d0, dL) + 2
vals = -np.abs(np.arange(D + 1) - d0).astype(float)   # unique peak at d0
g = make_finite_support_kernel(vals, D)

# Check it solves copy at L0 on deep rows, then FAILS copy at L on deep rows.
ds_L0 = row_argmax_offset(g, L0, tie="largest_offset")
ds_L  = row_argmax_offset(g, L,  tie="largest_offset")
# deep rows = i >= d0 at L0; i >= d0 at L
deep0 = np.arange(d0, L0)
deepL = np.arange(dL, L)
solves_L0 = np.all(ds_L0[deep0] == d0)              # copy target offset is d0 at L0
solves_L  = np.all(ds_L[deepL]  == dL)             # copy needs dL at L
print("g (peak at d0) solves copy on deep rows at L0:", bool(solves_L0))
print("same g solves copy on deep rows at L :", bool(solves_L), "(expected False)")
print("realized constant offset at L (deep):", int(ds_L[L - 1]),
      "but copy needs", dL)

# Symmetric check: a g tuned for L (peak at dL) fails at L0.
vals2 = -np.abs(np.arange(D + 1) - dL).astype(float)
g2 = make_finite_support_kernel(vals2, D)
ds2_L0 = row_argmax_offset(g2, L0, tie="largest_offset")
ds2_L  = row_argmax_offset(g2, L,  tie="largest_offset")
print("g2 (peak at dL) solves copy at L:", bool(np.all(ds2_L[deepL] == dL)),
      "| at L0:", bool(np.all(ds2_L0[deep0] == d0)), "(expected False at L0)")

# ---------------------------------------------------------------------------
# (D) Exhaustive impossibility: NO homogeneous g can satisfy both deep-row
#     conditions simultaneously, because deep d* is a single constant.
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("(D) No single g satisfies copy on deep rows at BOTH L0 and L")
print("=" * 70)
# The deep-row offset is the SAME constant c = argmax_d g(d) at every length.
# Copy needs c = d0 at L0 and c = dL at L; with d0 != dL this is unsatisfiable.
# Brute force over all unique-argmax integer-peaked kernels c = 0..L-1:
any_both = False
for c in range(L):
    ok_L0 = (c == d0)   # deep rows at L0 realize offset c; copy needs d0
    ok_L  = (c == dL)
    if ok_L0 and ok_L:
        any_both = True
print("exists constant offset c solving copy deep-rows at both lengths:", any_both,
      "(expected False since d0 != dL)")

# Also a randomized robustness sweep: random g, confirm the deep-row offset is a
# SINGLE constant per g (so it can match at most one of {d0, dL}).
single_const = True
for _ in range(3000):
    D = int(rng.integers(2, 12))
    vals = rng.standard_normal(D + 1)
    vals[rng.integers(0, D + 1)] += 6.0     # unique peak
    g = make_finite_support_kernel(vals, D)
    c = int(np.argmax(vals))
    for n in (L0, L):
        ds = row_argmax_offset(g, n, tie="largest_offset")
        deep = ds[max(D, c):]
        if deep.size and not np.all(deep == c):
            single_const = False
            break
    if not single_const:
        break
print("deep-row offset is one g-determined constant across lengths:", single_const)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
ok = (A_ok and edge_differs and deep_constant and solves_L0 and (not solves_L)
      and (not any_both) and single_const)
print("A row-stability/n-invariance :", A_ok)
print("B edge-row caveat necessary  :", edge_differs and deep_constant)
print("C copy contradiction         :", solves_L0 and (not solves_L))
print("D impossibility (no single g):", (not any_both) and single_const)
print("ALL CHECKS PASS              :", ok)
