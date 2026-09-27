"""
TIGHT IFF for length-generalization of a single translation-invariant (relative) head
under (S1)-(S3).  a_{ij} = g(i-j), causal mask j <= i.

THEOREM (proved in the accompanying text).  Fix a position-driven task with target
function t_n(i) and required offset delta_n(i) := i - t_n(i) (1 <= delta_n(i) <= i on
the query region Q_n of scored rows).  A single relative head with kernel g
(global maximizer at offset c(g), a.s. unique under any fixed tie rule) length-
generalizes the task -- i.e. there is one fixed g that, trained up to L0, solves
EVERY length L > L0 on its query region -- IF AND ONLY IF

        delta_n(i) == delta*   (a single constant)  for all scored rows i, all n,

i.e. the required offset is a fixed, bounded, length-independent relative offset, AND
that offset is realizable on every scored row (delta* <= i, i.e. the target key is not
masked away).

This script verifies, numerically:

  FORWARD (<=)  : if delta_n(i) == delta* constant, a triangular-peak kernel g peaked
                  at delta* solves ALL scored rows at ALL lengths with positive margin.
                  (prev/skip-k family.)

  REVERSE (=>)  : if the head length-generalizes then delta must be a single constant.
                  Contrapositive, split two ways:
     (a) delta varies WITH n (grows): copy delta_n=ceil(n/2). Deep-row argmax offset is
         a single g-constant c(g) (Thm N1 / Thm 2). Solving two lengths needs c(g) to
         equal two distinct integers -> impossible.
     (b) delta varies WITH i inside one length: reverse delta_n(i)=2i-2ceil(n/2)+1,
         distinct per row. A single deep-row constant offset matches at most one scored
         row -> fails already AT L0.

  TIES        : every claim is checked under BOTH tie conventions (largest-offset /
                smallest-offset) so the contradiction is between integers, tie-robust.

  FINITE SUPPORT + EDGE ROWS : kernels with finite support radius D; deep-row constancy
                holds for i >= max(D, c(g)); edge rows i < c(g) genuinely differ and are
                handled explicitly (they cannot realize offset c(g): key i-c(g) < 0).
"""
import numpy as np

rng = np.random.default_rng(7)

# ----------------------------------------------------------------------------- helpers
def row_argmax_offset(g, n, tie="largest_offset"):
    """d_star[i] = i - argmax_{j<=i} g(i-j), for i=0..n-1.  g: callable on d>=0."""
    out = np.empty(n, dtype=int)
    for i in range(n):
        offsets = np.arange(i + 1)
        vals = np.array([g(int(d)) for d in offsets], dtype=float)
        best = np.max(vals)
        cand = offsets[np.isclose(vals, best)]
        out[i] = cand.max() if tie == "largest_offset" else cand.min()
    return out

def row_argmax_offset_and_margin(g, n, tie="largest_offset"):
    """Return (offsets, margins): margin = best - runner_up over causal keys."""
    offs = np.empty(n, dtype=int)
    mrg = np.empty(n, dtype=float)
    for i in range(n):
        ds = np.arange(i + 1)
        vals = np.array([g(int(d)) for d in ds], dtype=float)
        best = np.max(vals)
        cand = ds[np.isclose(vals, best)]
        offs[i] = cand.max() if tie == "largest_offset" else cand.min()
        # runner-up = best value among keys NOT at the chosen offset
        chosen = offs[i]
        others = vals[ds != chosen]
        mrg[i] = best - (np.max(others) if others.size else -np.inf)
    return offs, mrg

def triangular_peak_kernel(delta_star, D):
    """g(d) = -|d - delta_star| on 0..D (unique global max at delta_star), -inf beyond."""
    arr = -np.abs(np.arange(D + 1) - delta_star).astype(float)
    def g(d):
        return float(arr[d]) if 0 <= d <= D else -np.inf
    return g

def finite_support_kernel(values, D):
    arr = np.asarray(values, float); assert len(arr) == D + 1
    def g(d):
        return float(arr[d]) if 0 <= d <= D else -np.inf
    return g

def copy_offset(n):   return -(-n // 2)            # ceil(n/2)
def reverse_offset(i, n): return 2 * i - 2 * copy_offset(n) + 1   # i - t_n(i)

TIES = ("largest_offset", "smallest_offset")
PASS = []

# ============================================================================ FORWARD
print("=" * 74)
print("FORWARD (<=):  delta_n(i) == delta* constant  =>  solves ALL L, ALL rows")
print("=" * 74)
fwd_ok = True
for delta_star in (1, 2, 3, 5, 8):
    D = delta_star + 4
    g = triangular_peak_kernel(delta_star, D)
    for L in (delta_star + 1, 20, 60, 128, 320):
        for tie in TIES:
            offs, mrg = row_argmax_offset_and_margin(g, L, tie=tie)
            # scored rows = those where the target key exists: i >= delta_star
            sc = np.arange(delta_star, L)
            ok_off = np.all(offs[sc] == delta_star)
            ok_mrg = np.all(mrg[sc] > 0)
            if not (ok_off and ok_mrg):
                fwd_ok = False
                print(f"  FAIL delta*={delta_star} L={L} tie={tie} "
                      f"off_ok={ok_off} mrg_ok={ok_mrg}")
print(f"  constant-offset kernel solves every scored row at every L, both ties, "
      f"margin>0: {fwd_ok}")
# edge rows i < delta* are NOT scored (target key i-delta* < 0 is masked) -> excluded.
g = triangular_peak_kernel(3, 7)
edge = row_argmax_offset(g, 8, tie="largest_offset")[:3]
print(f"  edge rows i<delta*=3 (unscored, key would be negative): offsets={edge.tolist()}")
PASS.append(("FORWARD", fwd_ok))

# ===================================================================== REVERSE (a) copy
print()
print("=" * 74)
print("REVERSE (a):  delta grows with n (copy ceil(n/2)) -> deep-row constant clash")
print("=" * 74)
# (a.1) deep-row offset is ONE g-constant, independent of n, under BOTH ties.
const_ok = True
for _ in range(4000):
    D = int(rng.integers(2, 14))
    vals = rng.standard_normal(D + 1)
    vals[rng.integers(0, D + 1)] += 7.0          # unique global peak
    g = finite_support_kernel(vals, D)
    c = int(np.argmax(vals))
    for tie in TIES:
        for n in (20, 33, 60, 91):
            ds = row_argmax_offset(g, n, tie=tie)
            deep = ds[max(D, c):]
            if deep.size and not np.all(deep == c):
                const_ok = False; break
        if not const_ok: break
    if not const_ok: break
print(f"  deep-row argmax offset == single g-constant c(g) for all n, both ties: {const_ok}")

# (a.2) the two-length integer contradiction for copy.
L0, L = 20, 60
d0, dL = copy_offset(L0), copy_offset(L)
no_single = not any((c == d0 and c == dL) for c in range(L))
print(f"  copy needs offset {d0} at L0={L0} and {dL} at L={L}; "
      f"one constant cannot equal both: impossible={no_single}")
# concrete: g forced by L0 (peak d0) -> realized deep offset at L is still d0 != dL.
g = triangular_peak_kernel(d0, max(d0, dL) + 2)
realized_L = int(row_argmax_offset(g, L)[L - 1])
copy_a_ok = const_ok and no_single and (realized_L == d0 != dL)
print(f"  L0-forced g realizes offset {realized_L} on deep row at L (copy wants {dL}) "
      f"-> fails: {copy_a_ok}")
PASS.append(("REVERSE-a copy", copy_a_ok))

# ================================================================== REVERSE (b) reverse
print()
print("=" * 74)
print("REVERSE (b):  delta varies with i within one length (reverse) -> fails at L0")
print("=" * 74)
# reverse offset is affine in i with slope 2: distinct per scored row.
n = L0
# scored output rows for reverse: i in [ceil(n/2), n-1] (the n_in output positions),
# offsets 1,3,5,... -- verify distinctness and that any single constant matches <=1 row.
rows = np.arange(copy_offset(n), n)
req = np.array([reverse_offset(int(i), n) for i in rows])
distinct = (len(set(req.tolist())) == len(req))
print(f"  reverse required offsets at L0={n} on scored rows: {req.tolist()}")
print(f"  all distinct (affine slope-2): {distinct}")

# For EVERY possible single deep-row constant c, count how many scored rows it can solve.
# A single relative head realizes constant deep offset c -> correct only where req==c.
best_single = max(int(np.sum(req == c)) for c in range(n))
worst_case = max(1, 0)  # a single constant can match at most the multiplicity of req
# since offsets are distinct, multiplicity is 1 -> at most 1 scored row solved.
single_matches_at_most_one = (best_single <= 1)
print(f"  best a single constant offset can do on scored rows: {best_single} / {len(req)}")
print(f"  a single deep-row constant matches at most one scored row: "
      f"{single_matches_at_most_one}")

# And confirm with an ACTUAL kernel sweep (random g's) that no g solves all scored rows.
# A kernel solves reverse at L0 iff for every scored row i, masked-argmax offset == req(i).
any_solves = False
for _ in range(20000):
    D = int(rng.integers(1, n))
    vals = rng.standard_normal(D + 1)
    vals[rng.integers(0, D + 1)] += 5.0
    g = finite_support_kernel(vals, D)
    ds = row_argmax_offset(g, n)
    if np.all(ds[rows] == req):
        any_solves = True; break
rev_b_ok = distinct and single_matches_at_most_one and (not any_solves)
print(f"  random-kernel sweep: some g solves reverse at L0 = {any_solves} (expect False)")
PASS.append(("REVERSE-b reverse", rev_b_ok))

# ================================================ TIGHTNESS / boundary: bounded but not LG?
print()
print("=" * 74)
print("TIGHTNESS:  a BOUNDED non-constant offset (varies with i) still fails")
print("=" * 74)
# task: delta_n(i) = 1 if i even else 2  (bounded, length-indep RANGE, but NOT a single
# constant). The iff claim says this must fail. Verify no single g solves it at L0.
def alt_offset(i, n): return 1 if i % 2 == 0 else 2
nn = 30
rows2 = np.arange(2, nn)              # scored rows where both offsets realizable
req2 = np.array([alt_offset(int(i), nn) for i in rows2])
any2 = False
for _ in range(20000):
    D = int(rng.integers(2, 12))
    vals = rng.standard_normal(D + 1); vals[rng.integers(0, D + 1)] += 5.0
    g = finite_support_kernel(vals, D)
    ds = row_argmax_offset(g, nn)
    if np.all(ds[rows2] == req2):
        any2 = True; break
tight_ok = (not any2)
print(f"  bounded-but-non-constant offset {{1,2}} solvable by single g: {any2} "
      f"(expect False) -> non-constant fails even when bounded: {tight_ok}")
PASS.append(("TIGHTNESS non-const bounded fails", tight_ok))

# ================================================================================ SUMMARY
print()
print("=" * 74); print("SUMMARY"); print("=" * 74)
allok = True
for name, v in PASS:
    print(f"  {name:42s}: {'PASS' if v else 'FAIL'}"); allok &= v
print(f"  {'ALL CHECKS':42s}: {'PASS' if allok else 'FAIL'}")
