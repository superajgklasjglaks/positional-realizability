"""Numerical checks for the hypotheses of Theorem 2 and Propositions 2 and 4.

Each block exhibits the concrete example that shows why a hypothesis in the paper is needed.
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

L0 = 20
print("=" * 78)
print("CHECK 1: shallow (clipped) rows never leave [L0,L), so 'alpha_L=1 for all large L'")
print("  can be unreachable even under EXACT constancy on every deep row.")
print("=" * 78)
# kappa peaked at 30 -> c(kappa)=30, so rows [20,30) are shallow AND scored, at every L.
kap = -np.abs(np.arange(0, 400) - 30).astype(float)
def realized(kap, i):
    return int(np.argmax(kap[: i + 1]))
def delta(i):                      # exact constancy on deep rows (i>=30); shallow rows demand 1
    return 1 if i < 30 else 30
for L in (40, 100, 1000, 10000, 200000):
    rows = range(L0, L)
    ok = sum(1 for i in rows if realized(kap, i) == delta(i))
    a = ok / len(list(rows))
    print(f"  L={L:>7}: alpha_L={a:.6f}   1-10/(L-20)={1-10/(L-20):.6f}")
print("  -> alpha_L < 1 at EVERY L (the 10 clipped rows always fail), so alpha_L=1 requires")
print("     hitting every SCORED row, not just every deep row.\n")

print("=" * 78)
print("CHECK 2: without a strict unique GLOBAL maximizer, a translation-invariant kernel CAN")
print("  solve a growing offset (copy). This is why the theorem carries that hypothesis.")
print("=" * 78)
# NOTE: a finite 'record' ladder whose largest value is attained once DOES have a strict unique
# global maximizer c, and Theorem 2 then covers it (it solves copy only near L=2c). To exhibit a
# genuine counterexample the values must grow without bound, so no maximizer exists -- as here.
K = np.full(400000, -1.0)
m = 0
while 20 * 4 ** m < len(K):
    K[20 * 4 ** m] = m + 2.0
    m += 1
for L in (40, 160, 640, 2560, 10240, 40960):
    off = (L + 1) // 2
    rows = [i for i in range((L + 1) // 2, L)]
    ok = sum(1 for i in rows if realized(K, i) == off)
    print(f"  L={L:>6}: required offset={off:>6}  alpha_L={ok/len(rows):.4f}")
print("  -> An unbounded ladder kappa(20*4^m)=m+2 has NO maximizer and solves copy at all these")
print("     lengths, so growing-offset impossibility needs the global-maximizer hypothesis.\n")

print("=" * 78)
print("CHECK 3: Prop 2 needs the hypothesis t_L(i) in N_i; here the target is NOT a new key.")
print("  attend-first has t=0, an OLD key.")
print("=" * 78)
rng = np.random.default_rng(0)
for d, L in [(24, 30), (24, 60), (48, 60), (24, 128)]:
    T, hits, inset = 20000, 0, 0
    for _ in range(T):
        V = rng.standard_normal((L, d))
        W = rng.standard_normal((d, d))           # non-degenerate selection
        i = L - 1
        lg = V[i] @ W @ V[: i + 1].T
        am = int(np.argmax(lg))
        hits += (am == 0)                          # attend-first target
        inset += (L0 <= am < i)
    k = len(range(L0, L))
    print(f"  d={d:>3} L={L:>4}: P[M_i=0]={hits/T:.4f}   1/(k-1)={1/(k-1):.5f}"
          f"   ratio={hits/T/(1/(k-1)):.1f}x   P[win in N_i]={inset/T:.3f}")
print("  -> With a RANDOM W the ceiling is not exceeded here; the decisive point is that the")
print("     CONDITIONAL claim is vacuous when t=0 is not in N_i (that conditional is exactly 0),")
print("     and with a FITTED W the ceiling IS exceeded -- see verify_prop2_hypothesis.py,")
print("     which measures up to 5.2x. Either way the statement NEEDS t_L(i) in N_i.\n")

print("=" * 78)
print("CHECK 4: a translation-invariant kernel CAN realize attend-first,")
print("  because the causal window truncates at d=i (no 'peak at offset i' needed).")
print("=" * 78)
mono = np.arange(0, 20000).astype(float)          # kappa(d)=d : monotone increasing
for L in (30, 60, 128, 512):
    rows = range(1, L)
    ok = sum(1 for i in rows if realized(mono, i) == i)   # offset i  <=> key j=0
    print(f"  L={L:>4}: kappa(d)=d  attend-first alpha={ok/len(list(rows)):.4f}")
print("  -> Monotone-increasing relative kernel attends to j=0 at EVERY row and length.")
print("     So 'relative cannot do a fixed absolute position' is a statement about kernels with")
print("     a global maximizer and about what the L0-fit selects, not about existence.\n")

print("=" * 78)
print("CHECK 5: the clamped and i.i.d.-extended learned-absolute tables are different objects")
print("=" * 78)
import spectra as S
print("  clamped implementation (the one in spectra.py / our experiments):")
for L in (30, 60, 128):
    print(f"    L={L:>4}: learned-abs / first  alpha={S.simulate('learned-abs','first',L0,L)['alpha']:.4f}"
          f"    / prev alpha={S.simulate('learned-abs','prev',L0,L)['alpha']:.4f}")
print("  -> clamped: first=1.0, prev=0.0 (an exact tie). The i.i.d.-extended table is a")
print("     DIFFERENT object, governed by Prop 2's chance bound, so Prop 4(b) is stated for")
print("     the clamped object and Prop 2 covers the i.i.d. one.")
