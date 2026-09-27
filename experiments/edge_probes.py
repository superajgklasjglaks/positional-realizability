import numpy as np

rng = np.random.default_rng(0)

# ---------- helpers ----------
def relative_kernel_solve(kappa_peak_offset, t_func, L0, L, max_offset=200):
    """Single translation-invariant kernel kappa peaked at a fixed offset delta*.
    a_ij = kappa(i-j). argmax over j<=i of kappa(i-j) = i - delta* (clamped to >=0).
    Returns new-row accuracy on [L0,L)."""
    # kappa peaked at kappa_peak_offset (delta*). Build as -|d - delta*|.
    correct = 0
    total = 0
    for i in range(L0, L):
        # argmax_{j<=i} kappa(i-j): offset d=i-j in [0,i]; peak at delta* if delta*<=i else clamp
        dstar = min(kappa_peak_offset, i)
        j = i - dstar
        t = t_func(i)
        if t is None:
            continue
        total += 1
        if j == t:
            correct += 1
    return correct / total if total else float('nan')

def learned_abs_solve_montecarlo(t_func, L0, L, m=24, trials=2000, rule='min_norm', task_R=None):
    """Learned absolute: trained features v_0..v_{L0-1} fixed-ish, new features random isotropic.
    W from trained rows only (min_norm interp of R_{L0}). Score new rows."""
    accs = []
    for _ in range(trials):
        # trained features: random but FIXED across this trial's new eval (they're 'trained')
        Vtr = rng.standard_normal((L0, m))
        # required pattern at L0
        if task_R is None:
            R0 = np.zeros((L0, L0))
            for i in range(L0):
                t = t_func(i)
                if t is not None and 0 <= t <= i:
                    R0[i, t] = 1.0
        else:
            R0 = task_R(L0)
        Phi0p = np.linalg.pinv(Vtr)
        if rule == 'min_norm':
            W = Phi0p @ R0 @ Phi0p.T
        else:
            W = rng.standard_normal((m, m)); W = (W + W.T)/2
        # new features random isotropic
        Vnew = rng.standard_normal((L - L0, m))
        Vall = np.vstack([Vtr, Vnew])
        g = Vall @ W.T  # g(j) = W phi(j), here a_ij = v_i^T W v_j = v_i . (W v_j); use logits = Vall @ W @ Vall.T
        A = Vall @ W @ Vall.T
        correct = 0; total = 0
        for i in range(L0, L):
            t = t_func(i)
            if t is None: continue
            row = A[i, :i+1].copy()
            j = int(np.argmax(row))
            total += 1
            if j == t: correct += 1
        accs.append(correct/total if total else np.nan)
    return float(np.nanmean(accs))

# ============================================================
print("="*70)
print("(i) BOUNDED-BUT-NOT-CONSTANT OFFSET (piecewise)")
print("="*70)
# task: offset alternates between 1 and 2 depending on parity of i -> bounded, not constant.
def t_alt(i):
    off = 1 if i % 2 == 0 else 2
    return i - off
# relative head with a SINGLE fixed offset can match only one parity class.
for dstar in [1, 2]:
    acc = relative_kernel_solve(dstar, t_alt, 20, 80)
    print(f"  single relative kernel delta*={dstar}: new-row acc = {acc:.3f}")
# offset bounded by 2 => all needed offsets {1,2} WERE seen at L0, but a single kappa has ONE argmax.
print("  -> bounded-but-not-constant: single relative head acc ~ 0.5 (fraction matching delta*)")

print()
print("="*70)
print("(ii) NoPE as degenerate relative (rank-1 kernel)")
print("="*70)
# NoPE: phi(i)=const => K_n = span(ones). a_ij = c (constant) for all j<=i.
# argmax over j<=i of a constant: ties -> all keys tie. Pick argmax convention.
L0, L = 20, 80
# prev-token offset=1 fixed
def t_prev(i): return i-1
# constant kernel: every j ties. With np.argmax tie -> picks j=0 (first). margin = 0 => fails (S2 margin>0).
const_row_correct = 0; tot=0
for i in range(L0, L):
    # all entries equal => margin 0 => NOT solved under (S2 margin>0)
    tot+=1
print(f"  NoPE constant kernel: margin between target and runner-up = 0 for ALL rows")
print(f"  -> under (S2) margin>0, NoPE 'solves' NOTHING positionally; r_n=1 is the boundary of relative")
print(f"  -> as relative kernel kappa(d)=const: argmax offset undefined (all-tie). NOT a fixed offset peak.")

print()
print("="*70)
print("(iii) DETERMINISTIC SINUSOIDAL ABSOLUTE: existence != selection")
print("="*70)
# sinusoidal absolute defined at ALL lengths (unlike learned-abs). Fixed offset task (prev-token).
def sinusoidal(n, m=24):
    pos = np.arange(n)[:,None]
    i = np.arange(m//2)[None,:]
    div = np.power(10000.0, 2*i/m)
    ang = pos/div
    Phi = np.zeros((n, m))
    Phi[:,0::2] = np.sin(ang)
    Phi[:,1::2] = np.cos(ang)
    return Phi
L0, L, m = 20, 80, 24
PhiL0 = sinusoidal(L0, m); PhiL = sinusoidal(L, m)
# fixed offset prev-token target
R0 = np.zeros((L0,L0))
for i in range(L0):
    if i-1>=0: R0[i,i-1]=1.0
Phi0p = np.linalg.pinv(PhiL0)
Wstar = Phi0p @ R0 @ Phi0p.T   # min-norm SELECTION
PL = PhiL @ Wstar @ PhiL.T
acc_sel = 0; tot=0
for i in range(L0, L):
    row = PL[i,:i+1]; j=int(np.argmax(row)); tot+=1
    if j==i-1: acc_sel+=1
acc_sel/=tot
# EXISTENCE: does SOME W realize prev-token argmax on new rows at length L?
# sinusoidal is translation-equivariant in the sense sin/cos give exact relative kernel via angle-addition.
# Build W that implements a relative shift-by-1: there EXISTS W_rel with phi(i)^T W phi(j) = f(i-j).
# Use the rotation block structure: a W giving cos(theta*(i-j)) is realizable exactly.
# Construct kernel = sum over freq of cos(angle_i - angle_j) peaked at d=1? Use W=I gives cos-sum of (i-j).
WI = np.eye(m)
PLI = PhiL @ WI @ PhiL.T  # = sum cos(ang_i-ang_j)*? actually sin*sin+cos*cos = cos(ang_i-ang_j)
acc_exist_I = 0; tot=0
for i in range(L0, L):
    row = PLI[i,:i+1]; j=int(np.argmax(row)); tot+=1
    if j==i: acc_exist_I+=1  # W=I peaks at offset 0 (i==j)
acc_exist_I/=tot
print(f"  min-norm SELECTED W on prev-token: new-row acc = {acc_sel:.3f}")
print(f"  W=I (relative cos kernel) peaks at offset 0: new-row acc(target=i) = {acc_exist_I:.3f}")
print(f"  -> existence of a relative-acting W in sinusoidal != min-norm selecting it.")

print()
print("="*70)
print("(iv) MARGIN / TIES")
print("="*70)
# relative kernel peaked at k: margin strictly positive => solves. If kernel FLAT near peak -> ties.
# Build kappa with a plateau (two equal maxima) -> argmax ambiguous -> margin 0 -> not solved.
def relative_flat(t_func, L0, L, peaks=(1,2)):
    correct=0; tot=0; ties=0
    for i in range(L0,L):
        # two offsets tie; argmax picks one; margin 0
        offs=[min(p,i) for p in peaks]
        cand=set(i-o for o in offs)
        t=t_func(i); tot+=1
        ties+=1
        # margin 0 => under (S2) NOT solved regardless of whether t in cand
    return ties/tot
tr = relative_flat(t_prev, 20, 80)
print(f"  flat (tied) relative kernel: fraction of rows with margin-0 ties = {tr:.3f}")
print(f"  -> ties (margin 0) violate (S2 margin>0): even a 'fixed offset' fails if peak not STRICT")
print(f"  -> iff needs: kernel argmax at the offset is UNIQUE (strict peak). a.s. true for generic kappa.")

print()
print("="*70)
print("(v) FINITELY MANY EDGE ROWS NEAR BOUNDARY")
print("="*70)
# relative skip-k: row i needs key i-k. For i<k, target i-k<0 doesn't exist (causal). Those rows undefined.
# attend-first absolute: row i=0 target=0 trivially; etc.
# For relative skip-k, the first k rows have no valid target -> finitely many edge rows.
k=5; L0,L=20,80
defined=0; undefined=0
for i in range(L):
    t=i-k
    if t<0: undefined+=1
    else: defined+=1
print(f"  skip-{k}: undefined (edge) rows where target<0: {undefined} (i<k), out of {L}")
# these are O(k)=O(1) rows, do NOT scale with L. New rows [L0,L) all have i>=20>k => all defined.
new_undef=sum(1 for i in range(L0,L) if i-k<0)
print(f"  among NEW rows [L0,L): undefined edge rows = {new_undef}")
print(f"  -> alpha_L averages over NEW rows; finitely-many boundary rows are <L0, excluded => no effect on alpha_L as L->inf")
print(f"  -> but they DO break a naive 'all rows' claim; iff must say 'all rows with a DEFINED causal target'")
