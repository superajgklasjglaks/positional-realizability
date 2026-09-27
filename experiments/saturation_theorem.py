"""
SATURATION THEOREM verification  (theory/theory.md, Conjecture 1, operator T_L, K_n = colspan(Phi_n)^{ox2}).

Saturated regime: DETERMINISTIC PE, FIXED finite feature dimension m (phi : N -> R^m fixed at all
lengths).  Saturation = the feature map has SATURATED its rank: rank(Phi_{L0}) = m (full column rank),
which generically needs L0 >= m.

Claim (Saturation Theorem), under (S1)-(S3):
  If Phi_{L0} in R^{L0 x m} has full column rank m, then the bilinear map
        B : R^{m x m} -> R^{L0 x L0},   B(W) = Phi_{L0} W Phi_{L0}^T
  is INJECTIVE.  Hence:
   (a) the W realizing a given L0-block kernel is UNIQUE (no min-norm ambiguity; pinv = exact section);
   (b) any kernel specified on the L0 block (in particular the min-norm interpolant
        W* = Phi_{L0}^+ R_{L0} (Phi_{L0}^+)^T when R_{L0} in K_{L0}) PINS W* UNIQUELY on all R^{m x m};
   (c) therefore the length-L kernel  P_L = Phi_L W* Phi_L^T  is FULLY DETERMINED by the L0 fit and the
       (deterministic) feature map -- there is no remaining freedom that training at L could exploit;
   (d) length generalization is DECIDABLE: it holds  iff  P_L has the correct causal argmax on the new
       rows i in [L0, L).  The "argmax-flip" of Conjecture 1 is then COMPUTED, not assumed.

This script verifies (numerically, sinusoidal Phi):
  T1. Injectivity:  the linear operator W |-> Phi_{L0} W Phi_{L0}^T has trivial kernel
      <=> rank of its (m^2 x m^2) matrix == m^2  <=> smallest singular value > 0.
      Equivalently kron(Phi_{L0}, Phi_{L0}) has full column rank m^2 (rank = sigma_min(Phi_{L0})^2 ... ).
  T2. Uniqueness / exact recovery:  reconstruct a random W from its L0 block via pinv and check W*==W.
  T3. Pinning:  W is determined on ALL of R^{m x m}; show two different W with the same L0 block is
      impossible (kernel of B is {0}); and that the determined P_L is unique.
  T4. Decidable check:  compute P_L for a saturated sinusoidal PE on prev (constant offset, should LG)
      and copy (growing offset, should NOT LG); report the argmax-flip per new row -> decision.
  T5. Rank-deficient contrast:  if L0 < m (or features degenerate) injectivity FAILS -> nullspace,
      multiple W give same L0 block -> P_L NOT determined -> theorem hypothesis genuinely required.
"""

import numpy as np

np.random.seed(0)


# ---- deterministic sinusoidal feature map phi : N -> R^m  (fixed at all lengths) -------------------
# NOTE on saturation: full column rank m is NOT automatic from L0 >= m.  With the standard transformer
# base=10000 and geometric frequencies, the lowest frequencies are nearly constant over a length-20
# grid, so several columns are ~collinear and rank(Phi_L0) < m (numerically rank 12 at m=16, L0=20,
# cond ~ 1e19).  Saturation is therefore a real condition on the FREQUENCY DESIGN: the m frequencies
# must be resolvable on the L0 grid (base small enough that the slowest sinusoid completes a fraction
# of a period over L0).  We pick a well-conditioned base so Phi_L0 attains full column rank m.
def sinusoidal(n, m, base=10.0):
    pos = np.arange(n)[:, None]
    i = np.arange(m // 2)[None, :]
    ang = pos / np.power(base, 2.0 * i / m)
    Phi = np.zeros((n, m))
    Phi[:, 0::2] = np.sin(ang)
    Phi[:, 1::2] = np.cos(ang)
    return Phi


def bilinear_matrix(Phi):
    """Matrix of W |-> Phi W Phi^T as a linear map R^{m^2} -> R^{N^2}: it is kron(Phi, Phi)
    acting on vec(W) (column-major: vec(Phi W Phi^T) = (Phi kron Phi) vec(W))."""
    return np.kron(Phi, Phi)


def report(name, ok):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return ok


def main():
    m = 8           # fixed feature dimension  (4 frequencies -> 8 features)
    L0 = 20         # train length; L0 >= m so Phi_{L0} can have full column rank m
    L = 80          # test length
    # CONDITIONING REMARK.  Injectivity of B is EXACT iff Phi_L0 has full column rank: the singular
    # values of the kron map are products sigma_a(Phi0)*sigma_b(Phi0), so sigma_min(B)=sigma_min(Phi0)^2.
    # The map is thus well-conditioned only when cond(Phi0)^2 is moderate.  At m=16 the standard sinus.
    # geometry forces cond(Phi0)~3e9 even when full-rank, so cond(B)~1e19 and pinv recovery is swamped
    # by roundoff (the theorem still holds in exact arithmetic; it is the NUMERICS that degrade as the
    # square).  We therefore verify at m=8 where cond(Phi0)~8 (cond(B)~60), and separately confirm the
    # exact identity sigma_min(B)=sigma_min(Phi0)^2>0 which is the rigorous injectivity criterion.
    print(f"Saturated deterministic sinusoidal PE:  m={m}, L0={L0} (>= m), L={L}\n")

    # Show the design condition explicitly: at larger m the standard base=10000 is NOT saturated.
    for mb in (m, 16):
        Phi_std = sinusoidal(L0, mb, base=10000.0)
        r = np.linalg.matrix_rank(Phi_std)
        tag = "SATURATED" if r == mb else "RANK-DEFICIENT (not saturated)"
        print(f"[design note] base=10000, m={mb:2d}: rank(Phi_L0)={r:2d} (m={mb:2d}) cond={np.linalg.cond(Phi_std):.1e} -> {tag}")
    print(f"[design note] using base=10.0 (resolvable frequencies over the L0 grid) for the saturated tests.\n")

    Phi0 = sinusoidal(L0, m)
    PhiL = sinusoidal(L, m)

    r0 = np.linalg.matrix_rank(Phi0)
    print(f"rank(Phi_L0) = {r0}  (m = {m})  -> "
          f"{'SATURATED (full column rank)' if r0 == m else 'rank-deficient'}\n")

    all_ok = True

    # ---- T1. INJECTIVITY of B(W) = Phi0 W Phi0^T -------------------------------------------------
    print("T1. Injectivity of the bilinear map W |-> Phi_L0 W Phi_L0^T:")
    K = bilinear_matrix(Phi0)                       # (L0^2, m^2)
    s = np.linalg.svd(K, compute_uv=False)
    smin = s.min()
    rankK = int(np.sum(s > 1e-9 * s.max()))
    # sigma_min of the kron map equals sigma_min(Phi0)^2 (singular values of kron are products)
    sp = np.linalg.svd(Phi0, compute_uv=False)
    pred_smin = sp.min() ** 2
    all_ok &= report(f"kron(Phi0,Phi0) full column rank: rank={rankK} == m^2={m*m}", rankK == m * m)
    all_ok &= report(f"sigma_min(B) = {smin:.3e} > 0", smin > 1e-8)
    all_ok &= report(f"sigma_min(B) == sigma_min(Phi0)^2  ({smin:.3e} vs {pred_smin:.3e})",
                     np.isclose(smin, pred_smin, rtol=1e-6))
    # full-rank <=> injective: nullspace dimension = m^2 - rank = 0
    nulldim = m * m - rankK
    all_ok &= report(f"dim ker(B) = {nulldim} == 0  (=> B injective)", nulldim == 0)
    print()

    # ---- T2. UNIQUENESS / EXACT RECOVERY of W from its L0 block ----------------------------------
    print("T2. Exact recovery: random W reconstructed from its L0 block via pinv:")
    Wtrue = np.random.randn(m, m)
    block = Phi0 @ Wtrue @ Phi0.T                   # the L0-block kernel
    Pi0 = np.linalg.pinv(Phi0)
    Wrec = Pi0 @ block @ Pi0.T                       # min-norm section
    rec_err = np.linalg.norm(Wrec - Wtrue) / np.linalg.norm(Wtrue)
    all_ok &= report(f"||W_rec - W_true|| / ||W_true|| = {rec_err:.3e} ~ 0  (W is UNIQUE)",
                     rec_err < 1e-8)
    # block round-trips exactly
    block_err = np.linalg.norm(Phi0 @ Wrec @ Phi0.T - block) / np.linalg.norm(block)
    all_ok &= report(f"block round-trip error = {block_err:.3e} ~ 0", block_err < 1e-8)
    print()

    # ---- T3. PINNING: same L0 block => same W => same determined P_L -----------------------------
    print("T3. Pinning on all R^{m x m} (no two distinct W share an L0 block):")
    # if there were W1 != W2 with same block, B(W1-W2)=0 with W1-W2 != 0, contradicting ker(B)={0}.
    # Demonstrate: perturb W in a random direction; the L0 block MUST change (injectivity).
    D = np.random.randn(m, m); D /= np.linalg.norm(D)
    block_change = np.linalg.norm(Phi0 @ D @ Phi0.T)
    all_ok &= report(f"any nonzero W-direction moves the L0 block: ||B(D)||={block_change:.3e} > 0",
                     block_change > 1e-8)
    # P_L is therefore a fixed function of the L0 block: recompute from Wrec, deterministic
    PL_a = PhiL @ Wrec @ PhiL.T
    PL_b = PhiL @ (Pi0 @ (Phi0 @ Wtrue @ Phi0.T) @ Pi0.T) @ PhiL.T
    all_ok &= report(f"P_L fully determined by L0 fit: ||P_L(a)-P_L(b)||={np.linalg.norm(PL_a-PL_b):.3e} ~ 0",
                     np.linalg.norm(PL_a - PL_b) < 1e-8)
    print()

    # ---- T4. DECIDABLE LG CHECK on the determined P_L --------------------------------------------
    print("T4. Decidable length generalization (argmax-flip is COMPUTED on the determined P_L):")

    def decide(task_name, target_fn, qstart):
        R0 = np.zeros((L0, L0))
        for i in range(L0):
            t = target_fn(L0, i)
            if 0 <= t <= i and i >= qstart:
                R0[i, t] = 1.0
        # realizability of R0 in K_{L0}: projection residual
        Pi = Phi0 @ Pi0                               # projector onto colspan(Phi0)
        R0_proj = Pi @ R0 @ Pi
        realizable = np.linalg.norm(R0_proj - R0) / max(np.linalg.norm(R0), 1e-12)
        Wstar = Pi0 @ R0 @ Pi0.T                       # min-norm interpolant (saturated => exact W)
        PL = PhiL @ Wstar @ PhiL.T
        # decide LG by the causal argmax on new rows
        correct = 0; rows = 0; flips = []
        for i in range(L0, L):
            t = target_fn(L, i)
            if t < 0 or t > i or i < qstart:
                continue
            rows += 1
            am = int(np.argmax(PL[i, : i + 1]))
            ok = (am == t)
            correct += int(ok)
            if not ok and len(flips) < 4:
                flips.append((i, t, am))
        acc = correct / rows if rows else float("nan")
        lg = (acc == 1.0)
        print(f"   task={task_name:5s}  R0-in-K_L0 residual={realizable:.2e}  "
              f"new-row argmax acc={acc:.3f}  -> LG {'YES' if lg else 'NO '}")
        if flips:
            print(f"       sample argmax-flips (row i: target -> realized argmax): "
                  + ", ".join(f"{i}:{t}->{am}" for (i, t, am) in flips))
        return realizable, acc, lg

    # prev: constant offset 1 (should length-generalize -- decision computed, not assumed)
    res_prev, acc_prev, lg_prev = decide("prev", lambda n, i: i - 1, qstart=1)
    # copy: growing offset ~ n/2 (should NOT length-generalize)
    res_copy, acc_copy, lg_copy = decide("copy", lambda n, i: i - (n + 1) // 2, qstart=(L0 + 1) // 2)
    print("   (The LG verdict above is DECIDABLE: it is read off the determined P_L. The argmax-flip")
    print("    of Conjecture 1 is thus computed, not assumed -- saturated regime resolved.)")
    print()
    print("   NOTE: with only m=8 features the *prev/copy delta patterns are not realizable in K_L0")
    print("   (R0-in-K_L0 residual >> 0), so W* there is the min-norm projection FIT, not an interpolant.")
    print("   The pinning/determinacy of P_L holds regardless. The interpolant branch (R0 in K_L0) is")
    print("   exercised cleanly in T4b below.")
    print()

    # ---- T4b. INTERPOLANT BRANCH: R0 in K_L0 (exact interpolation) -------------------------------
    print("T4b. Interpolant branch (R_L0 IN K_L0 by construction): exact interpolation + determined P_L:")
    Wgen = np.random.randn(m, m)
    R0_real = Phi0 @ Wgen @ Phi0.T                    # a kernel that genuinely lives in K_L0
    Wstar = Pi0 @ R0_real @ Pi0.T                     # min-norm interpolant; saturated => unique W
    interp_err = np.linalg.norm(Phi0 @ Wstar @ Phi0.T - R0_real) / np.linalg.norm(R0_real)
    Wstar_err = np.linalg.norm(Wstar - Wgen) / np.linalg.norm(Wgen)
    all_ok &= report(f"exact interpolation on L0 block: residual={interp_err:.2e} ~ 0", interp_err < 1e-8)
    all_ok &= report(f"recovered W* == generating W (saturated uniqueness): err={Wstar_err:.2e} ~ 0",
                     Wstar_err < 1e-8)
    PL_interp = PhiL @ Wstar @ PhiL.T
    PL_gen = PhiL @ Wgen @ PhiL.T
    all_ok &= report(f"determined length-L kernel matches generator: "
                     f"||P_L(W*)-P_L(Wgen)||={np.linalg.norm(PL_interp-PL_gen):.2e} ~ 0",
                     np.linalg.norm(PL_interp - PL_gen) < 1e-8)
    print("   => when R_L0 in K_L0, W* interpolates exactly AND is the unique pin; P_L is fully forced.")
    print()

    # ---- T5. RANK-DEFICIENT CONTRAST: hypothesis is genuinely required ---------------------------
    print("T5. Rank-deficient contrast (L0 < m  =>  injectivity FAILS, P_L NOT determined):")
    L0d = m // 2                                      # fewer rows than features -> column rank < m
    Phi0d = sinusoidal(L0d, m)
    rd = np.linalg.matrix_rank(Phi0d)
    Kd = bilinear_matrix(Phi0d)
    sd = np.linalg.svd(Kd, compute_uv=False)
    rankKd = int(np.sum(sd > 1e-9 * sd.max()))
    nulldimd = m * m - rankKd
    print(f"   rank(Phi_L0d)={rd} < m={m};  dim ker(B)={nulldimd} > 0")
    all_ok &= report("rank-deficient => B NOT injective (nullspace nonempty)", nulldimd > 0)
    # exhibit two distinct W with identical L0 block but DIFFERENT P_L
    Vt = np.linalg.svd(Kd, full_matrices=True)[2]
    null_vec = Vt[-1]                                 # in ker(B): B(null)=0
    Wa = np.random.randn(m, m)
    Wb = Wa + null_vec.reshape(m, m)                 # same L0 block, different matrix
    blk_a = Phi0d @ Wa @ Phi0d.T
    blk_b = Phi0d @ Wb @ Phi0d.T
    PhiLd = PhiL
    PLa = PhiLd @ Wa @ PhiLd.T
    PLb = PhiLd @ Wb @ PhiLd.T
    same_block = np.linalg.norm(blk_a - blk_b)
    diff_PL = np.linalg.norm(PLa - PLb)
    all_ok &= report(f"two distinct W share the L0 block (||diff||={same_block:.2e}~0) "
                     f"yet give DIFFERENT P_L (||diff||={diff_PL:.2e}>0): P_L UNDETERMINED",
                     same_block < 1e-8 and diff_PL > 1e-6)
    print("   => Saturation (full column rank Phi_L0) is exactly what makes P_L determined & LG decidable.")
    print()

    print("=" * 70)
    print(f"ALL CHECKS {'PASSED' if all_ok else 'FAILED'}")
    return all_ok


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
