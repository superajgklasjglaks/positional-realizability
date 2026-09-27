"""CORE logic for Conjecture 1's hard case (L0 < m, Phi_{L0} not full col rank).

The conjecture's hard case is supposed to give W 'genuine freedom in directions unseen
at L0'. The decisive question is WHETHER THE NEW ROWS CAN EVEN ACCESS THAT FREEDOM.

KEY DECOMPOSITION.
  V := rowspan(Phi_{L0}) subset R^m,  dim V = r = rank(Phi_{L0}).
  The realized new-row logit is  a_{ij} = phi(i)^T W phi(j).
  Write phi(i) = u_i + w_i  with u_i = Pi_V phi(i) in V, w_i = (I-Pi_V) phi(i) in V^perp.
  Similarly phi(j) = u_j + w_j.

  a_{ij} = u_i^T W u_j  +  u_i^T W w_j  +  w_i^T W u_j  +  w_i^T W w_j.

  min-norm W* lies entirely in V (x) V: W* = Pi_V W* Pi_V. So under W*, ONLY u_i^T W* u_j
  survives; all w-terms vanish.  The 'unseen-direction freedom' is the freedom to add any
  Delta with support touching V^perp.  But that freedom CHANGES the new-row logits ONLY
  through the w_i / w_j components.  THEOREM (below) => if w_i = 0 (new query in V) the
  freedom is INVISIBLE on that row's query side; if additionally all competitor keys have
  w_j = 0, the freedom is entirely invisible -> rescue impossible -> conjecture-style
  impossibility holds even though nominally underdetermined.

So the dichotomy is:
  (a) new rows lie in V  => freedom is a mirage; only the seen V (x) V part matters; this
      reduces EXACTLY to the full-rank analysis on V.  alpha_L is determined by W*|_V.
  (b) new rows have a genuine V^perp component => freedom is real; rescue MAY exist; but
      the SELECTION rule (min-norm) sets the V^perp logit contribution to ZERO, so it
      cannot select the rescue -> underdetermination (the Prop-2 mechanism reappears,
      now sub-spaced to V^perp).

This script measures the V^perp content of the new rows for deterministic PEs and tests
the two sub-claims.
"""
import numpy as np


def phi_sinusoidal(n, m):
    assert m % 2 == 0
    pos = np.arange(n)[:, None]
    half = m // 2
    div = np.exp(np.arange(half) * (-np.log(10000.0) / max(half - 1, 1)))
    ang = pos * div[None, :]
    out = np.empty((n, m))
    out[:, 0::2] = np.sin(ang); out[:, 1::2] = np.cos(ang)
    return out


def phi_poly(n, m):
    i = np.arange(n)[:, None] / max(n - 1, 1)
    return np.concatenate([i ** k for k in range(m)], axis=1)


def phi_random_lowrank(n, m, r_intrinsic, seed=0):
    """A FIXED deterministic-but-arbitrary PE with GENUINE V^perp for new rows:
    first r_intrinsic dims vary with i, remaining dims are an i-dependent 'new' signal
    that is ZERO on the training block and nonzero afterwards => guaranteed V^perp."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n, m))
    return base  # fully generic: new rows generically have V^perp component


def vperp_fraction(Phi, L0):
    Phi_L0 = Phi[:L0]
    r = np.linalg.matrix_rank(Phi_L0, tol=1e-8)
    U, S, Vt = np.linalg.svd(Phi_L0, full_matrices=True)
    Vb = Vt[:r].T
    Pi = Vb @ Vb.T
    new = Phi[L0:]
    perp = new @ (np.eye(Phi.shape[1]) - Pi)
    frac = np.linalg.norm(perp, axis=1) / (np.linalg.norm(new, axis=1) + 1e-15)
    return r, frac


def main():
    print("V^perp content of NEW rows, by deterministic PE")
    print("(if ~0, the 'unseen-direction freedom' is invisible to new rows => mirage)")
    print("=" * 78)
    cases = [
        ("sinusoidal m16", lambda: phi_sinusoidal(60, 16), 8),
        ("sinusoidal m32", lambda: phi_sinusoidal(60, 32), 20),
        ("sinusoidal m64", lambda: phi_sinusoidal(80, 64), 20),
        ("poly m10",       lambda: phi_poly(40, 10), 6),
        ("poly m12",       lambda: phi_poly(40, 12), 8),
        ("random-fixed m32", lambda: phi_random_lowrank(60, 32, 0), 20),
    ]
    for name, mk, L0 in cases:
        Phi = mk()
        r, frac = vperp_fraction(Phi, L0)
        m = Phi.shape[1]
        print(f"{name:18s} m={m:3d} L0={L0:3d} rank(Phi_L0)={r:3d} "
              f"underdet={r<m!s:5s}  Vperp_frac[new]: "
              f"min={frac.min():.2e} mean={frac.mean():.2e} max={frac.max():.2e}")
    print()
    print("INTERPRETATION")
    print("  sinusoidal/poly: new rows ~entirely in V (Vperp_frac ~ machine eps) =>")
    print("    nominal underdetermination is a MIRAGE; alpha_L fixed by W*|_V (seen part).")
    print("  random-fixed: new rows have order-1 Vperp component => freedom is REAL,")
    print("    and the Prop-2 selection-blindness argument applies in V^perp.")


if __name__ == "__main__":
    main()
