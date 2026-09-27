"""Probe Conjecture 1's HARD CASE: deterministic PE, L0 < m, Phi_{L0} NOT full column rank.

Here W has genuine freedom in V^perp = (rowspan Phi_{L0})^perp -- directions unseen at L0.
We ask three distinct questions:

  (Q1) EXISTENCE of rescue: does some interpolating W (= W* + free part in V^perp) make
       the new rows argmax-correct?  (cone non-emptiness)
  (Q2) SELECTION: does the min-norm W* (what GD/ridgeless picks) rescue?  (this is what
       alpha_L computes -- the actual predictor)
  (Q3) Does the unseen-direction freedom ADD anything that the seen subspace did not?
       I.e. compare min-norm W* on full R^m vs. the solution that ignores V^perp.
       If new rows live ENTIRELY in V (no V^perp component) the freedom is irrelevant;
       if they have a V^perp component, min-norm sets the logit contribution there to 0.

Deterministic PEs probed: sinusoidal (truncated m frequencies), polynomial features
[1, i, i^2, ...] (controllable degeneracy), and random-but-FIXED low-rank features.
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)


# ---------- deterministic feature maps phi : N -> R^m ----------
def phi_sinusoidal(n, m):
    """Standard transformer sinusoidal features, m even."""
    assert m % 2 == 0
    pos = np.arange(n)[:, None]
    half = m // 2
    div = np.exp(np.arange(half) * (-np.log(10000.0) / max(half - 1, 1)))
    ang = pos * div[None, :]
    out = np.empty((n, m))
    out[:, 0::2] = np.sin(ang)
    out[:, 1::2] = np.cos(ang)
    return out


def phi_poly(n, m):
    """[1, i_scaled, i_scaled^2, ...] -- low effective rank, controllable."""
    i = np.arange(n)[:, None] / max(n - 1, 1)
    return np.concatenate([i ** k for k in range(m)], axis=1)


def target(task, i, n, n_in):
    if task == "prev":
        return i - 1
    if task == "skip2":
        return i - 2
    if task == "first":
        return 0
    if task == "copy":
        return i - n_in
    raise ValueError(task)


def min_norm_W(Phi_L0, R_L0):
    P = np.linalg.pinv(Phi_L0)
    return P @ R_L0 @ P.T


def acc_argmax(W, Phi, rows, task, n, n_in):
    ok = 0
    margins = []
    for i in rows:
        cands = np.arange(i + 1)
        if len(cands) < 2:
            continue
        lg = (Phi[cands] @ W.T) @ Phi[i]
        am = int(cands[int(np.argmax(lg))])
        t = target(task, i, n, n_in)
        if not (0 <= t < len(cands)):
            continue
        ok += int(am == t)
        srt = np.sort(lg)
        margins.append(lg[t] - (srt[-2] if am == t else srt[-1]))
    denom = max(len(margins), 1)
    return ok / denom, float(np.min(margins)) if margins else 0.0


def rescue_exists(Phi_L0, R_L0, Phi, new_rows, task, n, n_in, ridge=0.0):
    """Try to BUILD an interpolating W that is correct on new rows, using V^perp freedom.

    Solve a least-squares fit to the FULL required pattern (old interpolated + new target),
    then check argmax. This is one concrete constructive attempt (not exhaustive)."""
    L = Phi.shape[0]
    # required pattern: rows < L0 -> R_L0 (projected), new rows -> one-hot target
    R_full = np.zeros((L, L))
    R_full[:R_L0.shape[0], :R_L0.shape[1]] = R_L0
    for i in new_rows:
        R_full[i, target(task, i, n, n_in)] = 1.0
    P = np.linalg.pinv(Phi)
    W = P @ R_full @ P.T
    return acc_argmax(W, Phi, new_rows, task, n, n_in)


def run(phi_fn, task, L0, L, m, n_in=None):
    n_in = n_in or L0
    Phi = phi_fn(L, m)
    Phi_L0 = Phi[:L0]
    r = np.linalg.matrix_rank(Phi_L0, tol=1e-9)
    new_rows = list(range(L0, L))

    # trained pattern: realize the task ON the training block (in-distribution targets)
    R_L0 = np.zeros((L0, L0))
    for i in range(L0):
        t = target(task, i, L0, n_in)
        if 0 <= t <= i:
            R_L0[i, t] = 1.0

    # seen subspace V and its complement
    U, S, Vt = np.linalg.svd(Phi_L0, full_matrices=True)
    Vbasis = Vt[:r].T            # m x r, basis of V = rowspan(Phi_L0)
    # component of new-row features in V^perp
    Pi_V = Vbasis @ Vbasis.T
    new_feats = Phi[np.array(new_rows)]
    perp_frac = np.linalg.norm(new_feats @ (np.eye(m) - Pi_V), axis=1) / (
        np.linalg.norm(new_feats, axis=1) + 1e-12)

    # (Q2) min-norm selection -- the actual alpha_L predictor
    Wmn = min_norm_W(Phi_L0, R_L0)
    a_train, _ = acc_argmax(Wmn, Phi_L0, range(L0), task, L0, n_in)
    a_mn, mg_mn = acc_argmax(Wmn, Phi, new_rows, task, L, n_in)

    # (Q1) constructive rescue attempt using full V^perp freedom
    a_resc, mg_resc = rescue_exists(Phi_L0, R_L0, Phi, new_rows, task, L, n_in)

    return dict(
        phi=phi_fn.__name__, task=task, m=m, L0=L0, L=L,
        rank_PhiL0=int(r), full_col_rank=(r == m),
        underdetermined=(r < m),
        train_acc=a_train,
        perp_frac_min=float(perp_frac.min()), perp_frac_mean=float(perp_frac.mean()),
        alpha_L_minnorm=a_mn, margin_minnorm=mg_mn,
        rescue_acc=a_resc, rescue_margin=mg_resc,
    )


def main():
    rng = np.random.default_rng(0)
    configs = []
    # sinusoidal: m can exceed L0 -> underdetermined
    for task in ["prev", "skip2", "first", "copy"]:
        configs.append((phi_sinusoidal, task, 8, 40, 16))   # L0=8 < m=16: UNDER-DET
        configs.append((phi_sinusoidal, task, 20, 60, 32))  # L0=20 < m=32: UNDER-DET
    for task in ["prev", "first", "copy"]:
        configs.append((phi_poly, task, 6, 30, 10))         # poly, L0=6 < m=10

    print(f"{'phi':14s} {'task':6s} {'m':>3s} {'L0':>3s} {'rk':>3s} {'under':>5s} "
          f"{'trn':>5s} {'perpmn':>7s} {'aL_mn':>6s} {'mg_mn':>8s} {'resc':>5s} {'resc_mg':>8s}")
    for cfg in configs:
        r = run(*cfg)
        print(f"{r['phi']:14s} {r['task']:6s} {r['m']:3d} {r['L0']:3d} {r['rank_PhiL0']:3d} "
              f"{str(r['underdetermined']):>5s} {r['train_acc']:5.2f} "
              f"{r['perp_frac_min']:7.3f} {r['alpha_L_minnorm']:6.3f} {r['margin_minnorm']:8.3f} "
              f"{r['rescue_acc']:5.3f} {r['rescue_margin']:8.3f}")


if __name__ == "__main__":
    main()
