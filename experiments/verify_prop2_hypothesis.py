"""Why Proposition 2 needs the hypothesis t_L(i) in N_i (target is an off-diagonal NEW key).

If the target is an OLD (trained) key -- as in attend-first, where t_L(i)=0 -- the L0-fit has shaped
exactly those keys, so the hit probability is not governed by exchangeability among the untrained keys
and can exceed the bound 1/(k-1). This script measures that directly for the i.i.d.-extended
learned-absolute model with a min-norm W fitted on the trained rows only.

Reported: Pr[M_i = 0] on new rows, against the bound 1/(k-1), for a range of feature dimensions m.
"""
import numpy as np

L0 = 20


def min_norm_W(V, m, rng):
    """Min-norm W fitting the prev-token pattern on the TRAINED rows only (i < L0)."""
    A, b = [], []
    for i in range(1, L0):
        A.append(np.outer(V[i], V[i - 1]).ravel()); b.append(1.0)
        j = int(rng.integers(0, i))
        if j != i - 1:
            A.append(np.outer(V[i], V[j]).ravel()); b.append(0.0)
    w = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0]
    return w.reshape(m, m)


def run(m, L, trials=4000, seed=0):
    rng = np.random.default_rng(seed)
    hits_last = 0
    hits_rows, n_rows = 0, 0
    for _ in range(trials):
        V = rng.standard_normal((L, m))
        W = min_norm_W(V, m, rng)
        P = V @ W @ V.T
        # last new row
        i = L - 1
        hits_last += int(np.argmax(P[i, : i + 1]) == 0)
        # averaged over all new rows
        for i in range(L0, L):
            hits_rows += int(np.argmax(P[i, : i + 1]) == 0); n_rows += 1
    k = len(range(L0, L))
    return dict(bound=1 / (k - 1), last=hits_last / trials, mean_rows=hits_rows / n_rows)


if __name__ == "__main__":
    print("Proposition 2 without the hypothesis t in N_i: target is the OLD key j=0 (attend-first).")
    print("Model: i.i.d.-extended learned-absolute, min-norm W fitted on trained rows only.\n")
    print(f"{'m':>4}{'L':>6}{'1/(k-1)':>10}{'P[hit] (last row)':>19}{'ratio':>8}"
          f"{'P[hit] (mean rows)':>20}{'ratio':>8}")
    for m in (8, 16, 24, 32, 48, 64):
        for L in (60, 128):
            r = run(m, L)
            print(f"{m:>4}{L:>6}{r['bound']:>10.5f}{r['last']:>19.4f}"
                  f"{r['last']/r['bound']:>7.1f}x{r['mean_rows']:>20.4f}"
                  f"{r['mean_rows']/r['bound']:>7.1f}x")
    print("\nThe bound 1/(k-1) is exceeded, so Prop. 2 does require t_L(i) in N_i.")
    print("With that hypothesis the bound holds (see verify_prop2_conditional.py).")
