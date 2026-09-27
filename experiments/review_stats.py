"""Headline statistics reported in the paper: cell-level correlation + bootstrap CI (M2), dropped-cell
correlation + threshold sweep (M3), alpha>0-only correlation (m3), stronger 2-feature baseline (m2)."""
import os, sys, json
import numpy as np
from scipy.stats import spearmanr

rows = json.load(open(os.path.join(os.path.dirname(__file__), "..", "results", "study_rows.json")))
rng = np.random.RandomState(0)
T = "1L1H"

def cells_of(subset):
    d = {}
    for r in subset:
        d.setdefault((r["pe"], r["task"]), []).append(r)
    return d

tier = [r for r in rows if r["tier"] == T]
learn = [r for r in tier if r["acc_train"] > 0.8]

# ---- M2: cell-level correlations + bootstrap CI over cells ----
def cell_means(subset):
    cm = cells_of(subset)
    pts = [(np.mean([r["alpha"] for r in rs]), np.mean([r["acc_lg"] for r in rs])) for rs in cm.values()]
    return np.array(pts), list(cm.keys())

pts, keys = cell_means(learn)
rho_cell = spearmanr(pts[:, 0], pts[:, 1])[0]
# bootstrap over cells
boot = []
for _ in range(5000):
    idx = rng.randint(0, len(pts), len(pts))
    s = pts[idx]
    if np.std(s[:, 0]) > 0:
        boot.append(spearmanr(s[:, 0], s[:, 1])[0])
ci = np.percentile(boot, [2.5, 97.5])

# (pe,task,L) level
ptl = np.array([(r["alpha"], r["acc_lg"]) for r in learn])
# but average over seeds at (pe,task,L)
agg = {}
for r in learn:
    agg.setdefault((r["pe"], r["task"], r["L"]), []).append(r)
ptl2 = np.array([(np.mean([x["alpha"] for x in v]), np.mean([x["acc_lg"] for x in v])) for v in agg.values()])
rho_ptl = spearmanr(ptl2[:, 0], ptl2[:, 1])[0]

print(f"[M2] cell-level (n={len(pts)} (pe,task)): Spearman={rho_cell:.3f}  bootstrap 95% CI [{ci[0]:.2f},{ci[1]:.2f}]")
print(f"[M2] (pe,task,L)-level (n={len(ptl2)}): Spearman={rho_ptl:.3f}")

# ---- M3: dropped cells + threshold sweep ----
dropped = [r for r in tier if r["acc_train"] <= 0.8]
pd, _ = cell_means(dropped)
rho_drop = spearmanr(pd[:, 0], pd[:, 1])[0] if np.std(pd[:, 0]) > 0 else float("nan")
print(f"\n[M3] dropped cells (n={len(pd)} (pe,task), all NoPE/ALiBi + copy/reverse): "
      f"Spearman={rho_drop:.3f}  (alpha std={np.std(pd[:,0]):.2f})")
print("[M3] threshold sweep (learnable cutoff -> cell-level Spearman):")
for thr in [0.5, 0.7, 0.9]:
    sub = [r for r in tier if r["acc_train"] > thr]
    p, k = cell_means(sub)
    rho = spearmanr(p[:, 0], p[:, 1])[0] if np.std(p[:, 0]) > 0 else float("nan")
    print(f"    train>{thr}: rho={rho:.3f}  (n_cells={len(p)})")

# ---- m3: alpha>0-only correlation ----
pos = [r for r in learn if r["alpha"] > 1e-6]
ppos = np.array([(r["alpha"], r["acc_lg"]) for r in pos])
print(f"\n[m3] alpha>0 only (n={len(pos)} rows, drops trivial 0-separation): "
      f"Spearman={spearmanr(ppos[:,0],ppos[:,1])[0]:.3f}")

# ---- m2: stronger 2-feature baseline (relative-PE x offset-type) ----
REL = {"rope": 1, "alibi": 1, "nope": 0, "sinusoidal": 0, "learned-abs": 0}
OFFSET = {"prev": "bounded", "skip2": "bounded", "skip3": "bounded", "skip5": "bounded",
          "copy": "growing", "reverse": "growing", "first": "absolute"}
# baseline B2 predicts LG iff (relative PE & bounded) OR (absolute PE & absolute task)
def b2(pe, task):
    rel = REL[pe]; off = OFFSET[task]
    return 1.0 if (rel and off == "bounded") or ((not rel) and off == "absolute") else 0.0
b2v = np.array([b2(r["pe"], r["task"]) for r in learn])
lgv = np.array([r["acc_lg"] for r in learn])
av = np.array([r["alpha"] for r in learn])
print(f"\n[m2] strong 2-feature baseline (PE-type x offset-type) vs alpha_L, on {len(learn)} learnable rows:")
print(f"    baseline B2 : Spearman={spearmanr(b2v,lgv)[0]:.3f}  precision={np.mean(lgv[b2v>=0.5]>=0.7):.2f}")
print(f"    alpha_L     : Spearman={spearmanr(av,lgv)[0]:.3f}  precision={np.mean(lgv[av>=0.5]>=0.7):.2f}")
# where B2 fails but alpha is right: the intermediate sinusoidal + absolute-PE-absolute-task graded cases
print("    B2 errors that alpha_L gets right (1L1H @ L=60):")
for r in tier:
    if r["L"] == 60 and r["acc_train"] > 0.8:
        pred_b2 = b2(r["pe"], r["task"]); correct_b2 = (pred_b2 >= 0.5) == (r["acc_lg"] >= 0.5)
        correct_a = (r["alpha"] >= 0.5) == (r["acc_lg"] >= 0.5)
        if not correct_b2 and correct_a:
            print(f"      {r['pe']:11s} {r['task']:6s}  B2={pred_b2:.0f} alpha={r['alpha']:.2f} LG={r['acc_lg']:.2f}")

# ---- m5: distinct-cell support for sufficiency precision ----
hi = [r for r in learn if r["alpha"] >= 0.5]
hi_cells = sorted({(r["pe"], r["task"]) for r in hi})
print(f"\n[m5] sufficiency precision support: {len(hi)} rows over {len(hi_cells)} distinct cells: {hi_cells}")
