"""Definitive study: training-free alpha_L vs trained-model length generalization.

Sweeps test length L, multiple seeds, two tiers (1L1H theorem-matched; 2L4H scale), expanded tasks.
Each row = (tier, pe, task, L, seed) with its own alpha_L and empirical LG. Conditions correlation on
in-distribution learnability (train_acc>0.8): asking about LG of a task a model never learned at L0 is
meaningless. Produces results/study_rows.json + results/study_summary.json + figure data.
"""

import os, sys, json, time, random
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import Tiny
from data import make_batch, query_positions, VOCAB
from spectra import simulate

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
L0 = 20
TEST_LENS = [30, 40, 50, 60, 80]
PES = ["learned-abs", "nope", "alibi", "sinusoidal", "rope"]
TASKS = os.environ.get(
    "TASKS_ONLY", "prev,skip2,skip3,skip5,copy,reverse,first").split(",")
_ALL_TIERS = {"1L1H": dict(layers=1, heads=1), "2L4H": dict(layers=2, heads=4),
              "4L4H": dict(layers=4, heads=4)}  # 4L added for the depth trend (H3)
_only = os.environ.get("TIERS_ONLY")
TIERS = {k: v for k, v in _ALL_TIERS.items() if (not _only or k in _only.split(","))}
SEEDS = int(os.environ.get("SEEDS", 5))
STEPS = 2000
BATCH = 64
DMODEL = int(os.environ.get("DMODEL", 64))
STUDY_TAG = os.environ.get("STUDY_TAG", "")  # suffix to avoid clobbering the main study


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def train_one(pe, task, layers, heads, seed):
    set_seed(seed)
    model = Tiny(VOCAB, d=DMODEL, h=heads, layers=layers, pe=pe, max_train_len=L0).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    model.train()
    for _ in range(STEPS):
        n = random.randint(L0 // 2, L0)
        s, tgt = make_batch(task, n, BATCH, DEV)
        loss = torch.nn.functional.cross_entropy(
            model(s).view(-1, VOCAB), tgt.view(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def eval_acc(model, task, n_eval, lo, hi, reps=6):
    model.eval()
    qpos = [i for i in query_positions(task, n_eval) if lo <= i < hi]
    if not qpos:
        return float("nan")
    correct = total = 0
    for _ in range(reps):
        s, tgt = make_batch(task, n_eval, BATCH, DEV)
        pred = model(s).argmax(-1)
        for i in qpos:
            m = tgt[:, i] != -100
            correct += int((pred[:, i][m] == tgt[:, i][m]).sum())
            total += int(m.sum())
    return correct / total if total else float("nan")


def alpha_cached(cache, pe, task, L):
    key = f"{pe}|{task}|{L}"
    if key not in cache:
        cache[key] = simulate(pe, task, L0, L)["alpha"]
    return cache[key]


def main():
    print(f"device={DEV}  L0={L0} test_lens={TEST_LENS}  seeds={SEEDS} steps={STEPS}", flush=True)
    # RESUME: reuse already-saved rows; only run missing (tier,pe,task,seed) combos.
    rows, acache, t0 = [], {}, time.time()
    if os.path.exists(f"results/study_rows{STUDY_TAG}.json"):
        rows = json.load(open(f"results/study_rows{STUDY_TAG}.json"))
        for r in rows:
            acache[f"{r['pe']}|{r['task']}|{r['L']}"] = r["alpha"]
        print(f"[resume] loaded {len(rows)} existing rows", flush=True)
    done = {(r["tier"], r["pe"], r["task"], r["seed"]) for r in rows}
    for tname, tc in TIERS.items():
        for pe in PES:
            for task in TASKS:
                ran = False
                for seed in range(SEEDS):
                    if (tname, pe, task, seed) in done:
                        continue
                    ran = True
                    model = train_one(pe, task, tc["layers"], tc["heads"], seed)
                    acc_train = eval_acc(model, task, L0, L0 // 2, L0)
                    for L in TEST_LENS:
                        rows.append(dict(
                            tier=tname, pe=pe, task=task, L=L, seed=seed,
                            alpha=alpha_cached(acache, pe, task, L),
                            acc_lg=eval_acc(model, task, L, L0, L),
                            acc_train=acc_train))
                    json.dump(rows, open(f"results/study_rows{STUDY_TAG}.json", "w"))
                if ran:
                    last = rows[-1]
                    print(f"{tname} {pe:12s} {task:8s} train={last['acc_train']:.2f}  "
                          f"alpha@60={alpha_cached(acache, pe, task, 60):.2f}  "
                          f"({(time.time()-t0)/60:.1f}m elapsed)", flush=True)

    # ---- analysis ----
    from scipy.stats import spearmanr, pearsonr

    def corr(subset):
        a = np.array([r["alpha"] for r in subset]); lg = np.array([r["acc_lg"] for r in subset])
        if len(a) < 3 or np.std(a) == 0:
            return dict(n=len(a), spearman=float("nan"), pearson=float("nan"), p=float("nan"))
        rho, p = spearmanr(a, lg)
        return dict(n=len(a), spearman=float(rho), pearson=float(pearsonr(a, lg)[0]), p=float(p))

    summary = {}
    for tname in TIERS:
        tier_rows = [r for r in rows if r["tier"] == tname]
        learn = [r for r in tier_rows if r["acc_train"] > 0.8]
        hi = [r for r in learn if r["alpha"] >= 0.5]
        summary[tname] = dict(
            all=corr(tier_rows),
            learnable=corr(learn),
            n_learnable=len(learn),
            sufficiency_precision=(float(np.mean([r["acc_lg"] >= 0.7 for r in hi])) if hi else None),
            n_high_alpha=len(hi))
    summary["minutes"] = round((time.time() - t0) / 60, 1)
    json.dump(summary, open(f"results/study_summary{STUDY_TAG}.json", "w"), indent=2)
    print("\n==== STUDY SUMMARY ====", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
