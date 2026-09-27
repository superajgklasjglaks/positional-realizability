"""Pilot: does training-free alpha_L predict trained-model length generalization?

Tier-1 (theorem-matched): 1-layer, 1-head models -> alpha_L should be near-exact (Theorem 1 scope).
Tier-2 (heuristic at scale): 2-layer, 4-head -> alpha_L expected to remain a sufficient indicator.
Configurable via env: LAYERS, HEADS, SEEDS, STEPS.
"""

import os, sys, json, time, random
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import Tiny
from data import make_batch, query_positions, VOCAB
from spectra import simulate

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
L0, L = 20, 60
PES = ["learned-abs", "nope", "alibi", "sinusoidal", "rope"]
TASKS = ["prev", "skip3", "copy", "reverse", "first"]
LAYERS = int(os.environ.get("LAYERS", 1))
HEADS = int(os.environ.get("HEADS", 1))
SEEDS = int(os.environ.get("SEEDS", 1))
STEPS = int(os.environ.get("STEPS", 2500))
BATCH = 64
TAG = os.environ.get("TAG", f"L{LAYERS}H{HEADS}")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def train_one(pe, task, seed):
    set_seed(seed)
    model = Tiny(VOCAB, d=64, h=HEADS, layers=LAYERS, pe=pe, max_train_len=L0).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    model.train()
    for step in range(STEPS):
        n = random.randint(L0 // 2, L0)
        s, tgt = make_batch(task, n, BATCH, DEV)
        loss = torch.nn.functional.cross_entropy(
            model(s).view(-1, VOCAB), tgt.view(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def eval_lg(model, task, n_eval, lo, hi, reps=8):
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


def main():
    print(f"[{TAG}] device={DEV}  L0={L0} L={L}  layers={LAYERS} heads={HEADS} "
          f"seeds={SEEDS} steps={STEPS}", flush=True)
    results, t0 = {}, time.time()
    for pe in PES:
        for task in TASKS:
            ts = time.time()
            alpha = simulate(pe, task, L0, L)["alpha"]
            lgs, trs = [], []
            for seed in range(SEEDS):
                model = train_one(pe, task, seed)
                trs.append(eval_lg(model, task, L0, L0 // 2, L0))
                lgs.append(eval_lg(model, task, L, L0, L))
            results[f"{pe}|{task}"] = dict(
                alpha=alpha, acc_lg=float(np.mean(lgs)), acc_lg_std=float(np.std(lgs)),
                acc_train=float(np.mean(trs)))
            print(f"{pe:12s} {task:8s} alpha={alpha:.2f}  LG={np.mean(lgs):.3f}"
                  f"±{np.std(lgs):.2f}  train={np.mean(trs):.3f}  ({time.time()-ts:.0f}s)", flush=True)
            json.dump(results, open(f"results/pilot_{TAG}.json", "w"), indent=2)

    from scipy.stats import spearmanr, pearsonr
    a = np.array([v["alpha"] for v in results.values()])
    lg = np.array([v["acc_lg"] for v in results.values()])
    tr = np.array([v["acc_train"] for v in results.values()])
    rho, p = spearmanr(a, lg)
    mask = tr > 0.8
    rho_l, p_l = (spearmanr(a[mask], lg[mask]) if mask.sum() > 2 else (float("nan"), float("nan")))
    # sufficiency precision: of high-alpha cells, fraction with high LG
    hi = a >= 0.5
    prec = float((lg[hi] >= 0.7).mean()) if hi.sum() else float("nan")
    summary = dict(tag=TAG, layers=LAYERS, heads=HEADS, seeds=SEEDS,
                   spearman_all=dict(rho=float(rho), p=float(p), n=len(a)),
                   spearman_learnable=dict(rho=float(rho_l), p=float(p_l), n=int(mask.sum())),
                   pearson_all=float(pearsonr(a, lg)[0]),
                   sufficiency_precision=dict(value=prec, n_high_alpha=int(hi.sum())),
                   minutes=round((time.time() - t0) / 60, 1))
    json.dump(dict(results=results, summary=summary),
              open(f"results/pilot_{TAG}_full.json", "w"), indent=2)
    print(f"\n==== SUMMARY [{TAG}] ====", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
