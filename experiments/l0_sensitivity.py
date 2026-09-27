"""P1 #8: L0-sensitivity. Is the AUC=1.0 screen / Theorem-2 verdict an artifact of L0=20?
Re-run the 1L1H study at L0 in {10,20,40} (test lengths scaled to L0) and report, per L0, the
cell-level alpha_L-screen AUC for LG>=0.5 and the per-(PE,task) generalize verdicts."""
import os, sys, json, time, random
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import data as D
from model import Tiny
from data import make_batch, query_positions, VOCAB
import spectra as S

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
PES = ["learned-abs", "nope", "alibi", "sinusoidal", "rope"]
TASKS = ["prev", "skip3", "copy", "reverse", "first"]
SEEDS = 3
STEPS = 1800


def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)


def train(pe, task, L0):
    m = Tiny(VOCAB, d=64, h=1, layers=1, pe=pe, max_train_len=L0).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01); m.train()
    for _ in range(STEPS):
        n = random.randint(max(6, L0 // 2), L0)
        s, tgt = make_batch(task, n, 64, DEV)
        loss = torch.nn.functional.cross_entropy(m(s).view(-1, VOCAB), tgt.view(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()
    return m


@torch.no_grad()
def acc(m, task, n, lo, hi, reps=6):
    m.eval(); qp = [i for i in query_positions(task, n) if lo <= i < hi]
    if not qp: return float("nan")
    c = t = 0
    for _ in range(reps):
        s, tgt = make_batch(task, n, 64, DEV); pred = m(s).argmax(-1)
        for i in qp:
            msk = tgt[:, i] != -100
            c += int((pred[:, i][msk] == tgt[:, i][msk]).sum()); t += int(msk.sum())
    return c / t if t else float("nan")


def auc(a, y):
    p = a[y == 1]; n = a[y == 0]
    return float(np.mean([(x > z) + 0.5 * (x == z) for x in p for z in n])) if len(p) and len(n) else float("nan")


def main():
    out = {}
    t0 = time.time()
    for L0 in [10, 20, 40]:
        Ltest = int(round(3 * L0))            # extrapolate to 3x train length
        cells = {}
        for pe in PES:
            for task in TASKS:
                a = S.simulate(pe, task, L0, Ltest)["alpha"]
                lgs, trs = [], []
                for seed in range(SEEDS):
                    set_seed(seed); m = train(pe, task, L0)
                    trs.append(acc(m, task, L0, L0 // 2, L0))
                    lgs.append(acc(m, task, Ltest, L0, Ltest))
                cells[f"{pe}|{task}"] = dict(alpha=a, lg=float(np.mean(lgs)), train=float(np.mean(trs)))
                print(f"L0={L0} {pe:12s}{task:8s} a={a:.2f} LG={np.mean(lgs):.2f} tr={np.mean(trs):.2f} "
                      f"({(time.time()-t0)/60:.1f}m)", flush=True)
        A = np.array([c["alpha"] for c in cells.values()])
        G = np.array([c["lg"] for c in cells.values()])
        y = (G >= 0.5).astype(int)
        out[L0] = dict(Ltest=Ltest, auc=auc(A, y), n_pos=int(y.sum()), n=len(y), cells=cells)
        print(f"==> L0={L0}: screen AUC={out[L0]['auc']:.3f} ({y.sum()}/{len(y)} generalize)", flush=True)
        json.dump(out, open("results/l0_sensitivity.json", "w"), indent=2)
    print("\nSUMMARY:", {L0: round(out[L0]["auc"], 3) for L0 in out}, flush=True)
    print("L0_SENSITIVITY_DONE", flush=True)


if __name__ == "__main__":
    main()
