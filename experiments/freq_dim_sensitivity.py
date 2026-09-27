"""P1 #9: frequency / dimension sensitivity of alpha_L (predictor-only, no training).
Does the alpha_L verdict depend on sinusoidal/RoPE frequency count (= PE dimension m), the theta base,
or low-frequency scaling? We sweep each and report alpha_L on the core tasks at L0=20 -> L=60.
The claim is boundary-stability: the regime verdict (which tasks transfer) does not flip with these knobs."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import spectra as S

L0, L = 20, 60
TASKS = ["prev", "skip3", "copy", "reverse", "first"]


def row(pe, **kw):
    return {t: round(S.simulate(pe, t, L0, L, **kw)["alpha"], 2) for t in TASKS}


def main():
    out = {"L0": L0, "L": L, "tasks": TASKS, "rope": {}, "sinusoidal": {}}
    # RoPE: frequency count (num) and theta base
    for num in [8, 16, 32]:
        out["rope"][f"num={num} (m={2*num}), base=1e4"] = row("rope", num=num, base=10000.0)
    for base in [1e3, 1e4, 1e5]:
        out["rope"][f"num=16, base={base:.0e}"] = row("rope", num=16, base=base)
    # low-frequency RoPE scales (the resolution-limited regime; cross-ref tab:freqsweep)
    for sc in [0.5, 0.1, 0.03]:
        out["rope"][f"low-freq scale={sc}"] = row("rope-lowfreq", num=16, scale=sc)
    # Sinusoidal: frequency count (num) and base
    for num in [8, 16, 32]:
        out["sinusoidal"][f"num={num} (m={2*num}), base=1e4"] = row("sinusoidal", num=num, base=10000.0)
    for base in [1e3, 1e4, 1e5]:
        out["sinusoidal"][f"num=16, base={base:.0e}"] = row("sinusoidal", num=16, base=base)

    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/freq_dim_sensitivity.json", "w"), indent=2)
    for pe in ["rope", "sinusoidal"]:
        print(f"\n=== {pe} : alpha_L (L0={L0} -> L={L}) ===")
        print("setting".ljust(28) + "".join(t.ljust(9) for t in TASKS))
        for k, v in out[pe].items():
            print(k.ljust(28) + "".join(f"{v[t]:.2f}".ljust(9) for t in TASKS))
    print("\nFREQ_DIM_DONE")


if __name__ == "__main__":
    main()
