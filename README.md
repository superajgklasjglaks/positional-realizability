# A Positional Realizability Theory of Length Generalization in Single-Head Attention

Code and data for the TMLR paper

> Jiasong Han, Yunkai Chai, Jintao Wang. *A Positional Realizability Theory of Length Generalization in
> Single-Head Attention.* Transactions on Machine Learning Research, 2026.
> [OpenReview](https://openreview.net/forum?id=DRwVtZF4b6)

The paper studies why length generalization depends so sharply on the positional encoding. A
*simulation transfer operator* re-evaluates a selected length-`L0`-realizing positional kernel at a longer
length `L`, and the scalar `alpha_L` in [0, 1] is the fraction of scored new rows whose masked argmax
still hits the task's target. `alpha_L` is computed from the positional feature map and the task target
alone, before any Transformer is trained.

## Layout

```
src/           spectra.py  -- alpha_L (simulate(pe, task, L0, L))
               model.py    -- tiny decoder-only transformer with 5 positional encodings
               data.py     -- the 9 synthetic tasks
experiments/   run_study.py              main study (resumable)
               review_stats.py           correlations, bootstrap CIs, precisions
               make_figures.py           figures
               l0_sensitivity.py         training-length robustness
               freq_dim_sensitivity.py   frequency / theta-base / dimension robustness
               verify_*.py, audit_check.py, iff_constant_offset_lg.py, ...
                                         executable checks for the formal statements
results/       study_rows.json (2625 rows), study_rowsabsinterior.json (750 rows),
               study_rows_v64d128.json, l0_sensitivity.json, freq_dim_sensitivity.json, ...
figures/       figures used in the paper
```

## Reproducing the results

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/spectra.py                         # alpha_L table (no training)
python experiments/run_study.py               # main study, 5 seeds (~55 min on an Apple M-series laptop)
python experiments/review_stats.py            # headline correlations and precisions
python experiments/make_figures.py            # figures

# interior fixed-absolute positions (Appendix)
TASKS_ONLY=pos2,pos5 STUDY_TAG=absinterior python experiments/run_study.py
# robustness studies
python experiments/l0_sensitivity.py
python experiments/freq_dim_sensitivity.py
```

`run_study.py` is resumable and reads `SEEDS`, `DMODEL`, `TIERS_ONLY`, `TASKS_ONLY` and `STUDY_TAG` from
the environment. Use a fresh `STUDY_TAG` to avoid reusing the stored rows in `results/`.

## Checks for the formal statements

| Statement | Script |
|---|---|
| Theorem 2 (density-one constancy and attainability) | `experiments/verify_thm2_density.py`, `experiments/iff_constant_offset_lg.py` |
| Hypotheses of Theorem 2 and Propositions 2, 4 (counterexamples) | `experiments/audit_check.py` |
| Proposition 2 (conditional chance rate; target-in-`N_i` hypothesis) | `experiments/verify_prop2_conditional.py`, `experiments/verify_prop2_hypothesis.py`, `experiments/theorem2b_learned_abs_montecarlo.py` |
| Relative growing-offset impossibility | `experiments/theorem_relative_copy_impossibility.py` |
| Absolute/relative duality | `experiments/duality_abs_vs_rel.py` |
| Conjecture 1 probes | `experiments/underdetermined_conj1_*.py` |

Each script exits with a non-zero status if its checks fail.

## Citation

```bibtex
@article{han2026positional,
  title   = {A Positional Realizability Theory of Length Generalization in Single-Head Attention},
  author  = {Han, Jiasong and Chai, Yunkai and Wang, Jintao},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  url     = {https://openreview.net/forum?id=DRwVtZF4b6}
}
```

## License

MIT
