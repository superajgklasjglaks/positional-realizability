"""Synthetic position-essential tasks, shared format with the predictor.

Given a random sequence s of length n over a small vocab, the target at query position i is the
token s[t_n(i)], where t_n is the SAME target function used by src/spectra.py. Small vocab forces
the model to localize the position t_n(i) rather than retrieve by content (position-essential).
"""

import os
import torch

VOCAB = int(os.environ.get("VOCAB", 12))  # small -> repeated tokens -> position-essential

# target position t_n(i); None where undefined (masked out of the loss)
def _half(n):
    return (n + 1) // 2

TARGET = {
    "prev":    lambda n, i: i - 1,
    "skip2":   lambda n, i: i - 2,
    "skip3":   lambda n, i: i - 3,
    "skip5":   lambda n, i: i - 5,
    "copy":    lambda n, i: i - _half(n),
    "reverse": lambda n, i: (_half(n) - 1) - (i - _half(n)),
    "first":   lambda n, i: 0,
    # fixed absolute positions in the interior (not the causal boundary position 0)
    "pos2":    lambda n, i: 2,
    "pos5":    lambda n, i: 5,
}
QSTART = {"prev": 1, "skip2": 2, "skip3": 3, "skip5": 5,
          "copy": None, "reverse": None, "first": 1,
          "pos2": 3, "pos5": 6}  # None -> second half


def make_batch(task, n, batch, device, vocab=VOCAB):
    if task == "induction":
        return _make_induction(n, batch, device, vocab)
    s = torch.randint(0, vocab, (batch, n), device=device)
    tgt = torch.full((batch, n), -100, device=device, dtype=torch.long)  # -100 = ignore_index
    tfn = TARGET[task]
    qs = QSTART[task]
    qstart = qs if qs is not None else _half(n)
    for i in range(qstart, n):
        t = tfn(n, i)
        if t is None or t < 0 or t > i:
            continue
        tgt[:, i] = s[:, t]
    return s, tgt


def _make_induction(n, batch, device, vocab):
    """Content-driven task (induction head): target at i = token following the previous occurrence of
    x_i. The target POSITION depends on the sequence content, not on i -- so no length-dependent
    positional rule realizes it, but a length-independent content-match circuit does."""
    s = torch.randint(0, vocab, (batch, n), device=device)
    tgt = torch.full((batch, n), -100, device=device, dtype=torch.long)
    sl = s.tolist()
    for b in range(batch):
        last = {}
        row = sl[b]
        for i in range(n):
            tok = row[i]
            if tok in last and last[tok] + 1 < i:           # previous occurrence exists; copy its successor
                tgt[b, i] = row[last[tok] + 1]
            last[tok] = i
    return s, tgt


def query_positions(task, n):
    if task == "induction":
        return list(range(1, n))           # validity is content-dependent; eval masks on tgt != -100
    qs = QSTART[task]
    qstart = qs if qs is not None else _half(n)
    out = []
    for i in range(qstart, n):
        t = TARGET[task](n, i)
        if t is not None and 0 <= t <= i:
            out.append(i)
    return out
