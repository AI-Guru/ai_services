#!/usr/bin/env python3
"""Paired comparison of depth_ab.py sweeps: analyze_ab.py A.jsonl B.jsonl [C.jsonl ...]

tok/s under MTP at temp 1.0 is dominated by draft acceptance, which varies run to run
and is not what a build or flag changes. ms per target forward pass is the stable
metric: every verify pass emits the target's own token plus the accepted drafts, so
passes = n_gen - draft_accepted. Same-depth runs agree within ~1 ms on it.
"""
import json, sys
from collections import defaultdict


def load(path):
    return [r for r in map(json.loads, open(path)) if r["valid"]]


def per_pass_ms(r):
    passes = r["n_gen"] - (r["draft_acc"] or 0)
    return (r["n_gen"] / r["tg_tps"] * 1000) / passes


def summarize(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["depth"]].append(r)
    out = {}
    for d, v in by.items():
        dn = sum(r["draft_n"] or 0 for r in v)
        da = sum(r["draft_acc"] or 0 for r in v)
        out[d] = {
            "tg": sum(r["tg_tps"] for r in v) / len(v),
            "pp": sum(r["pp_tps"] for r in v) / len(v),
            "ms_pass": sum(per_pass_ms(r) for r in v) / len(v),
            "tok_pass": sum(r["n_gen"] / (r["n_gen"] - (r["draft_acc"] or 0)) for r in v) / len(v),
            "acc": da / dn if dn else float("nan"),
            "n": len(v),
        }
    return out


paths = sys.argv[1:]
sums = [(p, summarize(load(p))) for p in paths]
depths = sorted(set().union(*[s.keys() for _, s in sums]))

for metric, fmt, better in [("ms_pass", "{:7.2f}", "lower"), ("tok_pass", "{:7.2f}", "higher"),
                            ("tg", "{:7.2f}", "higher"), ("pp", "{:7.0f}", "higher"), ("acc", "{:7.1%}", "")]:
    print(f"\n== {metric} ({better} is better)" if better else f"\n== {metric}")
    print("depth   " + " | ".join(f"{p.split('/')[-1][:16]:>16}" for p, _ in sums)
          + (" | last/first" if len(sums) > 1 else ""))
    for d in depths:
        vals = [s.get(d, {}).get(metric) for _, s in sums]
        cells = " | ".join(f"{fmt.format(v):>16}" if v is not None else f"{'-':>16}" for v in vals)
        ratio = ""
        if len(vals) > 1 and vals[0] and vals[-1] and metric != "acc":
            ratio = f" | {vals[-1] / vals[0]:.3f}x"
        print(f"{d:>7} " + cells + ratio)
