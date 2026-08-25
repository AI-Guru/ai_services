#!/usr/bin/env python3
"""Build the fp4-vs-fp8 scaling tables from the 2026-08-22 grid.

Reads benchmarks/guidellm/<label>-<scen>/benchmarks.json for every label and
scenario and prints markdown: aggregate tok/s, per-stream tok/s and request
latency, one block per metric so the same grid can be read three ways.
"""
import json, os, sys, math

SCENS = [("pp512", "512"), ("pp2k", "2K"), ("pp4k", "4K"),
         ("pp8k", "8K"), ("pp10k", "10K")]
LC = [("lc100k", "100K"), ("lc150k", "150K")]
LC_BUCKETS = [1, 4, 8, 10, 16]
# guidellm's achieved concurrency drifts below the requested rate; bucket it back.
BUCKETS = [1, 2, 4, 8, 16, 32, 64]

def g(d, *path, default=None):
    for k in path:
        if not isinstance(d, dict): return default
        d = d.get(k)
    return d if d is not None else default

# A 60s window is not enough for 8K/10K prompts at conc 32-64: the window closes
# while vLLM is still draining the prefill backlog, and the successful-only
# metrics then under-report badly (10K x 64 read 166.9 tok/s at 60s vs 324.2 at
# 300s). Those four cells come from the 300s re-run instead.
STEADY = {("pp8k", 32), ("pp8k", 64), ("pp10k", 32), ("pp10k", 64)}
STEADY_LABEL = "nvfp4-w4a4-steady"

def load(label, scen):
    p = f"benchmarks/guidellm/{label}-{scen}/benchmarks.json"
    if not os.path.exists(p): return {}
    data = json.load(open(p))
    out = {}
    for b in data.get("benchmarks", []):
        # guidellm 0.5.4 writes the requested stream count under config.strategy;
        # older files in this repo carry it under args. Accept either.
        req = (g(b, "config", "strategy", "streams") or g(b, "args", "strategy", "streams")
               or g(b, "args", "rate"))
        if req is None: continue
        req = int(round(float(req)))
        m = b.get("metrics", {})
        tpot = g(m, "time_per_output_token_ms", "successful", "median", default=float("nan"))
        dur = b.get("duration") or float("nan")
        ok = g(b, "scheduler_state", "successful_requests", default=0)
        out[req] = dict(
            agg=g(m, "output_tokens_per_second", "successful", "mean", default=float("nan")),
            ach=g(m, "request_concurrency", "successful", "mean", default=float("nan")),
            tpot=tpot,
            per=1000.0 / tpot if tpot and tpot == tpot and tpot > 0 else float("nan"),
            lat=g(m, "request_latency", "successful", "median", default=float("nan")),
            # At 100K+ prompts output_tokens_per_second ignores ~99.7% of the
            # work, so long-context rows need prompt+output throughput too.
            tot=g(m, "tokens_per_second", "successful", "mean", default=float("nan")),
            rpm=(ok / dur * 60.0) if dur and dur == dur and dur > 0 else float("nan"),
            ok=ok,
            err=g(b, "scheduler_state", "errored_requests", default=0),
        )
    return out

def fmt(v, d=1):
    return "—" if v is None or v != v else f"{v:.{d}f}"

def block(title, labels, scens, buckets, field, dec=1, note=""):
    print(f"\n### {title}")
    if note: print(f"\n{note}")
    print("\n| PP | quant | " + " | ".join(str(c) for c in buckets) + " |")
    print("|---|---|" + "---|" * len(buckets))
    for scen, nice in scens:
        for label, short in labels:
            d = load(label, scen)
            if not d: continue
            st = load(STEADY_LABEL, scen) if label.startswith("nvfp4-w4a4") else {}
            row = []
            for c in buckets:
                src = st if ((scen, c) in STEADY and c in st) else d
                row.append(fmt(src.get(c, {}).get(field), dec))
            print(f"| {nice} | {short} | " + " | ".join(row) + " |")

def block_lc(title, _labels, scens, buckets, field, dec=1, note=""):
    block(title, [("nvfp4-w4a4-lc2", "NVFP4")], scens, buckets, field, dec, note)

def errors(labels, scens, title="Errors"):
    bad = []
    for scen, nice in scens:
        for label, short in labels:
            for c, r in sorted(load(label, scen).items()):
                if r["err"]: bad.append(f"{short} {nice} conc{c}: {r['err']} errored / {r['ok']} ok")
    print(f"\n### {title}\n")
    print("\n".join(bad) if bad else "None. Every level of every scenario completed with zero failed requests.")

def achieved(labels, scens, buckets):
    print("\n### Achieved vs requested concurrency\n")
    print("| PP | quant | " + " | ".join(str(c) for c in buckets) + " |")
    print("|---|---|" + "---|" * len(buckets))
    for scen, nice in scens:
        for label, short in labels:
            d = load(label, scen)
            if not d: continue
            print(f"| {nice} | {short} | " + " | ".join(fmt(d.get(c, {}).get("ach")) for c in buckets) + " |")

if __name__ == "__main__":
    labels = [("nvfp4-w4a4", "NVFP4")]
    lc_labels = [("nvfp4-w4a4-lc2", "NVFP4")]
    print("# Qwen3.8-27B scaling grid — RTX PRO 6000, TG 300 (132-500)")
    block("Aggregate output tok/s", labels, SCENS, BUCKETS, "agg")
    block("Per-stream output tok/s (1000/TPOT)", labels, SCENS, BUCKETS, "per")
    block("Request latency p50 (s)", labels, SCENS, BUCKETS, "lat", 1)
    block_lc("Long context — aggregate output tok/s", labels, LC, LC_BUCKETS, "agg")
    block_lc("Long context — request latency p50 (s)", labels, LC, LC_BUCKETS, "lat")
    block_lc("Long context — per-stream tok/s", labels, LC, LC_BUCKETS, "per")
    block_lc("Long context — TOTAL tok/s (prompt + output)", labels, LC, LC_BUCKETS, "tot", 0,
          note="This is the row that says what the card is really doing: at 100K "
               "prompts the generated 300 tokens are ~0.3% of the work.")
    block_lc("Long context — completed requests/min", labels, LC, LC_BUCKETS, "rpm", 2)
    achieved(lc_labels, LC, LC_BUCKETS)
    errors(labels, SCENS, "Errors — prompt-length grid")
    errors(lc_labels, LC, "Errors — long context")
