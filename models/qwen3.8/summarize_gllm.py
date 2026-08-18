#!/usr/bin/env python3
"""Flatten a guidellm benchmarks.json into one row per concurrency level."""
import json, sys, os

def g(d, *path, default=None):
    for k in path:
        if d is None: return default
        d = d.get(k) if isinstance(d, dict) else None
    return d if d is not None else default

def main():
    path, label, scen = sys.argv[1], sys.argv[2], sys.argv[3]
    if not os.path.exists(path):
        print(f"  !! no json at {path}"); return
    data = json.load(open(path))
    bms = data.get("benchmarks") or []
    rows = []
    for b in bms:
        conc = g(b, "args", "strategy", "streams") or g(b, "args", "rate") or \
               round(g(b, "metrics", "request_concurrency", "successful", "mean", default=0) or 0)
        m = b.get("metrics", {})
        def stat(name, sub="successful", field="median"):
            return g(m, name, sub, field, default=float("nan"))
        rows.append(dict(
            conc=float(conc or 0),
            out_tps=stat("output_tokens_per_second", field="mean"),
            tot_tps=stat("tokens_per_second", field="mean"),
            ttft=stat("time_to_first_token_ms"),   # median is ~0, see note below
            ttft95=g(m,"time_to_first_token_ms","successful","p95",default=float("nan")),
            tpot=stat("time_per_output_token_ms"),
            lat=stat("request_latency"),
            lat95=g(m,"request_latency","successful","p95",default=float("nan")),
            # counts live in scheduler_state, not request_totals (which is null in 0.5.4)
            err=g(b,"scheduler_state","errored_requests",default=0),
            ok=g(b,"scheduler_state","successful_requests",default=0),
            ttft_mean=g(m,"time_to_first_token_ms","successful","mean",default=float("nan")),
        ))
    rows.sort(key=lambda r: r["conc"])
    print(f"\n### {label} / {scen}")
    print("| conc | out tok/s | per-stream tok/s | TPOT | lat p50 | lat p95 | TTFT mean | ok | err |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        ps = 1000.0/r["tpot"] if r["tpot"] and r["tpot"]==r["tpot"] and r["tpot"]>0 else float("nan")
        print(f"| {r['conc']:.0f} | {r['out_tps']:.1f} | {ps:.1f} | {r['tpot']:.1f} ms | "
              f"{r['lat']:.1f} s | {r['lat95']:.1f} s | {r['ttft_mean']:.0f} ms | "
              f"{r['ok']:.0f} | {r['err']:.0f} |")
    # GuideLLM's TTFT MEDIAN is ~0 against vLLM: vLLM emits an initial SSE frame
    # carrying an empty role delta the instant the request is admitted, and
    # guidellm timestamps that as the first token. Only the mean is informative,
    # and even it is contaminated. Use request latency to judge responsiveness.

main()
