#!/usr/bin/env python3
"""Depth sweep for a speculative-decoding llama-server, avoiding both traps in CLAUDE.md.

- No ignore_eos: forced generation degenerates into repetition that a draft head
  predicts perfectly, inflating MTP. The model answers a real question about real text.
- n_gen is asserted: at temp 1.0 the model sometimes emits EOS at once. Runs below
  --min-gen are retried, never averaged in.
- Prefix reuse is defeated: cache_prompt false, plus a different corpus offset per run.

Rows are appended to --out as JSON lines as they complete, so an interrupted sweep
keeps what it measured. Compare two sweeps with analyze_ab.py.
"""
import argparse, json, sys, time, urllib.request

QUESTION = ("\n\n---\n\nWrite a detailed, well-structured technical summary of the "
            "document above: cover every section, the key measurements, and the "
            "open problems. Aim for about 800 words.")


def post(url, path, body, timeout=3600):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:11499")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--depths", default="2048,4096,8192,16384,32768,65536,131072,250000")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--n-predict", type=int, default=512)
    ap.add_argument("--min-gen", type=int, default=400)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    toks = post(a.url, "/tokenize", {"content": open(a.corpus).read()})["tokens"]
    print(f"corpus: {len(toks)} tokens", flush=True)

    rows = []
    out = open(a.out, "a")
    for depth in map(int, a.depths.split(",")):
        body_len = depth - 200  # room for the chat template + question
        for run in range(a.runs):
            for attempt in range(3):
                off = (run * 3 + attempt) * 1500
                if off + body_len > len(toks):
                    off = max(0, len(toks) - body_len)
                if body_len > len(toks):
                    sys.exit(f"corpus too short for depth {depth}")
                text = post(a.url, "/detokenize", {"tokens": toks[off:off + body_len]})["content"]
                prompt = post(a.url, "/apply-template",
                              {"messages": [{"role": "user", "content": text + QUESTION}]})["prompt"]
                t0 = time.time()
                r = post(a.url, "/completion", {
                    "prompt": prompt, "n_predict": a.n_predict, "cache_prompt": False,
                    "temperature": 1.0, "top_p": 0.95, "top_k": 20, "seed": 1000 + run})
                t = r["timings"]
                row = {"label": a.label, "depth": depth, "run": run, "attempt": attempt,
                       "prompt_n": t["prompt_n"], "pp_tps": t["prompt_per_second"],
                       "n_gen": t["predicted_n"], "tg_tps": t["predicted_per_second"],
                       "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted"),
                       "wall_s": round(time.time() - t0, 1)}
                ok = row["n_gen"] >= a.min_gen
                row["valid"] = ok
                rows.append(row)
                out.write(json.dumps(row) + "\n")
                out.flush()
                acc = (f"{row['draft_acc'] / row['draft_n']:.1%}" if row["draft_n"] else "n/a")
                print(f"{a.label} d={depth:>6} run={run} try={attempt} prompt_n={row['prompt_n']:>6} "
                      f"pp={row['pp_tps']:8.1f} tg={row['tg_tps']:7.2f} n_gen={row['n_gen']:>4} "
                      f"acc={acc} {'OK' if ok else 'SHORT-retry'}", flush=True)
                if ok:
                    break
    out.close()

    print(f"\n{a.label}: depth | tg mean | pp mean | acceptance (valid runs only)")
    for depth in sorted({r["depth"] for r in rows}):
        v = [r for r in rows if r["depth"] == depth and r["valid"]]
        if not v:
            print(f"{depth:>7} | NO VALID RUN")
            continue
        tg = sum(r["tg_tps"] for r in v) / len(v)
        pp = sum(r["pp_tps"] for r in v) / len(v)
        dn = sum(r["draft_n"] or 0 for r in v)
        da = sum(r["draft_acc"] or 0 for r in v)
        print(f"{depth:>7} | {tg:7.2f} | {pp:8.1f} | {da / dn:.1%} ({len(v)} runs)" if dn
              else f"{depth:>7} | {tg:7.2f} | {pp:8.1f} | n/a ({len(v)} runs)")


if __name__ == "__main__":
    main()
