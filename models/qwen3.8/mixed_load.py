#!/usr/bin/env python3
"""
Mixed text+image concurrent load test.

WHY THIS IS SEPARATE FROM GUIDELLM
----------------------------------
GuideLLM is text-only. It cannot answer the question that actually decides
whether a speculative multimodal config is production-safe: what happens when a
batch contains BOTH text-only and image-bearing requests at the same time.

That combination is the known-dangerous one for this family:
  * On Qwen3.6, MTP speculation crashed the engine on mixed-modality batches.
  * This server logs at startup:
        Draft model Qwen3DSparkForCausalLM does not support external multimodal
        embeddings. Embeddings from the target model will not be passed to the
        drafter; using text-only draft inputs instead.
    so on an image turn the drafter is predicting from text alone. That should
    only cost acceptance - but "should" is why we measure it.

Reports per request class (text vs image) so a modality-specific failure or
slowdown cannot hide inside an average.

Usage:
  python3 mixed_load.py --concurrency 8 --requests 48 --label cfg01
"""
import argparse, base64, json, random, statistics, sys, threading, time
import urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

IMAGES = [
    ("small-0.5MP", "../../scripts/testimage.jpg"),
    ("med-1.05MP",  "../../church1.png"),
    ("big-4.75MP",  "../../iconography.png"),
]
TEXT_PROMPTS = [
    "Explain the tradeoffs between optimistic and pessimistic locking.",
    "Summarise the CAP theorem and give one real-world example per corner.",
    "Write a Python function that merges two sorted lists, with a docstring.",
    "What causes TCP head-of-line blocking and how does QUIC address it?",
]

def load_images():
    out = []
    for name, path in IMAGES:
        try:
            b = open(path, "rb").read()
            ext = path.rsplit(".", 1)[-1].replace("jpg", "jpeg")
            out.append((name, f"data:image/{ext};base64," + base64.b64encode(b).decode()))
        except FileNotFoundError:
            print(f"  (skipping missing image {path})", file=sys.stderr)
    return out

def request(base, model, kind, payload, max_tokens):
    if kind == "text":
        content = payload
    else:
        content = [{"type": "image_url", "image_url": {"url": payload}},
                   {"type": "text", "text": "Describe this image in two sentences."}]
    body = {"model": model, "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens, "temperature": 1.0, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer none"})
    t0 = time.perf_counter(); ttft = None; usage = None
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data: "): continue
                p = line[6:]
                if p == "[DONE]": break
                c = json.loads(p)
                if c.get("usage"): usage = c["usage"]
                if not c.get("choices"): continue
                d = c["choices"][0].get("delta") or {}
                if (d.get("content") or d.get("reasoning")) and ttft is None:
                    ttft = time.perf_counter() - t0
    except Exception as e:
        return dict(ok=False, kind=kind, err=f"{type(e).__name__}: {str(e)[:120]}")
    total = time.perf_counter() - t0
    ntok = (usage or {}).get("completion_tokens", 0)
    ptok = (usage or {}).get("prompt_tokens", 0)
    dec = total - (ttft or 0)
    return dict(ok=True, kind=kind, ttft=ttft or 0, total=total, ntok=ntok, ptok=ptok,
                tps=(ntok - 1) / dec if dec > 0 and ntok > 1 else 0.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:11484/v1")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--requests", type=int, default=48)
    ap.add_argument("--image-frac", type=float, default=0.5)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--label", default="run")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    imgs = load_images()
    if not imgs: sys.exit("no images available")
    rnd = random.Random(a.seed)
    jobs = []
    for i in range(a.requests):
        if rnd.random() < a.image_frac:
            nm, data = imgs[i % len(imgs)]
            jobs.append(("image:" + nm, data))
        else:
            jobs.append(("text", rnd.choice(TEXT_PROMPTS)))
    rnd.shuffle(jobs)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        res = list(ex.map(lambda j: request(a.base_url, a.model,
                                            "text" if j[0] == "text" else "image",
                                            j[1], a.max_tokens), jobs))
    wall = time.perf_counter() - t0

    ok = [r for r in res if r["ok"]]; bad = [r for r in res if not r["ok"]]
    tot_out = sum(r["ntok"] for r in ok)
    print(f"\n### mixed text+image | {a.label} | concurrency {a.concurrency} | "
          f"{a.requests} requests ({int(a.image_frac*100)}% image)")
    print(f"wall {wall:.1f}s | aggregate {tot_out/wall:.1f} output tok/s | "
          f"ok {len(ok)}/{len(res)} | errors {len(bad)}")
    if bad:
        from collections import Counter
        for e, n in Counter(r["err"] for r in bad).most_common():
            print(f"  ERROR x{n}: {e}")
    print("| class | n | per-stream tok/s | TTFT p50 | latency p50 | latency p95 | prompt tok |")
    print("|---|---|---|---|---|---|---|")
    for cls in ("text", "image"):
        sub = [r for r in ok if r["kind"] == cls]
        if not sub: continue
        lat = sorted(r["total"] for r in sub)
        p95 = lat[min(len(lat)-1, int(0.95*len(lat)))]
        print(f"| {cls} | {len(sub)} | {statistics.median(r['tps'] for r in sub):.1f} | "
              f"{1000*statistics.median(r['ttft'] for r in sub):.0f} ms | "
              f"{statistics.median(lat):.1f} s | {p95:.1f} s | "
              f"{statistics.median(r['ptok'] for r in sub):.0f} |")
    return 1 if bad else 0

sys.exit(main())
