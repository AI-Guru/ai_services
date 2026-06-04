#!/usr/bin/env python3
"""
Embedding / reranker throughput benchmark for the Qwen3-Embedding family.

Measures single-request latency, batched throughput (one request, many inputs),
and concurrent throughput (many parallel requests) against an OpenAI-compatible
vLLM pooling endpoint.

Usage:
    python test_embed.py --port 11463 --model qwen3-embedding-0.6b
    python test_embed.py --port 11466 --model qwen3-reranker-0.6b --rerank
"""

import argparse, time, statistics, urllib.request, json
from concurrent.futures import ThreadPoolExecutor

SAMPLE = ("Mixture-of-experts models activate only a subset of parameters per "
          "token, trading memory bandwidth for compute efficiency at scale. ")


def post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    r = json.loads(urllib.request.urlopen(req).read())
    return time.perf_counter() - t, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=11463)
    ap.add_argument("--model", default="qwen3-embedding-0.6b")
    ap.add_argument("--rerank", action="store_true", help="benchmark /v1/score instead of /v1/embeddings")
    ap.add_argument("--batch", type=int, default=64, help="inputs per batched request")
    ap.add_argument("--concurrency", type=int, default=32, help="parallel single-item requests")
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--doc-len", type=int, default=4, help="repeats of the sample sentence per item (~tokens)")
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    text = SAMPLE * args.doc_len

    if args.rerank:
        url = f"{base}/v1/score"
        one = lambda: post(url, {"model": args.model, "text_1": "What is an MoE model?", "text_2": [text]})
        batch_payload = {"model": args.model, "text_1": "What is an MoE model?", "text_2": [text] * args.batch}
        unit = "pairs"
    else:
        url = f"{base}/v1/embeddings"
        one = lambda: post(url, {"model": args.model, "input": text})
        batch_payload = {"model": args.model, "input": [text] * args.batch}
        unit = "embeddings"

    print(f"Endpoint : {url}")
    print(f"Model    : {args.model}   (~{len(text)//4} tokens/item)\n")

    # warmup
    for _ in range(3): one()

    # 1) single-request latency
    lat = [one()[0] for _ in range(args.runs)]
    lat.sort()
    print(f"Single-request latency ({args.runs} runs):")
    print(f"  mean {statistics.mean(lat)*1000:6.1f} ms | p50 {lat[len(lat)//2]*1000:6.1f} ms | "
          f"p99 {lat[min(len(lat)-1, int(len(lat)*0.99))]*1000:6.1f} ms")
    print(f"  -> {1/statistics.mean(lat):6.1f} {unit}/s sequential\n")

    # 2) batched throughput (one request, many inputs)
    dt, r = post(url, batch_payload)
    n = len(r["data"])
    print(f"Batched request ({n} {unit} in one call):")
    print(f"  {dt*1000:6.1f} ms total -> {n/dt:7.1f} {unit}/s\n")

    # 3) concurrent throughput (many parallel single-item requests)
    N = args.concurrency * 4
    t = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        list(ex.map(lambda _: one(), range(N)))
    dt = time.perf_counter() - t
    print(f"Concurrent ({N} requests @ {args.concurrency} parallel):")
    print(f"  {dt*1000:6.1f} ms total -> {N/dt:7.1f} {unit}/s")


if __name__ == "__main__":
    main()
