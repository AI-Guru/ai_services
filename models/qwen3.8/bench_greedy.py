#!/usr/bin/env python3
"""
Greedy decode benchmark — the protocol published checkpoint cards use.

WHY THIS EXISTS ALONGSIDE ../shared/test_chat.py
-------------------------------------------------
test_chat.py measures the endpoint as clients actually hit it: the model's own
generation_config sampling (temperature=1.0, top_p=0.95, top_k=20) and thinking
on. That is the right number for "how fast is our endpoint", and every tok/s in
../qwen3.8/README.md's main table is measured that way.

It is the WRONG number to compare against a checkpoint card, because speculative
decoding's acceptance rate is a function of the target's sampling distribution.
At temperature 0 the target is deterministic, so a draft token is accepted iff
it is the argmax; at temperature 1.0 the target samples from a 248K-way
distribution and rejects far more. Vendor cards (gittensor's 80.6 / 136.9 /
147.9 tok/s on a 5090, for instance) are measured at `temperature=0` with
thinking off. Quoting those against our sampled numbers compares two different
experiments.

This script reproduces the vendor protocol so the comparison is honest:
  temperature=0, thinking off, ignore_eos, fixed output length (default 256).
`ignore_eos` is what makes the token count exact — otherwise a short answer
ends early and the average is dominated by TTFT.

Usage:
  python3 bench_greedy.py --base-url http://localhost:11484/v1 \
      --model qwen3.8-27b --runs 3 --output-tokens 256
"""
import argparse, json, time, urllib.request

PROMPT = ("Explain the key differences between mixture-of-experts and dense "
          "transformer architectures, focusing on efficiency trade-offs.")


def one(base_url, model, prompt, n_out, think, temperature=0):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": temperature,
        "max_tokens": n_out,
        # ignore_eos is a vLLM extension: keep generating to exactly n_out so
        # every run measures the same amount of decode work.
        "ignore_eos": True,
    }
    if not think:
        # NOTE: must be inside chat_template_kwargs. A top-level
        # enable_thinking is silently ignored by vLLM — see README.
        body["chat_template_kwargs"] = {"enable_thinking": False}

    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer none"},
    )
    t0 = time.perf_counter()
    ttft = None
    nchunk = 0
    usage = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage = chunk["usage"]
            if not chunk.get("choices"):
                continue
            d = chunk["choices"][0].get("delta") or {}
            if d.get("content") or d.get("reasoning") or d.get("reasoning_content"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                nchunk += 1
    total = time.perf_counter() - t0
    decode = total - (ttft or 0)

    # COUNT TOKENS FROM usage, NEVER FROM STREAM CHUNKS.
    # Under speculative decoding vLLM emits every token accepted in one
    # verify step as a SINGLE SSE chunk, so chunks are ~1/accept_length of
    # tokens. Counting chunks silently reports ~half the real throughput and
    # the error scales with how well speculation is working — i.e. it looks
    # like the fastest configs are the slowest.
    ntok = (usage or {}).get("completion_tokens") or nchunk
    return ttft, ntok, (ntok - 1) / decode if decode > 0 and ntok > 1 else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:11484/v1")
    p.add_argument("--model", default="qwen3.8-27b")
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--output-tokens", type=int, default=256)
    p.add_argument("--think", action="store_true",
                   help="leave thinking ON (default: off, matching vendor cards)")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0 = vendor-card protocol. Use 1.0 (with --think) to "
                        "measure the endpoint's REAL sampling distribution at a "
                        "FIXED token budget: same workload as test_chat.py but "
                        "without its dominant variance source, which is that "
                        "thinking length varies run to run, not the engine.")
    a = p.parse_args()

    one(a.base_url, a.model, "hi", 8, a.think, a.temperature)  # warmup
    rates, ttfts = [], []
    for i in range(a.runs):
        ttft, ntok, rate = one(a.base_url, a.model, a.prompt,
                               a.output_tokens, a.think, a.temperature)
        rates.append(rate)
        ttfts.append(ttft or 0)
        print(f"  run {i+1}: TTFT {1000*(ttft or 0):7.0f} ms | "
              f"{ntok:5d} tok | {rate:6.1f} tok/s")
        if ntok < a.output_tokens:
            print(f"    ! only {ntok}/{a.output_tokens} tokens - ignore_eos "
                  f"not honoured by this endpoint?")
    tag = "GREEDY" if a.temperature == 0 else "FIXEDLEN"
    print(f"{tag} avg tok/s: {sum(rates)/len(rates):.1f}  "
          f"avg TTFT: {1000*sum(ttfts)/len(ttfts):.0f} ms  "
          f"(temperature={a.temperature}, thinking={'on' if a.think else 'off'}, "
          f"ignore_eos, {a.output_tokens} out)")


if __name__ == "__main__":
    main()
