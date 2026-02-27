#!/usr/bin/env python3
"""
Chat API benchmark — measures time to first token (TTFT) and tokens per second.

Usage:
    python test_chat.py
    python test_chat.py --base-url http://localhost:8000/v1 --prompt "Explain MoE in one paragraph"
    python test_chat.py --runs 5
"""

import argparse
import time
import sys

try:
    from openai import OpenAI
except ImportError:
    sys.exit("openai package required: pip install openai")


DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL    = "qwen3.5-35b"
DEFAULT_PROMPT   = (
    "Explain the key differences between mixture-of-experts and dense transformer "
    "architectures, focusing on efficiency trade-offs."
)


def benchmark(client: OpenAI, model: str, prompt: str, verbose: bool = True) -> dict:
    messages = [{"role": "user", "content": prompt}]

    t_start = time.perf_counter()
    t_think_first: float | None = None  # first reasoning token
    t_answer_first: float | None = None  # first content token
    think_chunks: list[str] = []
    answer_chunks: list[str] = []

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
    )

    if verbose:
        print()

    usage = None
    in_thinking = False
    for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None

        # Reasoning / thinking tokens (Qwen3.5 chain-of-thought)
        # vLLM streams these as delta.reasoning (non-standard field → model_extra)
        extra = (getattr(delta, "model_extra", None) or {}) if delta else {}
        reasoning = extra.get("reasoning") or extra.get("reasoning_content")
        if reasoning:
            if t_think_first is None:
                t_think_first = time.perf_counter()
                in_thinking = True
                if verbose:
                    print("\033[2m[thinking…]\033[0m ", end="", flush=True)
            think_chunks.append(reasoning)
            if verbose:
                print(f"\033[2m{reasoning}\033[0m", end="", flush=True)

        # Answer tokens
        content = delta.content if delta else None
        if content:
            if t_answer_first is None:
                t_answer_first = time.perf_counter()
                if in_thinking and verbose:
                    think_ms = (t_answer_first - t_think_first) * 1000
                    print(
                        f"\n\033[2m[thought for {think_ms:.0f} ms / "
                        f"{len(think_chunks)} chars]\033[0m\n",
                        end="", flush=True,
                    )
                if verbose:
                    ttft_ms = (t_answer_first - t_start) * 1000
                    print(f"\033[2m[TTFT {ttft_ms:.0f} ms]\033[0m ", end="", flush=True)
            answer_chunks.append(content)
            if verbose:
                print(content, end="", flush=True)

        if hasattr(chunk, "usage") and chunk.usage is not None:
            usage = chunk.usage

    t_end = time.perf_counter()

    if verbose:
        print()

    answer_text    = "".join(answer_chunks)
    ttft           = (t_answer_first - t_start) if t_answer_first else None
    think_duration = (t_answer_first - t_think_first) if (t_think_first and t_answer_first) else None
    total_time     = t_end - t_start
    completion_tokens = (
        usage.completion_tokens if usage else len(answer_text.split())
    )
    tps = completion_tokens / total_time if total_time > 0 else 0.0

    return {
        "ttft_s":            ttft,
        "think_s":           think_duration,
        "think_chars":       len("".join(think_chunks)),
        "total_s":           total_time,
        "completion_tokens": completion_tokens,
        "tokens_per_sec":    tps,
    }


def main():
    parser = argparse.ArgumentParser(description="Chat API latency benchmark")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model",    default=DEFAULT_MODEL)
    parser.add_argument("--prompt",   default=DEFAULT_PROMPT)
    parser.add_argument("--runs",     type=int, default=1,
                        help="Number of benchmark runs (results are averaged)")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="not-required")

    print(f"Endpoint : {args.base_url}")
    print(f"Model    : {args.model}")
    print(f"Prompt   : {args.prompt[:80]}{'…' if len(args.prompt) > 80 else ''}")
    print(f"Runs     : {args.runs}")
    print()

    results = []
    for i in range(1, args.runs + 1):
        # Only stream tokens live on the first run (or when runs == 1)
        verbose = (i == 1)
        if args.runs > 1:
            print(f"\nRun {i}/{args.runs}", flush=True)
        try:
            r = benchmark(client, args.model, args.prompt, verbose=verbose)
            results.append(r)
            ttft_ms  = r["ttft_s"]  * 1000 if r["ttft_s"]  else float("nan")
            think_ms = r["think_s"] * 1000 if r["think_s"] else 0
            think_str = f"  |  think {think_ms:.0f} ms ({r['think_chars']} chars)" if r["think_s"] else ""
            print(
                f"\n\033[1mTTFT {ttft_ms:6.0f} ms  |  "
                f"{r['completion_tokens']:4d} tokens  |  "
                f"{r['tokens_per_sec']:6.1f} tok/s"
                f"{think_str}\033[0m"
            )
        except Exception as exc:
            print(f"FAILED: {exc}")

    if not results:
        sys.exit(1)

    if args.runs > 1:
        avg_ttft = sum(r["ttft_s"] for r in results if r["ttft_s"]) / len(results)
        avg_tps  = sum(r["tokens_per_sec"] for r in results) / len(results)
        print()
        print(f"Average  TTFT : {avg_ttft * 1000:.0f} ms")
        print(f"Average  tok/s: {avg_tps:.1f}")


if __name__ == "__main__":
    main()
