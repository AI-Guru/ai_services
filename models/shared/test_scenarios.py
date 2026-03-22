#!/usr/bin/env python3
"""
Multi-scenario benchmark — measures tok/s across different workload profiles.

Tests chat, RAG, code generation, summarization, and agentic scenarios
with realistic input/output ratios. Each scenario runs multiple times
and reports average throughput.

Usage:
    python test_scenarios.py
    python test_scenarios.py --base-url http://localhost:11440/v1 --model nemotron-cascade2-30b
    python test_scenarios.py --runs 5
    python test_scenarios.py --scenario chat codegen
    python test_scenarios.py --no-think
"""

import argparse
import time
import sys
import textwrap

try:
    from openai import OpenAI
except ImportError:
    sys.exit("openai package required: pip install openai")


DEFAULT_BASE_URL = "http://localhost:11435/v1"
DEFAULT_MODEL = "qwen3.5-35b"

# ---------------------------------------------------------------------------
# Scenario definitions
#
# Each scenario has a prompt designed to elicit a specific input/output ratio
# and a max_tokens cap that controls output length.
# ---------------------------------------------------------------------------

SCENARIOS = {
    "chat": {
        "label": "Chat",
        "description": "Short conversational Q&A (~200 tok in, ~300 tok out)",
        "max_tokens": 400,
        "prompt": "What are the three most important things to consider when choosing a GPU for local LLM inference? Be concise.",
    },
    "rag": {
        "label": "RAG",
        "description": "Long context retrieval (~2K tok in, ~200 tok out)",
        "max_tokens": 300,
        "prompt": textwrap.dedent("""\
            Based on the following retrieved documents, answer the question at the end.

            Document 1: Mixture-of-Experts (MoE) models use a gating network to route each
            token to a subset of expert subnetworks. This allows scaling total parameter count
            without proportionally increasing per-token compute. Common routing strategies
            include top-k selection with auxiliary load-balancing losses. The Switch Transformer
            introduced simplified single-expert routing. Mixtral 8x7B routes to 2 of 8 experts
            per token, achieving 47B total parameters with ~14B active compute. GShard scales
            to 600B parameters across thousands of devices using expert parallelism. The key
            trade-off is between model capacity (total params) and inference efficiency (active
            params per token). Expert collapse, where the router converges to using only a few
            experts, remains a challenge addressed through load-balancing losses and dropout.

            Document 2: Dense transformer architectures process every token through all model
            parameters. This makes them simpler to train and deploy but limits scaling due to
            the linear relationship between model size and compute cost. Techniques like tensor
            parallelism, pipeline parallelism, and sequence parallelism enable training dense
            models across multiple GPUs. Flash Attention reduces memory usage from O(n²) to
            O(n) for the attention mechanism. Key-value caching avoids recomputing attention
            for previous tokens during autoregressive generation. Quantization (FP8, INT4, NF4)
            reduces memory footprint and can improve throughput on hardware with dedicated
            low-precision compute units. PagedAttention in vLLM manages KV cache memory with
            virtual memory techniques, enabling efficient batching of requests with different
            sequence lengths.

            Document 3: Hybrid architectures combine multiple mechanism types within a single
            model. Mamba-2 uses selective state space models (SSMs) as an alternative to
            attention for sequence modeling, achieving linear complexity in sequence length.
            NemotronH interleaves Mamba-2 layers with standard attention layers and MoE feed-
            forward layers in patterns like M-M-M-*-E-M-M-*-E where M=Mamba, *=Attention,
            E=MoE. This hybrid approach captures both the linear-time efficiency of SSMs for
            long sequences and the strong in-context learning of attention. Gated DeltaNet is
            another linear attention variant used in Qwen3.5's hybrid architecture. The key
            benefit is reduced KV cache size for the SSM layers while maintaining attention's
            ability to handle tasks requiring precise token-to-token lookups.

            Document 4: Inference serving frameworks differ in how they handle concurrent
            requests. vLLM uses continuous batching where new requests join the batch as soon
            as a slot opens, maximizing GPU utilization. PagedAttention manages KV cache as
            non-contiguous blocks, reducing memory waste from fragmentation. SGLang uses
            RadixAttention for automatic KV cache reuse across requests sharing common prefixes.
            llama.cpp provides high single-request throughput through optimized CUDA kernels
            and GGUF quantization but lacks continuous batching, limiting multi-user scaling.
            TensorRT-LLM uses FP8 quantization with custom CUDA kernels optimized for NVIDIA
            hardware. The choice between frameworks depends on whether the deployment prioritizes
            single-user latency (llama.cpp) or multi-user throughput (vLLM, SGLang).

            Question: Compare the memory efficiency trade-offs between MoE, dense, and hybrid
            architectures for inference serving. Which approach gives the best throughput per
            GB of VRAM?
        """),
    },
    "codegen": {
        "label": "Code generation",
        "description": "Medium context, long output (~500 tok in, ~800 tok out)",
        "max_tokens": 1000,
        "prompt": textwrap.dedent("""\
            Write a Python class `AsyncRateLimiter` that implements a token bucket rate limiter
            for async HTTP requests. Requirements:

            1. Constructor takes `rate` (requests/sec) and `burst` (max tokens)
            2. `async acquire()` method that waits if no tokens available
            3. `async __aenter__` / `__aexit__` for context manager usage
            4. Thread-safe using asyncio.Lock
            5. Include docstrings and type hints
            6. Include a usage example with aiohttp

            Write clean, production-ready code.
        """),
    },
    "summarization": {
        "label": "Summarization",
        "description": "Long input, short output (~2K tok in, ~150 tok out)",
        "max_tokens": 200,
        "prompt": textwrap.dedent("""\
            Summarize the following technical discussion in 3-4 bullet points:

            The debate around optimal GPU memory allocation for LLM inference centers on
            the trade-off between model weight precision and KV cache capacity. FP8 quantization
            halves weight memory compared to BF16, freeing VRAM for KV cache. This directly
            translates to higher concurrent request capacity — our benchmarks show Qwen3.5-35B
            in FP8 (35 GB weights) supports ~15 concurrent chat requests on a 96 GB RTX PRO
            6000, while BF16 (70 GB weights) is limited to ~3-4.

            However, the relationship is not linear. KV cache memory grows with both sequence
            length and batch size. At 16K tokens per request (typical for agentic workloads),
            even FP8 weights leave only enough KV cache for 1-2 concurrent requests. The
            bottleneck shifts from weight memory to KV cache memory as context length increases.

            4-bit quantization (GPTQ, AWQ) further reduces weight memory but introduces
            dequantization overhead. On architectures requiring --enforce-eager (no CUDA graphs),
            like Qwen3.5's hybrid Mamba+MoE, the 4-bit penalty is severe: 3.5-4x slower per
            token due to the combination of dequantization overhead and missing CUDA graph
            optimization. On architectures where CUDA graphs work (like Nemotron's NemotronH),
            AWQ 4-bit achieves near-FP8 per-token speed while freeing even more VRAM for KV
            cache.

            The emerging consensus is that the optimal quantization depends on the architecture:
            use FP8 for Mamba hybrids that require --enforce-eager, and AWQ 4-bit for models
            with full CUDA graph support. KV cache quantization (FP8 or Q8) provides additional
            headroom but is blocked by framework bugs for some architectures. Multi-token
            prediction (MTP) offers a different scaling axis — speculative decoding with 2-3
            predicted tokens per step can increase effective throughput 1.5-2x without changing
            memory allocation.

            Flash attention and its variants remain critical — reducing attention compute from
            O(n²) to O(n) in memory is what makes 256K+ contexts feasible at all. Combined with
            prefix caching (reusing KV cache for shared system prompts), the effective KV cache
            budget can be stretched further for workloads with common prefixes like RAG or
            agent frameworks where multiple requests share the same system prompt and tool
            definitions.
        """),
    },
    "agentic": {
        "label": "Agentic",
        "description": "Very long context, medium output (~3K tok in, ~500 tok out)",
        "max_tokens": 600,
        "prompt": textwrap.dedent("""\
            You are a coding assistant with access to the following tools:

            - read_file(path: str) -> str: Read a file from the filesystem
            - write_file(path: str, content: str) -> bool: Write content to a file
            - run_command(cmd: str) -> dict: Run a shell command, returns {stdout, stderr, exit_code}
            - search_code(pattern: str, path: str) -> list[dict]: Grep for pattern in files

            Previous conversation:
            User: Fix the failing test in test_rate_limiter.py
            Assistant: Let me read the test file first.
            [Tool call: read_file("tests/test_rate_limiter.py")]
            [Tool result: '''
            import pytest
            import asyncio
            from rate_limiter import AsyncRateLimiter

            @pytest.mark.asyncio
            async def test_basic_acquire():
                limiter = AsyncRateLimiter(rate=10, burst=5)
                for _ in range(5):
                    await limiter.acquire()

            @pytest.mark.asyncio
            async def test_rate_limiting():
                limiter = AsyncRateLimiter(rate=2, burst=2)
                start = asyncio.get_event_loop().time()
                for _ in range(4):
                    await limiter.acquire()
                elapsed = asyncio.get_event_loop().time() - start
                assert elapsed >= 0.9, f"Expected >= 0.9s, got {elapsed}"

            @pytest.mark.asyncio
            async def test_context_manager():
                limiter = AsyncRateLimiter(rate=10, burst=10)
                async with limiter:
                    pass  # should not raise

            @pytest.mark.asyncio
            async def test_burst_overflow():
                limiter = AsyncRateLimiter(rate=1, burst=3)
                # Fill burst
                for _ in range(3):
                    await limiter.acquire()
                # Next should wait ~1 second
                start = asyncio.get_event_loop().time()
                await limiter.acquire()
                elapsed = asyncio.get_event_loop().time() - start
                assert 0.8 < elapsed < 1.5, f"Expected ~1s wait, got {elapsed}"
            ''']
            Assistant: Now let me read the implementation.
            [Tool call: read_file("rate_limiter.py")]
            [Tool result: '''
            import asyncio
            import time

            class AsyncRateLimiter:
                def __init__(self, rate: float, burst: int):
                    self.rate = rate
                    self.burst = burst
                    self.tokens = burst
                    self.last_refill = time.monotonic()
                    self._lock = asyncio.Lock()

                async def acquire(self):
                    async with self._lock:
                        self._refill()
                        if self.tokens < 1:
                            wait = (1 - self.tokens) / self.rate
                            await asyncio.sleep(wait)
                            self._refill()
                        self.tokens -= 1

                def _refill(self):
                    now = time.monotonic()
                    elapsed = now - self.last_refill
                    self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
                    self.last_refill = now

                async def __aenter__(self):
                    await self.acquire()
                    return self

                async def __aexit__(self, *exc):
                    pass
            ''']
            Assistant: Let me run the tests to see which one fails.
            [Tool call: run_command("python -m pytest tests/test_rate_limiter.py -v")]
            [Tool result: {
                "stdout": "test_basic_acquire PASSED\\ntest_rate_limiting FAILED\\ntest_context_manager PASSED\\ntest_burst_overflow PASSED",
                "stderr": "AssertionError: Expected >= 0.9s, got 0.42",
                "exit_code": 1
            }]

            User: What's the fix?

            Analyze the failing test and the implementation. Explain the bug, propose a fix,
            and provide the corrected code. Use a tool call to write the fixed file.
        """),
    },
}


def benchmark_one(client: OpenAI, model: str, prompt: str, max_tokens: int,
                  no_think: bool = False) -> dict:
    """Run a single benchmark request, return metrics."""
    kwargs = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        stream_options={"include_usage": True},
        max_tokens=max_tokens,
    )
    if no_think:
        kwargs["extra_body"] = {"enable_thinking": False}

    t_start = time.perf_counter()
    t_first: float | None = None
    chunks: list[str] = []
    think_chunks: list[str] = []
    usage = None

    stream = client.chat.completions.create(**kwargs)
    for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None

        # Reasoning tokens
        extra = (getattr(delta, "model_extra", None) or {}) if delta else {}
        reasoning = extra.get("reasoning") or extra.get("reasoning_content")
        if reasoning:
            think_chunks.append(reasoning)

        # Content tokens
        content = delta.content if delta else None
        if content:
            if t_first is None:
                t_first = time.perf_counter()
            chunks.append(content)

        if hasattr(chunk, "usage") and chunk.usage is not None:
            usage = chunk.usage

    t_end = time.perf_counter()

    total_time = t_end - t_start
    ttft = (t_first - t_start) if t_first else total_time
    completion_tokens = usage.completion_tokens if usage else len("".join(chunks).split())
    prompt_tokens = usage.prompt_tokens if usage else 0
    tps = completion_tokens / total_time if total_time > 0 else 0.0

    return {
        "ttft_s": ttft,
        "total_s": total_time,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "think_chars": len("".join(think_chunks)),
        "tokens_per_sec": tps,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Multi-scenario benchmark — tests chat, RAG, code, summarization, and agentic workloads"
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--runs", type=int, default=3,
                        help="Runs per scenario (default: 3)")
    parser.add_argument("--scenario", nargs="+", default=list(SCENARIOS.keys()),
                        choices=list(SCENARIOS.keys()),
                        help="Scenarios to run (default: all)")
    parser.add_argument("--no-think", action="store_true", default=True,
                        help="Disable thinking/reasoning (default: true)")
    parser.add_argument("--think", action="store_true",
                        help="Enable thinking/reasoning")
    parser.add_argument("--warmup", action="store_true",
                        help="Send a warmup request before benchmarking")
    args = parser.parse_args()

    if args.think:
        args.no_think = False

    client = OpenAI(base_url=args.base_url, api_key="not-required")

    print(f"Endpoint  : {args.base_url}")
    print(f"Model     : {args.model}")
    print(f"Runs/scen : {args.runs}")
    print(f"Scenarios : {', '.join(args.scenario)}")
    print()

    if args.warmup:
        print("Warmup    : sending short request…", end="", flush=True)
        t0 = time.perf_counter()
        client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=1,
        )
        print(f" done ({time.perf_counter() - t0:.1f}s)")
        print()

    # Run all scenarios
    summary = []
    for name in args.scenario:
        sc = SCENARIOS[name]
        print(f"{'─' * 70}")
        print(f"  {sc['label']}  —  {sc['description']}")
        print(f"  max_tokens={sc['max_tokens']}")
        print(f"{'─' * 70}")

        runs = []
        for i in range(1, args.runs + 1):
            try:
                r = benchmark_one(client, args.model, sc["prompt"],
                                  sc["max_tokens"], no_think=args.no_think)
                runs.append(r)
                think_str = f"  think {r['think_chars']}ch" if r["think_chars"] else ""
                print(
                    f"  Run {i}/{args.runs}  "
                    f"TTFT {r['ttft_s']*1000:6.0f}ms  "
                    f"{r['prompt_tokens']:5d} in  "
                    f"{r['completion_tokens']:5d} out  "
                    f"{r['tokens_per_sec']:6.1f} tok/s"
                    f"{think_str}"
                )
            except Exception as exc:
                print(f"  Run {i}/{args.runs}  FAILED: {exc}")

        if runs:
            avg_tps = sum(r["tokens_per_sec"] for r in runs) / len(runs)
            avg_ttft = sum(r["ttft_s"] for r in runs) / len(runs)
            avg_in = sum(r["prompt_tokens"] for r in runs) / len(runs)
            avg_out = sum(r["completion_tokens"] for r in runs) / len(runs)
            print(
                f"  {'Average':>10}  "
                f"TTFT {avg_ttft*1000:6.0f}ms  "
                f"{avg_in:5.0f} in  "
                f"{avg_out:5.0f} out  "
                f"{avg_tps:6.1f} tok/s"
            )
            summary.append({
                "name": name,
                "label": sc["label"],
                "avg_tps": avg_tps,
                "avg_ttft_ms": avg_ttft * 1000,
                "avg_in": avg_in,
                "avg_out": avg_out,
                "runs": len(runs),
            })
        print()

    # Summary table
    if summary:
        print(f"{'═' * 70}")
        print(f"  SUMMARY — {args.model} @ {args.base_url}")
        print(f"{'═' * 70}")
        print(f"  {'Scenario':<16} {'Avg in':>7} {'Avg out':>8} {'TTFT':>8} {'tok/s':>8}")
        print(f"  {'─' * 52}")
        for s in summary:
            print(
                f"  {s['label']:<16} "
                f"{s['avg_in']:>6.0f}  "
                f"{s['avg_out']:>7.0f}  "
                f"{s['avg_ttft_ms']:>6.0f}ms "
                f"{s['avg_tps']:>7.1f}"
            )
        print(f"  {'─' * 52}")
        overall_tps = sum(s["avg_tps"] for s in summary) / len(summary)
        print(f"  {'Overall avg':<16} {'':>7} {'':>8} {'':>8} {overall_tps:>7.1f}")
        print()


if __name__ == "__main__":
    main()
