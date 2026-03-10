#!/usr/bin/env python3
"""Test how many parallel tool calls the model can emit in a single response."""

import json
import sys
import time

try:
    from openai import OpenAI
except ImportError:
    sys.exit("openai package required: pip install openai")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Return current weather conditions for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city":  {"type": "string",  "description": "City name"},
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression and return the result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                },
                "required": ["expression"],
            },
        },
    },
]

CITIES = [
    "Tokyo", "London", "New York", "Paris", "Berlin", "Sydney", "Mumbai",
    "Cairo", "Toronto", "Seoul", "Bangkok", "Istanbul", "Moscow", "Dubai",
    "Singapore", "Rome", "Madrid", "Vienna", "Oslo", "Lima", "Nairobi",
    "Jakarta", "Beijing", "Athens", "Lisbon", "Prague", "Warsaw", "Dublin",
    "Helsinki", "Stockholm", "Copenhagen", "Zurich",
]


def build_prompt(n: int) -> str:
    # Mix weather lookups and calculations
    weather_count = (n * 2) // 3  # ~2/3 weather
    calc_count = n - weather_count
    if weather_count == 0:
        weather_count = 1
        calc_count = n - 1

    cities = CITIES[:weather_count]
    calcs = [f"{i+1} * {i+7}" for i in range(calc_count)]

    parts = []
    if cities:
        parts.append(f"get the weather in {', '.join(cities)}")
    if calcs:
        parts.append(f"calculate these expressions: {', '.join(calcs)}")

    return (
        f"I need exactly {n} things done simultaneously in one batch. "
        f"{' AND '.join(parts)}. "
        f"Call all {n} tools at once in a single response. Do NOT chain them."
    )


def test_n(client: OpenAI, n: int) -> dict:
    prompt = build_prompt(n)
    print(f"\n{'─'*60}")
    print(f"  Requesting {n} parallel tool calls")
    print(f"{'─'*60}")

    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model="qwen3.5-35b",
        messages=[{"role": "user", "content": prompt}],
        tools=TOOLS,
        tool_choice="auto",
    )
    elapsed = time.perf_counter() - t0

    msg = resp.choices[0].message
    got = len(msg.tool_calls) if msg.tool_calls else 0
    fns = {}
    if msg.tool_calls:
        for tc in msg.tool_calls:
            fns[tc.function.name] = fns.get(tc.function.name, 0) + 1

    status = "PASS" if got >= n else ("PARTIAL" if got > 0 else "FAIL")
    print(f"  Requested : {n}")
    print(f"  Got       : {got}  ({', '.join(f'{k}={v}' for k, v in fns.items())})")
    print(f"  Latency   : {elapsed*1000:.0f} ms")
    print(f"  Status    : {status}")

    if msg.content and not msg.tool_calls:
        print(f"  Text reply: {msg.content[:200]}")

    return {"requested": n, "got": got, "latency_ms": elapsed * 1000, "status": status}


def main():
    client = OpenAI(base_url="http://localhost:11435/v1", api_key="not-required")
    targets = [2, 4, 8, 16, 32]
    results = []

    for n in targets:
        try:
            r = test_n(client, n)
            results.append(r)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            results.append({"requested": n, "got": 0, "latency_ms": 0, "status": "ERROR"})

    print(f"\n{'═'*60}")
    print(f"  Summary")
    print(f"{'═'*60}")
    print(f"  {'Requested':>10}  {'Got':>5}  {'Latency':>10}  {'Status'}")
    print(f"  {'─'*10}  {'─'*5}  {'─'*10}  {'─'*8}")
    for r in results:
        print(f"  {r['requested']:>10}  {r['got']:>5}  {r['latency_ms']:>8.0f}ms  {r['status']}")

    max_pass = max((r["got"] for r in results if r["status"] in ("PASS", "PARTIAL")), default=0)
    print(f"\n  Max parallel tool calls achieved: {max_pass}")


if __name__ == "__main__":
    main()
