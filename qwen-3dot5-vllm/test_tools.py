#!/usr/bin/env python3
"""
Tool-calling test — verifies that vLLM correctly routes function calls.

Runs three scenarios:
  1. Single tool call  — get current weather
  2. Parallel tool calls — fetch weather for two cities simultaneously
  3. Multi-turn        — tool result fed back; model produces a final answer

Usage:
    python test_tools.py
    python test_tools.py --base-url http://localhost:8000/v1 --scenario single
"""

import argparse
import json
import sys
import textwrap
import time

try:
    from openai import OpenAI
except ImportError:
    sys.exit("openai package required: pip install openai")


DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL    = "qwen3.5-35b"

# ---------------------------------------------------------------------------
# Tool definitions (JSON Schema)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Return current weather conditions for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city":  {"type": "string",  "description": "City name, e.g. 'Tokyo'"},
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit (default: celsius)",
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
                    "expression": {
                        "type": "string",
                        "description": "A valid Python math expression, e.g. '(3 + 5) * 2'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Simulated tool execution
# ---------------------------------------------------------------------------

def execute_tool(name: str, args: dict) -> str:
    if name == "get_weather":
        city  = args.get("city", "Unknown")
        units = args.get("units", "celsius")
        temp  = {"Tokyo": 22, "London": 15, "New York": 18}.get(city, 20)
        if units == "fahrenheit":
            temp = temp * 9 // 5 + 32
        unit_sym = "°C" if units == "celsius" else "°F"
        return json.dumps({
            "city": city, "temperature": f"{temp}{unit_sym}",
            "condition": "partly cloudy", "humidity": "65%",
        })

    if name == "calculate":
        expr = args.get("expression", "")
        try:
            # Restrict to safe math operations
            result = eval(expr, {"__builtins__": {}}, {})  # noqa: S307
            return json.dumps({"expression": expr, "result": result})
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    return json.dumps({"error": f"Unknown tool: {name}"})


# ---------------------------------------------------------------------------
# Pretty printers
# ---------------------------------------------------------------------------

def print_tool_calls(tool_calls) -> None:
    for tc in tool_calls:
        fn   = tc.function
        args = json.loads(fn.arguments)
        print(f"  ┌ tool call id : {tc.id}")
        print(f"  │ function     : {fn.name}")
        print(f"  └ arguments    : {json.dumps(args, indent=4).replace(chr(10), chr(10) + '               ')}")


def header(text: str) -> None:
    bar = "─" * 60
    print(f"\n{bar}")
    print(f"  {text}")
    print(bar)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def scenario_single(client: OpenAI, model: str) -> bool:
    header("Scenario 1 — single tool call: get_weather")

    messages = [{"role": "user", "content": "What's the weather like in Tokyo right now?"}]

    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model, messages=messages, tools=TOOLS, tool_choice="auto"
    )
    elapsed = time.perf_counter() - t0

    msg = resp.choices[0].message
    print(f"Finish reason : {resp.choices[0].finish_reason}  ({elapsed*1000:.0f} ms)")

    if not msg.tool_calls:
        print("FAIL — no tool call returned")
        if msg.content:
            print(f"Model replied with text: {msg.content[:200]}")
        return False

    print_tool_calls(msg.tool_calls)

    # Execute and feed result back
    messages.append(msg)
    for tc in msg.tool_calls:
        args   = json.loads(tc.function.arguments)
        result = execute_tool(tc.function.name, args)
        print(f"\n  Tool result: {result}")
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    follow_up = client.chat.completions.create(model=model, messages=messages, tools=TOOLS)
    final = follow_up.choices[0].message.content or ""
    print(f"\nFinal answer:\n{textwrap.fill(final, width=72, initial_indent='  ', subsequent_indent='  ')}")
    return True


def scenario_parallel(client: OpenAI, model: str) -> bool:
    header("Scenario 2 — parallel tool calls: weather for two cities")

    messages = [
        {
            "role": "user",
            "content": "Compare the weather in London and New York. Get both at the same time.",
        }
    ]

    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model, messages=messages, tools=TOOLS, tool_choice="auto"
    )
    elapsed = time.perf_counter() - t0

    msg = resp.choices[0].message
    print(f"Finish reason : {resp.choices[0].finish_reason}  ({elapsed*1000:.0f} ms)")

    if not msg.tool_calls:
        print("FAIL — no tool calls returned")
        if msg.content:
            print(f"Model replied with text: {msg.content[:200]}")
        return False

    print(f"Tool calls    : {len(msg.tool_calls)}")
    print_tool_calls(msg.tool_calls)

    if len(msg.tool_calls) < 2:
        print("WARN — expected 2 parallel calls, got fewer")

    messages.append(msg)
    for tc in msg.tool_calls:
        args   = json.loads(tc.function.arguments)
        result = execute_tool(tc.function.name, args)
        print(f"\n  Tool result ({tc.function.name}): {result}")
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    follow_up = client.chat.completions.create(model=model, messages=messages, tools=TOOLS)
    final = follow_up.choices[0].message.content or ""
    print(f"\nFinal answer:\n{textwrap.fill(final, width=72, initial_indent='  ', subsequent_indent='  ')}")
    return True


def scenario_multi_tool(client: OpenAI, model: str) -> bool:
    header("Scenario 3 — chained tools: weather + calculate")

    messages = [
        {
            "role": "user",
            "content": (
                "What is the temperature in Tokyo in celsius? "
                "Then calculate what that is multiplied by 3.14."
            ),
        }
    ]

    # Allow up to 4 turns to resolve chained tool calls
    for turn in range(1, 5):
        resp   = client.chat.completions.create(model=model, messages=messages, tools=TOOLS)
        msg    = resp.choices[0].message
        reason = resp.choices[0].finish_reason
        print(f"Turn {turn}: finish_reason={reason}")

        if not msg.tool_calls:
            final = msg.content or ""
            print(f"\nFinal answer:\n{textwrap.fill(final, width=72, initial_indent='  ', subsequent_indent='  ')}")
            return True

        print_tool_calls(msg.tool_calls)
        messages.append(msg)
        for tc in msg.tool_calls:
            args   = json.loads(tc.function.arguments)
            result = execute_tool(tc.function.name, args)
            print(f"  Result: {result}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    print("FAIL — did not reach a final answer within 4 turns")
    return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

SCENARIO_MAP = {
    "single":    scenario_single,
    "parallel":  scenario_parallel,
    "chained":   scenario_multi_tool,
}


def main():
    parser = argparse.ArgumentParser(description="Tool-calling integration test")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model",    default=DEFAULT_MODEL)
    parser.add_argument(
        "--scenario",
        choices=["single", "parallel", "chained", "all"],
        default="all",
        help="Which scenario to run (default: all)",
    )
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="not-required")
    print(f"Endpoint : {args.base_url}")
    print(f"Model    : {args.model}")

    scenarios = (
        list(SCENARIO_MAP.items())
        if args.scenario == "all"
        else [(args.scenario, SCENARIO_MAP[args.scenario])]
    )

    passed = 0
    for name, fn in scenarios:
        try:
            ok = fn(client, args.model)
            passed += ok
        except Exception as exc:
            print(f"\nERROR in scenario '{name}': {exc}")

    total = len(scenarios)
    print(f"\n{'─'*60}")
    print(f"Results: {passed}/{total} passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
