#!/usr/bin/env python3
"""GSM8K accuracy against a chat endpoint.

The image's own `sglang.test.run_eval --eval-name gsm8k` targets /v1/completions
and returns 0.000 with "All retry attempts exhausted" against these composes,
so this hits /v1/chat/completions instead. Thinking is OFF and temperature 0:
the question is whether the KV / GDN-state precision damages arithmetic, not how
well the model reasons when allowed to think.
"""
import argparse, json, re, urllib.request
from concurrent.futures import ThreadPoolExecutor

def ask(url, model, q, timeout):
    body = {"model": model,
            "messages": [{"role": "user", "content":
                q + "\n\nSolve step by step, then give the final numeric answer on its own last line as: #### <number>"}],
            "temperature": 0, "max_tokens": 1024,
            "chat_template_kwargs": {"enable_thinking": False}}
    r = urllib.request.Request(url, data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    for _ in range(3):
        try:
            d = json.load(urllib.request.urlopen(r, timeout=timeout))
            return d["choices"][0]["message"].get("content") or ""
        except Exception:
            continue
    return ""

def num(s):
    m = re.findall(r'-?\d[\d,]*\.?\d*', s.replace('$', ''))
    return m[-1].replace(',', '').rstrip('.') if m else None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:11485/v1")
    p.add_argument("--model", default="qwen3.8-27b")
    p.add_argument("--parquet", required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--label", default="")
    a = p.parse_args()

    import pandas as pd
    df = pd.read_parquet(a.parquet).head(a.n)
    url = a.base_url.rstrip('/') + "/chat/completions"

    def one(row):
        out = ask(url, a.model, row.question, a.timeout)
        gold = num(row.answer.split('####')[-1])
        got = num(out.split('####')[-1]) if '####' in out else num(out)
        return gold is not None and got == gold

    with ThreadPoolExecutor(a.concurrency) as ex:
        hits = list(ex.map(one, (r for _, r in df.iterrows())))
    ok = sum(hits)
    print(f"GSM8K {a.label}: {ok}/{len(hits)} = {100.0*ok/len(hits):.1f}%")

main()
