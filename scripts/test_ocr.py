#!/usr/bin/env python3
"""
OCR a single image against an OpenAI-compatible endpoint and print the transcript.

Sends the image as a base64 data: URL in an image_url message, so it works for
any vision model (Qwen3.6-27B, GLM-OCR, DeepSeek-OCR, dots.ocr, ...) served by
vLLM or SGLang. Non-streaming; reports wall-clock latency and token usage.

Usage:
  # baseline — Qwen3.6-27B already resident on :11436, served as qwen3.6-27b
  python3 test_ocr.py --base-url http://localhost:11436/v1 --model qwen3.6-27b

  # an OCR specialist with its own preferred prompt
  python3 test_ocr.py --base-url http://localhost:11438/v1 --model glm-ocr
  python3 test_ocr.py --base-url http://localhost:8000/v1  --model deepseek-ai/DeepSeek-OCR \
      --prompt '<image>\nFree OCR.'

Defaults to scripts/ocr-text.png. No API key needed for local servers.
"""
import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.request

DEFAULT_PROMPT = (
    "Transcribe ALL text in this image exactly as written, preserving line "
    "breaks and reading order. This is handwritten German. Do not translate, "
    "summarize, or add commentary — output only the transcription."
)


def data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="e.g. http://localhost:11436/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", default=os.path.join(here, "ocr-text.png"))
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    if not os.path.isfile(args.image):
        print(f"image not found: {args.image}", file=sys.stderr)
        return 2

    body = json.dumps({
        "model": args.model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url(args.image)}},
                {"type": "text", "text": args.prompt},
            ],
        }],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }).encode()

    url = args.base_url.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    })

    print(f"→ {args.model}  ({args.base_url})", file=sys.stderr)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as r:
            resp = json.load(r)
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode()[:800]}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"request failed: {e}", file=sys.stderr)
        return 1
    dt = time.time() - t0

    text = resp["choices"][0]["message"]["content"]
    usage = resp.get("usage", {})
    ct = usage.get("completion_tokens")
    tps = (ct / dt) if (ct and dt) else None

    print("=" * 70)
    print(text.strip())
    print("=" * 70)
    meta = f"latency {dt:.1f}s"
    if usage:
        meta += (f" | prompt {usage.get('prompt_tokens','?')} tok"
                 f" | completion {ct} tok")
        if tps:
            meta += f" | {tps:.1f} tok/s"
    print(meta, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
