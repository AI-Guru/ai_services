#!/usr/bin/env python3
"""
Qwen3-TTS end-to-end test + latency/RTF benchmark.

Hits the OpenAI-compatible speech endpoint (POST /v1/audio/speech) exposed by
vLLM-Omni, synthesizes a small built-in set of prompts, saves each result as a
listenable WAV under samples/, and prints per-clip latency, audio duration and
RTF (real-time factor = generation_time / audio_seconds; < 1.0 is faster than
real time).

Three suites, matching the three model variants:
  customvoice  — 9 preset voices + style `instructions`   (CustomVoice models)
  voicedesign  — invent a voice from a text description    (VoiceDesign model)
  clone        — zero-shot voice clone from --ref-audio    (Base model)

Usage:
    # CustomVoice server on 12437 (default)
    python test_tts.py
    python test_tts.py --suite customvoice --base-url http://localhost:12437/v1 \
        --model qwen3-tts-1.7b-customvoice

    # VoiceDesign server on 12438
    python test_tts.py --suite voicedesign --base-url http://localhost:12438/v1 \
        --model qwen3-tts-1.7b-voicedesign

    # Base voice clone on 12439 (needs a ~3-10 s reference clip)
    python test_tts.py --suite clone --base-url http://localhost:12439/v1 \
        --model qwen3-tts-1.7b-base \
        --ref-audio /path/to/ref.wav --ref-text "transcript of the reference"

Only the `requests` package is required (plus stdlib `wave`).
    uv pip install requests
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
import wave
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Missing dependency: requests. Install with: uv pip install requests")


SAMPLES_DIR = Path(__file__).parent / "samples"


# The 9 CustomVoice preset timbres. Each case: (label, voice, language, text).
CUSTOMVOICE_CASES = [
    ("en_ryan",    "ryan",     "English",    "The quick brown fox jumps over the lazy dog, clear as a bell."),
    ("de_serena",  "serena",   "German",     "Guten Morgen! Willkommen zu unserem selbst gehosteten Sprachdienst."),
    ("en_vivian",  "vivian",   "English",    "Self-hosted text to speech, running entirely on our own hardware."),
    ("zh_dylan",   "dylan",    "Chinese",    "欢迎使用通义千问语音合成模型，这是一段中文测试。"),
    ("ja_ono_anna","ono_anna", "Japanese",   "こんにちは、これは日本語の音声合成のテストです。"),
    ("fr_aiden",   "aiden",    "French",     "Bonjour, ceci est un test de synthèse vocale en français."),
]

# VoiceDesign: invent a voice from a natural-language description.
# Each case: (label, instructions, language, text).
VOICEDESIGN_CASES = [
    ("storyteller_de", "A calm, warm elderly male storyteller with a slow, gentle pace",
     "German",  "Es war einmal, vor langer Zeit, ein kleines Dorf am Rande des Waldes."),
    ("news_en",        "A crisp, professional female news anchor, neutral and articulate",
     "English", "Good evening. Tonight, our top story: local researchers self-host their own speech models."),
    ("excited_en",     "An energetic, upbeat young male voice, fast and enthusiastic",
     "English", "This is incredible — we just generated speech on our own GPU, no cloud required!"),
]

# Style-`instructions` variations on a single preset voice (CustomVoice suite
# addendum — demonstrates emotion control without changing the timbre).
STYLE_CASES = [
    ("style_whisper", "ryan",  "English", "whispering softly, almost a secret",
     "Come closer... I have something quiet to tell you."),
    ("style_angry",   "serena", "English", "angry and forceful",
     "I told you three times already — please listen to me!"),
]


def wav_duration_seconds(data: bytes) -> float | None:
    """Duration of a RIFF/WAV byte string, or None if not parseable as WAV."""
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
            return frames / rate if rate else None
    except (wave.Error, EOFError):
        return None


def synthesize(base_url: str, model: str, text: str, response_format: str = "wav",
               timeout: float = 300.0, **params) -> tuple[bytes, float]:
    """POST /v1/audio/speech and return (audio_bytes, latency_seconds).

    `params` may include: voice, language, instructions, task_type, speed,
    ref_audio, ref_text — passed straight through to the endpoint.
    """
    payload = {"model": model, "input": text, "response_format": response_format}
    payload.update({k: v for k, v in params.items() if v is not None})
    url = base_url.rstrip("/") + "/audio/speech"
    t0 = time.perf_counter()
    resp = requests.post(url, json=payload, timeout=timeout)
    latency = time.perf_counter() - t0
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:400]}")
    return resp.content, latency


def load_ref_audio(path_or_url: str) -> str:
    """Return a value usable as the `ref_audio` field: a URL as-is, otherwise a
    base64 data string of the local file."""
    if path_or_url.startswith(("http://", "https://", "file://", "data:")):
        return path_or_url
    raw = Path(path_or_url).read_bytes()
    return "data:audio/wav;base64," + base64.b64encode(raw).decode()


def build_cases(args) -> list[dict]:
    """Return the list of synthesis cases for the chosen suite."""
    cases: list[dict] = []
    if args.suite in ("customvoice", "all"):
        for label, voice, lang, text in CUSTOMVOICE_CASES:
            cases.append({"label": label, "text": text,
                          "params": {"voice": voice, "language": lang,
                                     "task_type": "CustomVoice"}})
        for label, voice, lang, instruct, text in STYLE_CASES:
            cases.append({"label": label, "text": text,
                          "params": {"voice": voice, "language": lang,
                                     "instructions": instruct,
                                     "task_type": "CustomVoice"}})
    if args.suite in ("voicedesign", "all"):
        for label, instruct, lang, text in VOICEDESIGN_CASES:
            cases.append({"label": label, "text": text,
                          "params": {"instructions": instruct, "language": lang,
                                     "task_type": "VoiceDesign"}})
    if args.suite in ("clone", "all"):
        if not args.ref_audio:
            if args.suite == "clone":
                sys.exit("--suite clone requires --ref-audio (and ideally --ref-text)")
        else:
            ref = load_ref_audio(args.ref_audio)
            for label, text in [
                ("clone_en", "This sentence is spoken in a cloned voice, generated locally."),
                ("clone_de", "Dieser Satz wird mit einer geklonten Stimme gesprochen."),
            ]:
                cases.append({"label": label, "text": text,
                              "params": {"ref_audio": ref, "ref_text": args.ref_text,
                                         "task_type": "Base"}})
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:12437/v1",
                        help="OpenAI-compatible base URL (…/v1)")
    parser.add_argument("--model", default="qwen3-tts-1.7b-customvoice",
                        help="served-model-name to send")
    parser.add_argument("--suite", default="customvoice",
                        choices=["customvoice", "voicedesign", "clone", "all"])
    parser.add_argument("--format", default="wav",
                        choices=["wav", "mp3", "flac", "pcm", "aac", "opus"],
                        help="response_format (wav recommended — enables RTF)")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--ref-audio", default=None,
                        help="clone suite: reference clip (path, URL, or data URI)")
    parser.add_argument("--ref-text", default=None,
                        help="clone suite: transcript of the reference clip")
    parser.add_argument("--out-dir", type=Path, default=SAMPLES_DIR)
    parser.add_argument("--warmup", action="store_true",
                        help="Send one un-measured request first (drops cold start)")
    parser.add_argument("--json", type=Path, default=None,
                        help="Optional path to write structured results JSON")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases = build_cases(args)
    if not cases:
        sys.exit("No cases to run for this suite.")

    print(f"Endpoint : {args.base_url}")
    print(f"Model    : {args.model}")
    print(f"Suite    : {args.suite}  ({len(cases)} clips)")
    print(f"Format   : {args.format}")
    print(f"Out dir  : {args.out_dir}")
    print()

    if args.warmup:
        print("Warmup   : synthesizing short clip…", end="", flush=True)
        try:
            wp = dict(cases[0]["params"])
            wp["speed"] = args.speed
            synthesize(args.base_url, args.model, "Warm up.", args.format, **wp)
            print(" done")
        except Exception as exc:
            print(f" FAILED: {exc}")
            return 1
        print()

    ext = args.format if args.format != "pcm" else "pcm"
    print(f"{'clip':<16}{'lat(s)':>9}{'audio(s)':>10}{'RTF':>8}  file")
    print("-" * 68)

    results = []
    any_failed = False
    for case in cases:
        params = dict(case["params"])
        params["speed"] = args.speed
        try:
            audio, latency = synthesize(args.base_url, args.model, case["text"],
                                        args.format, **params)
        except Exception as exc:
            print(f"{case['label']:<16}{'FAIL':>9}")
            print(f"    error: {exc}")
            any_failed = True
            continue

        out_path = args.out_dir / f"{args.suite}_{case['label']}.{ext}"
        out_path.write_bytes(audio)
        dur = wav_duration_seconds(audio) if args.format == "wav" else None
        rtf = (latency / dur) if dur else float("nan")
        dur_s = f"{dur:.2f}" if dur else "  -"
        rtf_s = f"{rtf:.3f}" if dur else "   -"
        print(f"{case['label']:<16}{latency:>9.2f}{dur_s:>10}{rtf_s:>8}  "
              f"{out_path.name} ({len(audio)//1024} KB)")
        results.append({
            "label": case["label"], "text": case["text"], "params": case["params"],
            "latency_s": latency, "audio_s": dur, "rtf": rtf,
            "bytes": len(audio), "file": out_path.name,
        })

    print("-" * 68)
    ok = [r for r in results if r["audio_s"]]
    if ok:
        mean_rtf = sum(r["rtf"] for r in ok) / len(ok)
        total_audio = sum(r["audio_s"] for r in ok)
        total_lat = sum(r["latency_s"] for r in ok)
        print(f"{len(ok)} clips · {total_audio:.1f}s audio in {total_lat:.1f}s · "
              f"mean RTF {mean_rtf:.3f}")

    # Write / update a manifest so the samples/ folder is self-describing.
    manifest = args.out_dir / "manifest.json"
    existing = json.loads(manifest.read_text()) if manifest.exists() else {}
    existing[f"{args.model}:{args.suite}"] = {
        "endpoint": args.base_url, "model": args.model, "suite": args.suite,
        "format": args.format, "results": results,
    }
    manifest.write_text(json.dumps(existing, ensure_ascii=False, indent=2))

    if args.json:
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"Wrote {args.json}")

    if any_failed:
        print("Status: FAIL (at least one clip errored)")
        return 1
    print("Status: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
