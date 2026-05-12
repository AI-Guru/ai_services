#!/usr/bin/env python3
"""
Qwen3-ASR end-to-end test + WER smoke check.

Hits the OpenAI-compatible transcription endpoint with a small built-in clip
set (5 English + 5 German samples from google/fleurs, streamed once on first
run and cached locally), then prints per-clip transcripts plus per-clip and
aggregate WER.

Usage:
    python test_asr.py
    python test_asr.py --endpoint http://localhost:12434/v1 --model qwen3-asr-1.7b
    python test_asr.py --endpoint http://localhost:12435/v1 --model qwen3-asr-0.6b
    python test_asr.py --langs en de fr es --n 3   # custom mix

FLEURS audio is 16 kHz mono — vLLM's transcription endpoint accepts it directly.
WER threshold is loose (0.25 default) — a real model targets <0.10 on FLEURS.
"""

from __future__ import annotations

import argparse
import json
import string
import sys
import time
from pathlib import Path

try:
    from openai import OpenAI
    from jiwer import wer as jiwer_wer
except ImportError as exc:
    sys.exit(
        f"Missing dependency: {exc.name}. "
        "Install with: uv pip install openai jiwer datasets soundfile"
    )


SAMPLES_DIR = Path(__file__).parent / "samples"
MANIFEST = SAMPLES_DIR / "manifest.json"


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace. CJK chars are
    space-separated so jiwer's whitespace tokenization computes character-level
    edit distance on Chinese/Japanese/Korean."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation + "，。！？、；：«»“”„"))
    out = []
    for ch in text:
        if "一" <= ch <= "鿿":
            out.append(" ")
            out.append(ch)
            out.append(" ")
        else:
            out.append(ch)
    return " ".join("".join(out).split())


# FLEURS language code → (config name, display name)
FLEURS_LANGS = {
    "en": ("en_us", "English"),
    "de": ("de_de", "German"),
    "fr": ("fr_fr", "French"),
    "es": ("es_419", "Spanish"),
    "it": ("it_it", "Italian"),
    "pt": ("pt_br", "Portuguese"),
    "nl": ("nl_nl", "Dutch"),
    "ja": ("ja_jp", "Japanese"),
    "ko": ("ko_kr", "Korean"),
    "zh": ("cmn_hans_cn", "Chinese"),
}


def build_manifest(langs: list[str], n_per_lang: int) -> list[dict]:
    """Stream FLEURS test split, take first n samples per language, save as
    16 kHz mono WAV. Returns a manifest list (also written to manifest.json)."""
    try:
        from datasets import load_dataset
        import soundfile as sf
    except ImportError as exc:
        sys.exit(f"Missing dependency: {exc.name}. uv pip install datasets soundfile")

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for lang in langs:
        if lang not in FLEURS_LANGS:
            print(f"  skipping unknown lang code: {lang}")
            continue
        config, display = FLEURS_LANGS[lang]
        print(f"  streaming {n_per_lang}x google/fleurs[{config}] (test) …", flush=True)
        ds = load_dataset("google/fleurs", config, split="test",
                          streaming=True, trust_remote_code=True)
        for i, ex in enumerate(ds):
            if i >= n_per_lang:
                break
            audio = ex["audio"]
            sr = audio["sampling_rate"]
            arr = audio["array"]
            ref = ex["transcription"]
            filename = f"fleurs_{lang}_{i:02d}.wav"
            path = SAMPLES_DIR / filename
            sf.write(str(path), arr, sr, subtype="PCM_16")
            manifest.append({
                "filename": filename,
                "lang": lang,
                "lang_display": display,
                "reference": ref,
                "samplerate": sr,
                "duration_s": len(arr) / sr,
            })
        print(f"    saved {n_per_lang} {display} clips")

    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def load_manifest(langs: list[str], n_per_lang: int) -> list[dict]:
    """Return a usable manifest, building it if missing or if the requested
    (langs, n) doesn't match cached contents."""
    if MANIFEST.exists():
        cached = json.loads(MANIFEST.read_text())
        have = {(c["lang"], c["filename"]) for c in cached}
        need = {(lang, f"fleurs_{lang}_{i:02d}.wav")
                for lang in langs for i in range(n_per_lang)}
        if need.issubset(have):
            return [c for c in cached if c["lang"] in langs
                    and int(c["filename"].split("_")[2].split(".")[0]) < n_per_lang]
        print("  cached manifest doesn't cover requested langs/n; rebuilding…")
    return build_manifest(langs, n_per_lang)


def transcribe(client: OpenAI, model: str, path: Path) -> tuple[str, float]:
    t0 = time.perf_counter()
    with path.open("rb") as fh:
        result = client.audio.transcriptions.create(model=model, file=fh)
    return result.text, time.perf_counter() - t0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://localhost:12434/v1",
                        help="Base URL of the OpenAI-compatible server")
    parser.add_argument("--model", default="qwen3-asr-1.7b",
                        help="served-model-name to send")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--langs", nargs="+", default=["en", "de"],
                        help=f"FLEURS language codes ({', '.join(FLEURS_LANGS)})")
    parser.add_argument("--n", type=int, default=5,
                        help="Number of clips per language (default 5)")
    parser.add_argument("--warmup", action="store_true",
                        help="Send a warmup request before measuring (drops cold-start)")
    parser.add_argument("--wer-threshold", type=float, default=0.25,
                        help="Fail run if aggregate WER exceeds this")
    parser.add_argument("--json", type=Path, default=None,
                        help="Optional path to write structured results JSON")
    args = parser.parse_args()

    print(f"Endpoint  : {args.endpoint}")
    print(f"Model     : {args.model}")
    print(f"Samples   : {SAMPLES_DIR}")
    print(f"Languages : {args.langs} ({args.n} per lang)")
    print()
    print("Preparing sample clips…")
    manifest = load_manifest(args.langs, args.n)
    if not manifest:
        sys.exit("No clips available — manifest is empty.")
    print()

    client = OpenAI(base_url=args.endpoint, api_key=args.api_key)

    if args.warmup:
        warmup_clip = manifest[0]
        print(f"Warmup    : transcribing {warmup_clip['filename']} (not measured)…",
              end="", flush=True)
        try:
            _, t = transcribe(client, args.model, SAMPLES_DIR / warmup_clip["filename"])
            print(f" done ({t:.1f}s)")
        except Exception as exc:
            print(f" FAILED: {exc}")
            return 1
        print()

    print(f"{'clip':<22}{'lang':<11}{'dur(s)':>7}{'lat(s)':>9}{'RTF':>8}{'WER':>8}")
    print("-" * 70)

    refs_norm: list[str] = []
    hyps_norm: list[str] = []
    results: list[dict] = []
    any_failed = False

    for clip in manifest:
        path = SAMPLES_DIR / clip["filename"]
        dur = clip["duration_s"]
        try:
            hyp, latency = transcribe(client, args.model, path)
        except Exception as exc:
            print(f"{clip['filename']:<22}{clip['lang_display']:<11}"
                  f"{dur:>7.2f}{'FAIL':>9}{'-':>8}{'-':>8}")
            print(f"    error: {exc}")
            any_failed = True
            continue

        ref_n = normalize(clip["reference"])
        hyp_n = normalize(hyp)
        clip_wer = jiwer_wer(ref_n, hyp_n) if ref_n else 0.0
        rtf = latency / dur if dur > 0 else float("nan")

        print(f"{clip['filename']:<22}{clip['lang_display']:<11}"
              f"{dur:>7.2f}{latency:>9.2f}{rtf:>8.3f}{clip_wer:>8.3f}")
        if clip_wer > 0:
            print(f"    ref: {clip['reference']}")
            print(f"    hyp: {hyp}")

        refs_norm.append(ref_n)
        hyps_norm.append(hyp_n)
        results.append({
            **clip,
            "hypothesis": hyp,
            "wer": clip_wer,
            "latency_s": latency,
            "rtf": rtf,
        })

    print("-" * 70)

    if not refs_norm:
        print("No clips transcribed successfully.")
        return 1

    # Aggregate WER per language (jiwer expects parallel lists)
    by_lang: dict[str, list[int]] = {}
    for i, r in enumerate(results):
        by_lang.setdefault(r["lang"], []).append(i)
    for lang, idxs in by_lang.items():
        lang_wer = jiwer_wer([refs_norm[i] for i in idxs],
                             [hyps_norm[i] for i in idxs])
        display = FLEURS_LANGS.get(lang, (lang, lang))[1]
        avg_rtf = sum(results[i]["rtf"] for i in idxs) / len(idxs)
        print(f"  {display:<10} ({len(idxs)} clips): WER {lang_wer:.3f}, "
              f"mean RTF {avg_rtf:.3f}")

    aggregate = jiwer_wer(refs_norm, hyps_norm)
    print(f"Aggregate ({len(refs_norm)} clips): WER {aggregate:.3f}")

    if args.json:
        args.json.write_text(json.dumps({
            "endpoint": args.endpoint,
            "model": args.model,
            "aggregate_wer": aggregate,
            "by_lang_wer": {lang: jiwer_wer([refs_norm[i] for i in idxs],
                                            [hyps_norm[i] for i in idxs])
                            for lang, idxs in by_lang.items()},
            "results": results,
        }, ensure_ascii=False, indent=2))
        print(f"Wrote {args.json}")

    if any_failed:
        print("Status: at least one clip failed to transcribe.")
        return 1
    if aggregate > args.wer_threshold:
        print(f"Status: FAIL (aggregate WER {aggregate:.3f} > threshold {args.wer_threshold})")
        return 1
    print(f"Status: OK (aggregate WER {aggregate:.3f} ≤ threshold {args.wer_threshold})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
