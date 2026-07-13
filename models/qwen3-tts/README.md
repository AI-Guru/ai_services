# Qwen3-TTS Family

Open text-to-speech models from Alibaba Qwen (Apache 2.0), derived from the
Qwen3-Omni foundation — the speech-*output* sibling of the `qwen3-asr` family.
All serve through **vLLM-Omni** over an **OpenAI-compatible `/v1/audio/speech`**
endpoint (the standard TTS API convention), so any OpenAI TTS client works
unmodified.

- **[Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)** — primary. 9 preset premium voices + style control via `instructions`.
- **[Qwen3-TTS-12Hz-1.7B-VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign)** — invent a new voice from a natural-language description.
- **[Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)** — zero-shot voice clone from ~3 s of reference audio.
- **[Qwen3-TTS-12Hz-0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)** — lightweight CustomVoice (0.6B versions of Base also exist).

Preset voices: **vivian, serena, uncle_fu, dylan, eric, ryan, aiden, ono_anna, sohee**.
Languages: Chinese, English, Japanese, Korean, German, French, Russian,
Portuguese, Spanish, Italian.

---

## Architecture — why serving differs from a normal LLM

Not a single forward pass. It's a **two-stage discrete multi-codebook LM**:

1. **Stage 0 — talker** (1.7B / 0.6B transformer) predicts 12 Hz audio codes.
2. **Stage 1 — code2wav** (lightweight non-DiT vocoder) decodes codes → waveform.

Both run as **separate vLLM engines on one GPU**, wired by a shared-memory
connector that streams codec chunks stage0→stage1 for low first-audio latency.
vLLM-Omni orchestrates the pipeline; the two-stage deploy config is what the
`--deploy-config` YAMLs in [`deploy/`](deploy/) describe.

## Quick start

```bash
# CustomVoice (primary) on port 12437
docker compose -f docker-compose.vllm-1.7b-customvoice-rtx.yml up -d

curl -s http://localhost:12437/v1/models | jq
curl -X POST http://localhost:12437/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hallo Welt","voice":"ryan","language":"German"}' \
  --output hello.wav
```

## Compose variants

| File | Model | Port | Served name | Deploy config |
|------|-------|------|-------------|---------------|
| `docker-compose.vllm-1.7b-customvoice-rtx.yml` | 1.7B CustomVoice | 12437 | `qwen3-tts-1.7b-customvoice` | 1.7b-conservative |
| `docker-compose.vllm-1.7b-voicedesign-rtx.yml` | 1.7B VoiceDesign | 12438 | `qwen3-tts-1.7b-voicedesign` | 1.7b-conservative |
| `docker-compose.vllm-1.7b-base-rtx.yml`        | 1.7B Base (clone)| 12439 | `qwen3-tts-1.7b-base`        | 1.7b-conservative |
| `docker-compose.vllm-0.6b-customvoice-rtx.yml` | 0.6B CustomVoice | 12440 | `qwen3-tts-0.6b-customvoice` | 0.6b-conservative |

Ports 12437–12440 sit just above the `qwen3-asr` band (12434–12436).

## Image & install

Uses the **official `vllm/vllm-omni:v0.22.0`** image (on Docker Hub, built on
the vLLM image; ships `qwen3_tts` model support). The image has **no default
entrypoint** — the compose `command:` supplies `vllm serve … --omni`. Newer tags
(`v0.24.0`, `latest`) exist; pin bumps go here. No git-source install needed.

## Conservative memory — built for parallel

The GPU is shared, so the mounted deploy configs pin **each stage low**:

| Config | `gpu_memory_utilization` / stage | ≈ total (both stages) | `max_num_seqs` | eager |
|--------|-------------------------------|-----------------------|----------------|-------|
| `deploy/qwen3_tts-1.7b-conservative.yaml` | 0.07 | ~10 GB measured (~13.4 GB budget) | 4 | yes |
| `deploy/qwen3_tts-0.6b-conservative.yaml` | 0.05 | ~7 GB | 4 | yes |

`gpu_memory_utilization` is a fraction of **total** card memory, and each stage
is a separate engine, so the two fractions add. `enforce_eager` skips cudagraph
capture to save the most memory (costs some decode speed, not quality). Verified:
**three TTS models + a 27B vision LLM resident simultaneously** at ~89 GB / 96 GB.
Running one model solo? Drop `enforce_eager` and raise the fractions in the YAML.

## Testing & samples — `test_tts.py`

Hits `/v1/audio/speech`, saves listenable WAVs to [`samples/`](samples/), reports
per-clip latency + RTF (generation_time / audio_seconds). Three suites:

```bash
python3 test_tts.py --suite customvoice --base-url http://localhost:12437/v1 --model qwen3-tts-1.7b-customvoice --warmup
python3 test_tts.py --suite voicedesign --base-url http://localhost:12438/v1 --model qwen3-tts-1.7b-voicedesign --warmup
python3 test_tts.py --suite clone       --base-url http://localhost:12439/v1 --model qwen3-tts-1.7b-base \
    --ref-audio samples/customvoice_en_ryan.wav \
    --ref-text "The quick brown fox jumps over the lazy dog, clear as a bell."
```

Measured (1.7B, conservative config, GPU shared with other models): mean
**RTF ≈ 0.14** (~7× real time), TTFA sub-second. See [`samples/README.md`](samples/README.md).

## API surface (vLLM-Omni)

`POST /v1/audio/speech` — OpenAI params `input`, `voice`, `response_format`
(wav/mp3/flac/pcm/aac/opus), `speed`; plus extensions `task_type`
(CustomVoice/VoiceDesign/Base), `language`, `instructions`, `ref_audio`/`ref_text`
(clone), `stream`/`stream_format` (SSE). Also `GET/POST/DELETE /v1/audio/voices`,
`POST /v1/audio/speech/batch`, `WS /v1/audio/speech/stream`.

## Notes

- **Community-quant policy (CLAUDE.md):** all checkpoints here are **first-party
  Qwen** BF16 weights — no third-party quants.
- Health: two-stage load takes ~100 s cold on this box; compose `start_period`
  is 300 s. `/health` goes green only when both stages are ready.
