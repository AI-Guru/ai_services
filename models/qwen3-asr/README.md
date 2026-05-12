# Qwen3-ASR Family

End-to-end speech-to-text models from Alibaba Qwen (Apache 2.0), derived from the Qwen3-Omni foundation. All three serve via vLLM through OpenAI-compatible endpoints.

- **[Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)** — primary STT model. State-of-the-art WER on English and Chinese; vendor-reported 1.63 on LibriSpeech-clean (beats Whisper-large-v3's 1.51 on clean / 3.97 on other with 3.38 on other).
- **[Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)** — lightweight sibling, ~2000× real-time at C=128. Trades ~0.5 WER points for huge throughput.
- **[Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)** — companion for word/char-level timestamp prediction. 11 languages, up to ~5 min per call, ~32 ms average alignment error.

All three support 52 languages: 30 base languages (en, zh, yue, ar, de, fr, es, pt, id, it, ko, ru, th, vi, ja, tr, hi, ms, nl, sv, da, fi, pl, cs, fil, fa, el, hu, mk, ro) plus 22 Chinese dialects.

---

# Qwen3-ASR-1.7B

[Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) — primary speech-to-text model.

## Architecture

- **Type**: end-to-end ASR (audio encoder + transformer decoder), single forward pass per clip
- **Backbone**: Qwen3-Omni
- **Parameters**: ~1.7 B (BF16 weights ≈ 3.4 GB)
- **Audio input**: 16 kHz mono; URL / local path / base64 / numpy array
- **Sample rate**: 16 kHz (resample upstream if your source is 44.1/48 kHz)
- **Max audio**: no hard limit for ASR (streaming + offline); aligner caps at ~5 min
- **License**: Apache 2.0

## Quick start

```bash
# RTX PRO 6000 (96 GB) — primary config
docker compose -f docker-compose.vllm-1.7b-bf16-rtx.yml up -d

# Any other CUDA GPU (≥16 GB VRAM)
docker compose -f docker-compose.vllm-1.7b-bf16.yaml up -d

# Wait for healthcheck, then transcribe a clip
curl -s http://localhost:12434/v1/models | jq

curl -s http://localhost:12434/v1/audio/transcriptions \
  -F model=qwen3-asr-1.7b \
  -F file=@samples/asr_en.wav
```

API endpoint: `http://localhost:12434/v1`, model name: `qwen3-asr-1.7b`

## Compose variants

| File | Engine | Model | Hardware | Notes |
|------|--------|-------|----------|-------|
| `docker-compose.vllm-1.7b-bf16-rtx.yml` | vLLM nightly | Qwen3-ASR-1.7B | RTX PRO 6000 | Primary config, util 0.40 (small model, lots of headroom) |
| `docker-compose.vllm-1.7b-bf16.yaml` | vLLM nightly | Qwen3-ASR-1.7B | Any | Generic GPU, util 0.85 |
| `docker-compose.vllm-0.6b-bf16-rtx.yml` | vLLM nightly | Qwen3-ASR-0.6B | RTX PRO 6000 | Lightweight, batch 128 |
| `docker-compose.vllm-0.6b-bf16.yaml` | vLLM nightly | Qwen3-ASR-0.6B | Any | Lightweight, util 0.85 |
| `docker-compose.vllm-aligner-0.6b-rtx.yml` | vLLM nightly | Qwen3-ForcedAligner-0.6B | RTX PRO 6000 | Word/char-level timestamps |
| `docker-compose.vllm-aligner-0.6b.yaml` | vLLM nightly | Qwen3-ForcedAligner-0.6B | Any | Word/char-level timestamps |

Ports: 1.7B → 12434, 0.6B → 12435, aligner → 12436 (sit above the 11430–11475 band already used by qwen3.5 / qwen3.6 / qwen3-coder-next / glm-4.7-flash / qwen3guard).

## Endpoints exposed by vLLM

| Path | Use |
|------|-----|
| `POST /v1/audio/transcriptions` | Whisper-style file upload. Returns `{"text": "..."}`. Use this from `openai.audio.transcriptions.create(...)`. |
| `POST /v1/chat/completions` | Send `audio_url` content type (URL or base64 data URI). Returns chat message with transcription in `content`. Supports context-prompt biasing. |
| `GET /v1/models` | List served models. |
| `GET /health` | Server readiness. |

## API usage

All three models speak the OpenAI-compatible API. The default served model name and port differ per variant — pick the right `--endpoint`/`--model` pair for the container you started:

| Variant | Base URL | `model` param |
|---------|---|---|
| Qwen3-ASR-1.7B | `http://localhost:12434/v1` | `qwen3-asr-1.7b` |
| Qwen3-ASR-0.6B | `http://localhost:12435/v1` | `qwen3-asr-0.6b` |
| Qwen3-ForcedAligner-0.6B | `http://localhost:12436/v1` | `qwen3-aligner-0.6b` (currently broken, see Known issues) |

**Audio format**: 16 kHz mono WAV (or formats `librosa` can decode to that). Non-conforming audio is rejected with `Invalid or unsupported audio file`. To convert: `ffmpeg -i input.mp3 -ar 16000 -ac 1 -sample_fmt s16 output.wav`.

### Transcription (Whisper-style file upload)

```bash
curl -s http://localhost:12434/v1/audio/transcriptions \
  -F model=qwen3-asr-1.7b \
  -F file=@audio.wav
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:12434/v1", api_key="EMPTY")
with open("audio.wav", "rb") as f:
    result = client.audio.transcriptions.create(model="qwen3-asr-1.7b", file=f)
print(result.text)
```

Response: `{"text": "..."}`.

### Translation (transcribe + translate to English)

```bash
curl -s http://localhost:12434/v1/audio/translations \
  -F model=qwen3-asr-1.7b \
  -F file=@spanish_clip.wav
```

```python
result = client.audio.translations.create(model="qwen3-asr-1.7b", file=open("spanish_clip.wav", "rb"))
```

### Chat completions (with context-prompt biasing)

Use this when you want to **bias decoding toward domain terms** (proper nouns, jargon, internal acronyms). Pass an `audio_url` content part alongside a text part:

```python
r = client.chat.completions.create(
    model="qwen3-asr-1.7b",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Context: Kubernetes cluster outage call. "
                                     "Expected terms: kubelet, CNI, CoreDNS, etcd, ingress-nginx."},
            {"type": "audio_url", "audio_url": {"url": "file:///path/to/standup.wav"}},
        ],
    }],
)
print(r.choices[0].message.content)
```

The `audio_url` accepts:
- a `file://` path on the server
- an `http(s)://` URL the server can reach
- a `data:audio/wav;base64,...` data URI

### Discovery

```bash
curl -s http://localhost:12434/v1/models | jq
curl -s http://localhost:12434/health
```

`/v1/models` returns the served model list with the exact `id` to pass as the `model` param.

## Vendor-reported WER benchmarks

From the [HuggingFace model card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B); **not** reproduced locally.

### English (WER ↓)

| Dataset | Qwen3-ASR-1.7B | Qwen3-ASR-0.6B | Whisper-large-v3 |
|---------|---------------:|---------------:|-----------------:|
| LibriSpeech-clean | **1.63** | 2.11 | 1.51 |
| LibriSpeech-other | **3.38** | 4.55 | 3.97 |
| GigaSpeech | **8.45** | 8.88 | 9.76 |
| CommonVoice-en | **7.39** | 9.92 | 9.90 |
| FLEURS-en | 3.35 | 4.39 | 4.08 |

### Chinese (WER ↓)

| Dataset | Qwen3-ASR-1.7B | Qwen3-ASR-0.6B |
|---------|---------------:|---------------:|
| WenetSpeech-net | **4.97** | 5.97 |
| WenetSpeech-meeting | **5.88** | 6.88 |
| AISHELL-2 | **2.71** | 3.15 |
| SpeechIO | **2.88** | 3.44 |

### Multilingual (avg WER, lower better)

| Suite | 1.7B |
|-------|-----:|
| FLEURS (12 langs) | 4.90 |
| CommonVoice (13 langs) | 9.18 |
| MLS (8 langs) | 8.55 |

### Forced aligner (avg absolute alignment error in ms, lower better)

| Lang | Qwen3-ForcedAligner-0.6B |
|------|-------------------------:|
| Average (human-labeled) | 32.4 ms |
| Chinese | 27.8 ms |
| English | 37.5 ms |

## Smoke test + WER measurement

`test_asr.py` streams a small sample set from **[google/fleurs](https://huggingface.co/datasets/google/fleurs)** (5 English + 5 German clips by default, cached locally on first run), hits the OpenAI-compatible `/v1/audio/transcriptions` endpoint, and prints per-clip transcripts + per-clip and per-language WER + aggregate WER. The script handles CJK character-level tokenization, normalizes punctuation, and writes a structured JSON if `--json` is passed.

```bash
# 1.7B (default endpoint)
uv run python test_asr.py --warmup --json benchmarks/run-1.7b.json

# 0.6B
uv run python test_asr.py --endpoint http://localhost:12435/v1 --model qwen3-asr-0.6b --warmup

# Custom mix
uv run python test_asr.py --langs en de fr es ja --n 3 --warmup

# Tighter threshold
uv run python test_asr.py --wer-threshold 0.10
```

Dependencies (added to repo `pyproject.toml`): `openai`, `httpx`, `jiwer`, plus `datasets<3.0` + `soundfile` + `librosa` for FLEURS streaming. Install with `uv sync` from the repo root.

## Measured benchmarks (RTX PRO 6000 Blackwell, 96 GB)

Tested **2026-05-12** with `test_asr.py --langs en de --n 5 --warmup` against vLLM nightly (`vllm/vllm-openai:nightly`) on the same RTX PRO 6000 used by the qwen3.6 directory. Both ASR models run **alongside** the qwen3.6-27b container — coexistence is the whole point of the tiny `--gpu-memory-utilization` setting.

### Per-model results

| Model | Clips | Agg WER | EN WER | DE WER | Mean RTF | Mean latency |
|-------|------:|--------:|-------:|-------:|---------:|-------------:|
| **Qwen3-ASR-1.7B** | 10 | **0.032** | 0.039 | 0.026 | 0.034 | 0.36 s |
| **Qwen3-ASR-0.6B** | 10 | 0.046 | 0.049 | 0.044 | 0.031 | 0.29 s |

- **WER**: 1.7B wins by ~30 % relative — matters most on hard utterances (proper nouns, code-switching, low-SNR). On clean FLEURS material both models are well under 5 % WER.
- **RTF** (real-time factor = latency ÷ audio duration): both models run **~30–35× faster than real time** on a single request. 0.6B is ~10 % faster per clip.
- **Latency**: 1.7B adds ~70 ms per clip vs. 0.6B. Negligible for batch, may matter for streaming UX.
- **Most "errors"**: punctuation/capitalization differences against FLEURS' lowercase-no-punct reference (e.g., `years` vs `year`, model adds `,`/`.`). True ASR errors mostly occur on proper nouns (`Sintra` vs `sintra` is normalized away; `sie` (the polite pronoun) misread as `say` in one EN clip).

### Vendor-reported WER for context

The vendor's full-FLEURS numbers — much larger eval set than ours (5 clips/lang here):

| Suite | 1.7B | 0.6B | Whisper-large-v3 |
|-------|-----:|-----:|-----------------:|
| FLEURS-en | 3.35 % | 4.39 % | 4.08 % |
| FLEURS avg (12 langs) | 4.90 % | — | — |

### Vendor-reported WER for context (English fixed datasets)

| Dataset | 1.7B | 0.6B | Whisper-large-v3 |
|---------|-----:|-----:|-----------------:|
| LibriSpeech-clean | **1.63** | 2.11 | 1.51 |
| LibriSpeech-other | **3.38** | 4.55 | 3.97 |
| GigaSpeech | **8.45** | 8.88 | 9.76 |
| CommonVoice-en | **7.39** | 9.92 | 9.90 |

### Raw runs (this hardware, this session)

| Clip (FLEURS test) | Duration | 1.7B latency / RTF / WER | 0.6B latency / RTF / WER |
|---|---:|---:|---:|
| fleurs_en_00.wav | 10.56 s | 0.24 s / 0.023 / 0.053 | 0.20 s / 0.019 / 0.053 |
| fleurs_en_01.wav | 8.76 s | 0.32 s / 0.036 / 0.095 | 0.26 s / 0.030 / 0.095 |
| fleurs_en_02.wav | 11.46 s | 0.43 s / 0.037 / 0.000 | 0.37 s / 0.032 / 0.030 |
| fleurs_en_03.wav | 5.76 s | 0.25 s / 0.044 / 0.067 | 0.21 s / 0.036 / 0.067 |
| fleurs_en_04.wav | 4.32 s | 0.24 s / 0.055 / 0.000 | 0.20 s / 0.045 / 0.000 |
| fleurs_de_00.wav | 17.94 s | 0.35 s / 0.019 / 0.000 | 0.30 s / 0.017 / 0.000 |
| fleurs_de_01.wav | 11.16 s | 0.34 s / 0.031 / 0.095 | 0.29 s / 0.026 / 0.095 |
| fleurs_de_02.wav | 12.42 s | 0.47 s / 0.038 / 0.043 | 0.40 s / 0.032 / 0.087 |
| fleurs_de_03.wav | 10.68 s | 0.51 s / 0.048 / 0.000 | 0.42 s / 0.039 / 0.000 |
| fleurs_de_04.wav | 9.90 s | 0.36 s / 0.036 / 0.000 | 0.30 s / 0.030 / 0.048 |

Full JSON results: [`benchmarks/fleurs-en5-de5-1.7b.json`](benchmarks/fleurs-en5-de5-1.7b.json), [`benchmarks/fleurs-en5-de5-0.6b.json`](benchmarks/fleurs-en5-de5-0.6b.json).

Interactive comparison chart: [`comparison-rtx.html`](comparison-rtx.html).

### How to reproduce

```bash
# Stop a coexisting container if you don't have ~6 GiB GPU memory free
docker compose -f ../qwen3guard/docker-compose.llama-gen-4b-rtx.yml down  # frees ~3.9 GiB

# Boot 1.7B
docker compose -f docker-compose.vllm-1.7b-bf16-rtx.yml up -d
docker logs -f qwen3asr-1.7b  # wait for "Application startup complete"

# Run benchmark
uv run python test_asr.py --warmup --json benchmarks/run-1.7b.json

# Repeat with 0.6B on port 12435
docker compose -f docker-compose.vllm-1.7b-bf16-rtx.yml down
docker compose -f docker-compose.vllm-0.6b-bf16-rtx.yml up -d
uv run python test_asr.py --endpoint http://localhost:12435/v1 --model qwen3-asr-0.6b --warmup --json benchmarks/run-0.6b.json
```

## Hardware sizing

| Config | Weights (BF16) | Recommended VRAM | Notes |
|--------|---------------:|-----------------:|-------|
| Qwen3-ASR-1.7B | ~3.4 GB | 16 GB+ | RTX 4090 / A10 / L4 all comfortable |
| Qwen3-ASR-0.6B | ~1.2 GB | 6 GB+ | Runs on consumer cards; bump `--max-num-seqs` on big cards |
| Qwen3-ForcedAligner-0.6B | ~1.2 GB | 6 GB+ | Same envelope as ASR-0.6B |

## Known issues

- **Nightly vLLM required.** Qwen3-ASR is not yet in vLLM stable v0.19.0 — the vendor recipe explicitly says `vllm --pre`. We pin `vllm/vllm-openai:nightly`. Once a stable release lands with Qwen3-ASR support, swap the tag.
- **Audio extras installed at container start.** The base `vllm-openai` image doesn't include `librosa` and `soundfile` (the `vllm[audio]` extras). We `pip install` them in the entrypoint — adds ~5–10 s to cold start. A custom Dockerfile would eliminate this; not done here to keep the layout single-file-per-variant.
- **Audio must be 16 kHz mono.** vLLM rejects other sample rates / bit-depths with `Invalid or unsupported audio file`. The Qwen vendor's own `asr_en.wav` (48 kHz 24-bit) is **not** accepted as-is — convert with `ffmpeg -ar 16000 -ac 1 -sample_fmt s16` first. `test_asr.py` uses FLEURS samples which are already 16 kHz mono.
- **Forced aligner currently doesn't load in vLLM.** Confirmed empirically (2026-05-12) — `Qwen/Qwen3-ForcedAligner-0.6B` crashes vLLM's weight loader with `AssertionError: loaded_weight.shape[output_dim] == self.org_vocab_size`. The aligner has a different vocab/output layer that vLLM's standard loader can't handle yet. The compose files are kept for when this is fixed upstream (or via a custom model class); for now use the `qwen-asr` Python package's `model.align(audio, text, ...)` API directly.
- **`--enforce-eager` is on by default.** Skipping CUDA-graph capture saves ~1–2 GiB of activation memory, which matters when coexisting with the heavy qwen3.6 container. Remove it (and bump `--gpu-memory-utilization` to e.g. 0.30) if you have headroom and want max single-stream throughput.
- **`--max-model-len` capped at 4096–8192.** Default 65 536 needs ~7 GiB just for one-sequence KV cache — too much when coexisting. ASR transcripts of clips ≤ 5 min stay well under 4 K tokens so this is safe.
- **No official FP8/AWQ variants from Qwen yet.** ~25 community quantizations exist on HF but none are vendor-blessed. At 1.7 B BF16 the weights are already only 3.4 GB, so quantization gains are limited.
- **No `--task transcription` flag set.** vLLM auto-detects from the model config; ASR routes (`/v1/audio/transcriptions`, `/v1/audio/translations`) appear in startup logs.

### Sizing math (in case you need to retune util)

vLLM's `--gpu-memory-utilization` is a **fraction of *total* GPU memory** that vLLM will plan to use. The actual constraint is two-sided:

1. `util × total ≥ weights + workspace + min KV` — too low and KV-cache budget goes negative.
2. `util × total ≤ free_at_startup` — too high and vLLM can't physically allocate.

Empirical numbers from our run (96 GiB card, ~9 GiB free after stopping qwen3guard, `--enforce-eager` on, `--max-model-len 4096`):

| Model | Weights | Workspace | Min budget | util used | Headroom |
|-------|--------:|----------:|-----------:|----------:|---------:|
| 0.6B | 1.5 GiB | ~3.1 GiB | ~5 GiB | 0.07 (6.7 GiB) | 2 GiB KV |
| 1.7B | 3.4 GiB | ~3.1 GiB | ~7 GiB | 0.09 (8.6 GiB) | 1 GiB KV |

The "workspace" line is mostly the audio encoder profiling — surprisingly heavy for these models. `--enforce-eager` is critical when coexisting; without it, add ~1–2 GiB to "workspace".

## Verification

After `docker compose ... up -d`:

```bash
# 1. wait for healthcheck green
docker compose -f docker-compose.vllm-1.7b-bf16-rtx.yml ps

# 2. confirm model loaded
docker logs -f qwen3asr-1.7b | grep -m1 "Uvicorn running"

# 3. confirm endpoint
curl -s http://localhost:12434/v1/models | jq

# 4. transcribe + WER smoke test
uv run python test_asr.py
```

Repeat with port 12435 (0.6B) to confirm. **The aligner (port 12436) is currently broken in vLLM** — see "Known issues" above; the compose file is preserved for when upstream support lands.
