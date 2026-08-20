# Stable Audio 3 Small-SFX — text-to-sound-effect on one RTX PRO 6000

`stabilityai/stable-audio-3-small-sfx` served as a resident FastAPI service with a Gradio
UI mounted on the same app — the same arrangement as [`models/minimax-h3`](../minimax-h3),
scaled down by a factor of thirty.

433M-parameter latent diffusion transformer over the SAME-S autoencoder. 44.1 kHz stereo,
up to 120 s, **8 sampling steps**. Three inference modes: text-to-audio, audio-to-audio
restyling, and inpainting/continuation.

| | |
|---|---|
| UI | <http://localhost:8400/ui> |
| REST | <http://localhost:8400/v1/audio> |
| Container | `stable-audio-3-sfx` |
| Compose | `docker-compose.sa3-small-sfx-rtx.yml` |
| VRAM | ~2 GB (1.69 GB at 5 s, 2.40 GB at 120 s, upstream figures) |

## License — no territorial carve-out, but two licences apply

[Stability AI Community License](https://stability.ai/license). Free for research and for
commercial use under an annual-revenue threshold; above it Stability wants an enterprise
licence. **Unlike the MiniMax H3 stack next door, Germany is not excluded.**

The bundled text encoder is T5Gemma, redistributed under the **Gemma Terms of Use**, which
carry their own prohibited-use policy. Both licences reach Outputs, not just
redistribution.

## Two gates, not one

Both source repos are gated on the Hub, and the second one is easy to miss because nothing
in the model card's quick-start mentions it:

| Repo | Size | What it is |
|---|---|---|
| [`stabilityai/stable-audio-3-small-sfx`](https://huggingface.co/stabilityai/stable-audio-3-small-sfx) | ~1.7 GB | DiT + SAME-S autoencoder |
| [`google/t5gemma-b-b-ul2`](https://huggingface.co/google/t5gemma-b-b-ul2) | ~1.2 GB | text conditioner |

Accept **both** with the account that owns `HF_TOKEN`, then:

```bash
echo 'HF_TOKEN=hf_...' > .env      # gitignored
```

A missing or unaccepted token fails at model load with a 401 that reads like a network
error and is not one. Check with:

```bash
docker logs stable-audio-3-sfx 2>&1 | grep -iE '401|gated|unauthor|awaiting a review'
```

## Run

```bash
docker compose -f docker-compose.sa3-small-sfx-rtx.yml up -d --build
```

First start downloads ~3 GB into the `stableaudio3_huggingface_cache` volume. Nothing to
prepare — unlike `models/minimax-h3` there is no local quantization pass; these weights
ship at the size they run at.

```bash
curl -s localhost:8400/health | python3 -m json.tool
```

### It does not need the card to itself

This is the one model in this repo that breaks the "one at a time" rule in
[`CLAUDE.md`](../../CLAUDE.md). At ~2 GB it fits beside a vLLM instance holding
`--gpu-memory-utilization 0.90` of the 96 GB card. Nothing needs stopping first.

Sharing SMs is still not free — if the resident LLM is being benchmarked for throughput,
generate your SFX before or after, not during.

## API

```bash
# blocking — the clip comes back in the body, metadata in the X-SA3-Info header
curl -s -X POST localhost:8400/v1/audio/sync \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"heavy wooden door creaking open on rusted hinges, echoing stone hallway",
       "duration":6,"seed":1234}' -o door.wav

# queued — returns a job id
curl -s -X POST localhost:8400/v1/audio -H 'Content-Type: application/json' \
  -d '{"prompt":"distant thunder over a valley","duration":12,"batch_size":4}'
curl -s localhost:8400/v1/audio/<id>
curl -s localhost:8400/v1/audio/<id>/content?index=2 -o thunder2.wav
```

| Field | Default | Notes |
|---|---|---|
| `prompt` | — | string, or a list of length `batch_size` for per-item prompts |
| `duration` | `7.0` | seconds, ceiling 120; list allowed with `batch_size` |
| `steps` | `8` | tuned default; see below |
| `seed` | `-1` | `-1` = random |
| `batch_size` | `1` | cheaper than N separate requests |
| `format` | `wav` | whatever this libsndfile build reports in `/health` |
| `init_audio` / `init_noise_level` | — | audio-to-audio restyle |
| `inpaint_audio` / `inpaint_mask_start_seconds` / `inpaint_mask_end_seconds` | — | scalars, or equal-length lists for several regions |
| `cfg_scale` / `negative_prompt` | `1.0` / — | **inert here** — see below |

`sfx.py` is a CLI over the same endpoints:

```bash
./sfx.py "single glass bottle shattering on concrete, close mic" --duration 4 -o glass.wav
./sfx.py --health
```

## Five things that will bite

### 1. `cfg_scale` and `negative_prompt` do nothing on this checkpoint

They are **base-model-only** parameters. `small-sfx` is post-trained, so both are silently
ignored by the sampler. The server does not pretend otherwise: pass either one and the
response carries an `ignored` field saying so, and the UI surfaces it as a warning. If you
need guided sampling, deploy `small-sfx-base` (`SA3_MODEL=small-sfx-base`) — and then also
raise `steps` to ~50, because the 8-step default is a property of the post-training.

### 2. More steps is not better

8 is not a speed compromise, it is the tuned operating point. Upstream is explicit that
going above 8 does not improve post-trained output. Lower trades quality for latency.

### 3. `init_audio` wants `(sample_rate, tensor)` — upstream's own examples are reversed

The docstring in `stable_audio_3/model.py` says `(sample_rate, audio)` and
`_encode_audio_input` unpacks `in_sr, audio_data = audio_input`. But every example in the
repo README and in `docs/workflows/inference.md` passes `torchaudio.load(path)` straight
in, and `torchaudio.load` returns `(tensor, sample_rate)` — the other way round.

Passing it upstream's way makes the sample rate a tensor. `server.py` builds the tuple
explicitly (`load_audio_field`) so callers never see this, but it will bite anyone
adapting the upstream snippets by hand.

### 4. The cu126 pin has no Blackwell kernels

`pyproject.toml` pins `torch==2.7.1` and routes it through an explicit `pytorch-cu126`
index. cu126 predates Blackwell — no `sm_120` kernels, so every launch dies with *"no
kernel image is available for execution on the device"*. Upstream documents the escape
hatch (install torch yourself from another channel, then sync without reinstalling it).

Rather than fight uv's lockfile, `Dockerfile.sa3-server-rtx` starts from
`pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime` — the same torch version upstream pins,
built against CUDA 12.8 — and installs the package `--no-deps` so its `torch==2.7.1`
requirement cannot pull the cu126 wheel back over it. The build asserts
`sm_120 in torch.cuda.get_arch_list()`, so a wrong base image fails at `docker build`
rather than at the first generation.

### 5. Flash-attention is a Medium problem, not a Small one

Upstream requires flash-attn only for `medium`/`medium-base`, which use the SAME-L
autoencoder; the symptom when it is missing is *output that is a static glitch sound*, not
a crash. Small-SFX uses SAME-S and never calls it, so this image deliberately does not
build it — which matters, because the community prebuilt-wheel index upstream points at
has nothing for `sm_120`, so a Medium deployment on this card means a source build.

**Do not set `SA3_MODEL=medium` on this image without adding flash-attn first.** It will
load, and it will produce noise.

## Chunked decode is off here, inverting the upstream default

Chunked decoding splits the final autoencoder decode into overlapping windows to cap peak
VRAM. Upstream enables it for every model. The ceiling it protects is 2.4 GB on a 96 GB
card, and it costs compute plus possible stitching artefacts at the seams — so
`SA3_CHUNKED_DECODE=0` in the compose. Set it to `1` to restore upstream behaviour, or
override per request with `"chunked_decode": true`.

## Prompting for SFX is not prompting for music

The model responds to the physical event and the space it happens in — **material,
action, distance, room** — far more than to genre or mood words. `heavy wooden door
creaking open slowly on rusted hinges, echoing stone hallway` outperforms `scary door
sound`. The UI ships eight example prompts in this shape.

## Measured — RTX PRO 6000 Blackwell, 8 steps, batch 1, chunked decode off

Three warm runs each, median. `generation_s` is `model.generate()` wall clock with an
explicit `torch.cuda.synchronize()` before the timer stops (see the note below).

| Duration | Generation | × realtime | Peak VRAM | Upstream H200 | Upstream peak |
|---|---|---|---|---|---|
| 5 s | 0.12 s | 42× | 1.78 GB | 0.41 s | 1.69 GB |
| 30 s | 0.13 s | 234× | 1.94 GB | 0.46 s | 1.89 GB |
| 120 s | 0.15 s | 800× | 2.46 GB | 0.45 s | 2.40 GB |

Cold first request after load is ~0.68 s; everything above is warm. Batch scales nearly
free — 3 × 12 s came back in 0.14 s total.

VRAM lands just above upstream's figures, which is expected: those were measured with
chunked decode on and this deployment turns it off.

The latency column sits well below upstream's H200 numbers, which is not a claim that this
card beats an H200. Upstream does not state whether their figure includes the autoencoder
decode, chunking, or per-call overhead, and at ~0.1 s the measurement is dominated by
whatever is or isn't counted. Treat the local column as internally consistent and the
cross-comparison as unresolved.

### Timing this model needs an explicit sync

`model.generate()` returns once the kernels are **queued**, not once they finish. Timed
naively it reports a flat ~0.13 s for every duration from 5 s to 120 s — a 24× range —
because the remaining GPU work lands on whatever forces the next sync, here the `.cpu()`
in the file write. At 120 s that hid ~40 % of the real time (round-trip 0.22 s vs a
reported 0.12 s). `server.py` calls `torch.cuda.synchronize()` before stopping the timer.
Anyone benchmarking this model with a bare `time.time()` pair will get numbers that are
too good and, more tellingly, too flat.

## Other variants

`SA3_MODEL` swaps the checkpoint without rebuilding:

| Value | Params | Max | Notes |
|---|---|---|---|
| `small-sfx` | 433M | 120 s | default |
| `small-music` | 433M | 120 s | music rather than SFX |
| `small-sfx-base`, `small-music-base` | 433M | 120 s | CFG + negative prompts work; use ~50 steps |
| `medium`, `medium-base` | 1.4B | 380 s | **needs flash-attn — see trap 5** |

## Troubleshooting

| Symptom | Cause |
|---|---|
| 401 / "gated" at load | one of the two Hub licences not accepted, or `HF_TOKEN` missing from `.env` |
| 401 on *every* repo, gated or not | the token itself is dead, not unscoped. Confirm before chasing licences: `curl -s -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/api/whoami-v2` — a revoked token answers `{"error":"Invalid username or password."}` while a merely unscoped one still names your user |
| "no kernel image is available" | torch from a pre-Blackwell CUDA channel — see trap 4 |
| output is static/noise | `SA3_MODEL=medium` without flash-attn — see trap 5 |
| `cfg_scale` seems to do nothing | it doesn't — see trap 1 |
| `format 'mp3' unsupported` | this libsndfile build lacks it; `/health` lists what works |
| container crash-loops | `docker logs stable-audio-3-sfx`, then `docker compose -f docker-compose.sa3-small-sfx-rtx.yml down` — do not leave it looping |
