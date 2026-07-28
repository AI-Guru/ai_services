# SeedVR2 — cannot be served by vLLM-Omni (tested, two independent blockers)

[SeedVR2](https://github.com/ByteDance-Seed/SeedVR) (ByteDance-Seed,
[ICLR 2026](https://openreview.net/forum?id=x1FRyko9eC), accepted 2026-01-27) is one-step video
restoration via diffusion adversarial post-training — >4× faster than multi-step video-restoration
diffusion at comparable quality, in 3B and 7B variants, under **Apache 2.0**.

This directory holds the vLLM-Omni serving attempt. **It does not work.** What follows is the
evidence, because a reproducible negative result is worth more than an assertion.

## What was tested

| # | vLLM-Omni | Input | Result |
|---|---|---|---|
| 1 | v0.22.0 | `ByteDance-Seed/SeedVR2-3B` | `Could not determine model_type` |
| 2 | **v0.24.0** (newest, 2026-07-07) | `ByteDance-Seed/SeedVR2-3B` | identical failure |
| 3 | v0.24.0 | synthesized `config.json` | `SeedVR2Model not found in diffusion model registry` |

Reproduce test 2 with [`docker-compose.vllm-omni-3b-rtx.yml`](docker-compose.vllm-omni-3b-rtx.yml)
(~30 s, exits non-zero, leaves nothing running).

## Blocker 1 — SeedVR2 is not an HF-format model at all

```
ValueError: Could not determine model_type for model: ByteDance-Seed/SeedVR2-3B.
Model is not in standard transformers format and does not have model_index.json.
  -- vllm_omni/entrypoints/utils.py:284, resolve_model_config_path()
```

The entire HF repo is raw PyTorch tensors:

```
seedvr2_ema_3b.pth      the model            ema_vae.pth    the VAE
pos_emb.pt / neg_emb.pt precomputed text embeddings
apex-0.1-cp39/cp310-linux_x86_64.whl         (yes, they ship apex wheels)
```

No `config.json`. No `model_index.json`. No safetensors, no diffusers layout. vLLM-Omni resolves a
model by reading `config.json` → `model_type`/`architectures` → registry, so it fails while resolving
the stage config, **before architecture lookup even begins**.

Note the `pos_emb.pt`/`neg_emb.pt`: text conditioning is precomputed and frozen. There is no text
encoder in the serving path — it is a fixed-prompt restoration model, not a promptable one.

## Blocker 2 — the registry has no restoration architecture to bind to

Blocker 1 looks like a packaging problem, so it is fair to ask whether repackaging would fix it. It
would not. Feeding vLLM-Omni a hand-written `config.json` (`model_type: seedvr2`,
`architectures: ["SeedVR2Model"]`) gets past the format check and straight into:

```
ValueError: Model class SeedVR2Model not found in diffusion model registry.
RuntimeError: Orchestrator initialization failed
```

[vLLM-Omni's supported models](https://docs.vllm.ai/projects/vllm-omni/en/latest/models/supported_models/)
covers text-to-image (Qwen-Image, Z-Image, SDXL/3, FLUX, GLM-Image), video **generation** (Wan2.1/2.2,
Cosmos3, LTX-2), TTS, audio generation and omni-comprehension. It lists **no restoration,
super-resolution, upscaling or enhancement model of any kind**. Supporting SeedVR2 means *writing a
model implementation* in vLLM-Omni, not configuring one — the same conclusion
[`upscaling/README.md`](../../upscaling/README.md) reached for SUPIR, still true against 0.24.0.

Even then, SeedVR2's own inference is `torchrun --nproc-per-node=N projects/inference_seedvr2_3b.py`
with sequence parallelism (`sp_size=4` for 1080p) — batch-job shaped, not continuous-batching-server
shaped. The impedance mismatch is architectural, not cosmetic.

## What would actually work

1. **ComfyUI + the SeedVR2 wrapper** — the pragmatic route. Its v2.5 added GGUF, VAE tiling and
   BlockSwap specifically to escape H100-class requirements, and it sidesteps the build problem below.
2. **The official repo in a container** — requires `flash_attn==2.5.9.post1` (a 2024 pin with no
   SM_120 build) and **apex**, whose prebuilt wheels are py3.9/3.10 only against this box's 3.12.
   Strictly harder than the three patches SUPIR needed; apex has no clean escape hatch.

Neither is vLLM. That door is closed and does not open by configuration.

## Whether it is worth pursuing

**Licensing says yes, fit says probably not.**

Apache 2.0 makes SeedVR2 the frontier-quality restoration model this repo could legally use
commercially — a real advantage over [SUPIR](../../upscaling/supir/), which is non-commercial-only
under a licence that explicitly names consulting work.

But SeedVR2 is a **video** model and [`upscaling/`](../../upscaling/) serves images. Community
wrappers run it per-frame, but that pays a video model's complexity and VRAM (1×H100-80G for
100×720×1280) for a single-image job Real-ESRGAN completes in 0.3 s. Worth building **if video
restoration joins the roadmap** — in which case the licence makes it the only frontier option — and
not before.

## Naming

Directory is `SeedVR2` as requested; every other family here is lowercase (`z-image`, `qwen3.6`).
Rename to `seedvr2` if consistency matters more than the upstream capitalisation.
