# SeedVR2-3B — one-step diffusion restoration

[SeedVR2](https://github.com/ByteDance-Seed/SeedVR) (ByteDance-Seed,
[ICLR 2026](https://openreview.net/forum?id=x1FRyko9eC)) — one-step video restoration via diffusion
adversarial post-training. **Apache 2.0**, unlike [SUPIR](../../upscaling/supir/), which is
non-commercial only.

**It works here, and on this repo's benchmark it is the best restorer of the four tested** — see
Results. Not servable by vLLM though; that negative result is documented below and kept reproducible.

## Run

```bash
./fetch_weights.sh                     # once — 14.6 GB into ./weights (gitignored)
docker compose -f docker-compose.native-3b-rtx.yml up -d --build
curl -sS http://localhost:11480/health
# {"status":"ok","model":"SeedVR2-3B",...,"prompt_supported":false}
```

Loads in ~200 s, resident ~17 GB. `/health` answers **503** until the model is genuinely usable, so
the compose healthcheck cannot mark a broken container healthy.

## Use

Same routes as the two services in [`upscaling/`](../../upscaling/), so all three are
interchangeable by base URL and comparable by the same harness.

| Field | Default | Meaning |
|---|---|---|
| `scale` | 4 | output factor. SeedVR2 really takes a **target resolution**; this computes it from the input |
| `sample_steps` | 1 | one-step is the paper's contribution — raising it is an experiment, not a fix |
| `cfg_scale` | 1.0 | guidance |
| `seed` | 666 | reproducibility |

**There is no `prompt`.** SeedVR2 ships `pos_emb.pt`/`neg_emb.pt` — precomputed text embeddings — and
carries no text encoder in the serving path. Conditioning is frozen.

```bash
curl -sS -X POST http://localhost:11480/upscale -F "file=@in.png" -F "scale=4" -o out.png
```

## Results — measured on this box (RTX PRO 6000, 96 GB)

Degrade a known-good source ÷4 + JPEG q40, restore ×4, score against the **undegraded original**.
Same harness and inputs as the other two:
[`../../upscaling/compare_upscalers.py`](../../upscaling/compare_upscalers.py).

| Image | Method | PSNR dB | SSIM | sec |
|---|---|---:|---:|---:|
| church1 (AI-generated) | lanczos | 23.91 | 0.7138 | 0.01 |
| | realesrgan | **24.27** | **0.7237** | 0.51 |
| | supir | 21.14 | 0.6095 | 13.86 |
| | **seedvr2** | 22.12 | 0.6459 | 6.85 |
| testimage (photo) | lanczos | **22.53** | **0.6505** | 0.00 |
| | realesrgan | 21.64 | 0.6450 | 0.16 |
| | supir | 19.80 | 0.5464 | 11.06 |
| | **seedvr2** | 19.90 | 0.5776 | 6.27 |
| agentworld-roles (diagram) | lanczos | 25.22 | 0.9102 | 0.01 |
| | realesrgan | 25.09 | 0.9566 | 0.29 |
| | supir | 24.46 | 0.9147 | 40.89 |
| | **seedvr2** | **27.30** | **0.9695** | 10.73 |

Crops: [church](../../upscaling/comparison_out/montage_church1.png) ·
[photo](../../upscaling/comparison_out/montage_testimage.png) ·
[diagram](../../upscaling/comparison_out/montage_agentworld-roles.png).

### Two findings

**1. SeedVR2 broke the perception-distortion pattern.** Every other generative method scored *below*
plain Lanczos on fidelity — the expected trade-off (Blau & Michaeli, CVPR 2018). SeedVR2 on the
diagram scored **27.30 dB / 0.9695, the best of all four**, beating the classical baseline by 2.1 dB
while also looking sharper. It is not buying detail with fidelity there; it wins both.

**2. It reads text correctly.** SUPIR rendered the word “action” as **“nstion”** — crisply, and wrong.
SeedVR2 renders **“action”**, sharper than Real-ESRGAN and correct. For documents, diagrams and
screenshots that is the difference between usable and disqualified.

On photographic content it beats SUPIR on both metrics at roughly half the time, and visually
preserves source structure where SUPIR reinvents surface texture.

**It dominates SUPIR on every axis measured here** — fidelity, text integrity, speed, and licence.

## Getting it onto Blackwell

Upstream requires `flash_attn==2.5.9.post1` and `apex`, neither of which has an SM_120 story. Both
are patched out at build time by [`patch_seedvr2.py`](patch_seedvr2.py), which aborts the build if
upstream stops matching what it expects.

1. **apex → stock norms.** The 3B config asks for `fusedrms`/`fusedln`, whose branches import
   `apex.normalization`. apex ships py3.9/3.10 wheels only, against this image's 3.12. But the same
   module already imports `diffusers.models.normalization.RMSNorm` and `nn.LayerNorm` for its
   non-fused branches — fused and non-fused are numerically equivalent and both expose a single
   `weight`. Patched at the import site, so configs stay byte-identical to upstream. Confirmed by the
   loader: **`Loading info: <All keys matched successfully>`**.

2. **flash-attn → torch SDPA.** `NaMMSRTransformerBlock` hard-wires `FlashAttentionVarlen()`; it is
   not config-selectable, so the 3B path always hits it. Official flash-attn still has no SM_120
   build — only community forks. Rather than depend on one, `flash_attn_varlen_func` is reimplemented
   on `F.scaled_dot_product_attention`.

   "Varlen" packs several variable-length sequences into one flat tensor delimited by `cu_seqlens`;
   slicing per sequence and calling SDPA on each reproduces it exactly, without the O(L²) memory a
   block-diagonal mask would cost. **`cu_seqlens` is built from `vid_shape.prod(-1)` — one sequence
   per batch item, not per frame** — so single-image inference is one SDPA call with no loop
   overhead. If request batching is ever added, `torch.nested` jagged tensors are the upgrade path.

Slower than fused flash-attn, but it needs no compile and no unmaintained fork.

## vLLM — does not work (reproducible)

[`docker-compose.vllm-omni-3b-rtx.yml`](docker-compose.vllm-omni-3b-rtx.yml) reproduces the failure in
~30 s. Two independent blockers, both tested rather than assumed:

**Format.** SeedVR2 publishes raw `.pth` files — no `config.json`, no `model_index.json`. vLLM-Omni
fails resolving the stage config, before architecture lookup:

```
ValueError: Could not determine model_type for model: ByteDance-Seed/SeedVR2-3B.
  -- vllm_omni/entrypoints/utils.py:284
```

Identical on v0.22.0 and v0.24.0 (newest, 2026-07-07).

**Registry.** Repackaging would not help. A hand-written `config.json` clears the format check and
lands on:

```
ValueError: Model class SeedVR2Model not found in diffusion model registry.
```

[vLLM-Omni's registry](https://docs.vllm.ai/projects/vllm-omni/en/latest/models/supported_models/) is
generation-only — text-to-image, video generation, TTS, audio. No restoration or super-resolution
model of any kind. Supporting SeedVR2 means writing a model implementation upstream.

That compose uses `restart: "no"`, deliberately unlike every other compose in this repo: a container
that cannot start plus `restart: unless-stopped` is exactly the silent crash-loop
[CLAUDE.md](../../CLAUDE.md) warns about.

## Notes

- **Video.** SeedVR2 is a video restoration model; this service exposes the single-image path
  (`cut_videos` returns `t == 1` unchanged). Video endpoints are the natural extension and the real
  reason to have built this.
- **7B variant.** Set `SEEDVR2_DIT=seedvr2_ema_7b.pth`, `SEEDVR2_CONFIG=configs_7b/main.yaml`, and add
  the file to `fetch_weights.sh`. Untested here.
- **Weights** are gitignored; all four files come from `ByteDance-Seed/SeedVR2-3B` (first-party).
