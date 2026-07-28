#!/usr/bin/env bash
# Fetch the four checkpoints SUPIR needs (~21.5 GB total) into ./weights.
#
#   SUPIR-v0Q_fp16.safetensors  2.7 GB  Kijai's pruned fp16 SUPIR-Q (vs 5.3 GB fp32 .ckpt pickle)
#   sd_xl_base_1.0_0.9vae       6.9 GB  SDXL base the SUPIR deltas are applied on top of
#   CLIP-ViT-bigG open_clip     10.2 GB SDXL text encoder 2
#   clip-vit-large-patch14      1.7 GB  SDXL text encoder 1 (transformers dir layout)
#
# LLaVA (13B captioner) is deliberately NOT fetched — it only auto-writes the
# restoration prompt, and this service takes the prompt over the API instead.
set -euo pipefail
cd "$(dirname "$0")"
W="$PWD/weights"
mkdir -p "$W/CLIP-ViT-bigG-14-laion2B-39B-b160k" "$W/clip-vit-large-patch14"

hf download Kijai/SUPIR_pruned SUPIR-v0Q_fp16.safetensors --local-dir "$W"
hf download stabilityai/stable-diffusion-xl-base-1.0 sd_xl_base_1.0_0.9vae.safetensors --local-dir "$W"
hf download laion/CLIP-ViT-bigG-14-laion2B-39B-b160k open_clip_pytorch_model.bin \
  --local-dir "$W/CLIP-ViT-bigG-14-laion2B-39B-b160k"
# Fetched file-by-file rather than with --include: a multi-pattern --include
# silently skipped config.json, and the model then fails to load at boot with a
# missing-checkpoint error that points at the one file that didn't arrive.
for f in config.json merges.txt preprocessor_config.json pytorch_model.bin \
         special_tokens_map.json tokenizer.json tokenizer_config.json vocab.json; do
  hf download openai/clip-vit-large-patch14 "$f" --local-dir "$W/clip-vit-large-patch14"
done

echo "--- weights ---"
du -sh "$W"/* 2>/dev/null
