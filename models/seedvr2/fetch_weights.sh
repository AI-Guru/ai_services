#!/usr/bin/env bash
# Fetch SeedVR2-3B (~14.6 GB) into ./weights.
#
#   seedvr2_ema_3b.pth  13.6 GB  the one-step restoration DiT
#   ema_vae.pth          1.0 GB  the VAE
#   pos_emb.pt/neg_emb.pt  ~0 GB PRECOMPUTED text embeddings — SeedVR2 has no text
#                                encoder in the serving path; conditioning is frozen
#
# Apache 2.0, unlike SUPIR next door in upscaling/supir (non-commercial only).
#
# Set SEEDVR2_SIZE=7B for the larger variant (different filenames, ~2x the weights).
set -euo pipefail
cd "$(dirname "$0")"
W="$PWD/weights"
mkdir -p "$W"

for f in seedvr2_ema_3b.pth ema_vae.pth pos_emb.pt neg_emb.pt; do
  hf download ByteDance-Seed/SeedVR2-3B "$f" --local-dir "$W"
done

echo "--- weights ---"
du -sh "$W"/* 2>/dev/null
