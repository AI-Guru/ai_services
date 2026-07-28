"""SeedVR2's generation_step, ported verbatim from upstream.

Lifted from projects/inference_seedvr2_3b.py so the sampling maths is identical to
the reference implementation — only the prints are dropped and sync_data is a no-op
here (it broadcasts across sequence-parallel ranks; this service is single-rank).

Kept in its own module so any divergence from upstream is reviewable in one place.
"""
from __future__ import annotations

import torch
from einops import rearrange


def generation_step(runner, text_embeds_dict: dict, cond_latents: list):
    from common.distributed import get_device

    device = get_device()

    noises = [torch.randn_like(latent) for latent in cond_latents]
    aug_noises = [torch.randn_like(latent) for latent in cond_latents]

    # Upstream calls sync_data(..., 0) here to broadcast noise across
    # sequence-parallel ranks. World size is 1, so it is an identity.
    noises = [x.to(device) for x in noises]
    aug_noises = [x.to(device) for x in aug_noises]
    cond_latents = [x.to(device) for x in cond_latents]

    cond_noise_scale = 0.0

    def _add_noise(x, aug_noise):
        t = torch.tensor([1000.0], device=device) * cond_noise_scale
        shape = torch.tensor(x.shape[1:], device=device)[None]
        t = runner.timestep_transform(t, shape)
        return runner.schedule.forward(x, aug_noise, t)

    conditions = [
        runner.get_condition(
            noise,
            task="sr",
            latent_blur=_add_noise(latent_blur, aug_noise),
        )
        for noise, aug_noise, latent_blur in zip(noises, aug_noises, cond_latents)
    ]

    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=True):
        video_tensors = runner.inference(
            noises=noises,
            conditions=conditions,
            dit_offload=True,
            **text_embeds_dict,
        )

    samples = [
        (
            rearrange(video[:, None], "c t h w -> t c h w")
            if video.ndim == 3
            else rearrange(video, "c t h w -> t c h w")
        )
        for video in video_tensors
    ]
    del video_tensors
    return samples
