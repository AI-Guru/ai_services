"""SM_120 smoke test: Z-Image-Turbo via diffusers on Blackwell.

Confirms the diffusers + PyTorch CUDA stack actually loads and generates on the
RTX PRO 6000 (compute capability 12.0) before we commit to compose files.
"""
import time

import torch
from diffusers import ZImagePipeline

print("torch:", torch.__version__, "| cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
cap = torch.cuda.get_device_capability(0)
print(f"compute capability: sm_{cap[0]}{cap[1]}")
assert cap == (12, 0), f"expected SM_120, got {cap}"

print("\nloading Tongyi-MAI/Z-Image-Turbo ...")
t0 = time.time()
pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)
pipe.to("cuda")
print(f"loaded in {time.time() - t0:.1f}s")

prompt = (
    "a photorealistic red panda barista pouring latte art into a cup, cozy "
    "cafe interior, warm light, shallow depth of field, the cup reads 'Z-IMAGE'"
)
print("\ngenerating 1024x1024 ...")
t0 = time.time()
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    num_inference_steps=9,
    guidance_scale=0.0,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
dt = time.time() - t0
image.save("/out/zimage_smoke.png")
print(f"\nOK: generated in {dt:.1f}s")
print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
print("saved /out/zimage_smoke.png")
