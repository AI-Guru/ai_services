"""Make SUPIR's tiled VAE run without xformers.

Applied at image build time to the vendored checkout. Two separate upstream
problems, both only reachable when xformers is absent — which is our config,
because installing xformers would drag torch off the cu128/SM_120 wheel that
Blackwell needs (see requirements.txt).

1. tilevae.py:364 reads `if is_xformers_available:` — that tests the imported
   FUNCTION OBJECT, which is always truthy, so the tiled-VAE path unconditionally
   dispatches to xformers attention. With xformers missing, the first tile dies
   with `NameError: name 'xformers' is not defined`.

2. Fixing (1) the obvious way (adding `()`) is WRONG: the `elif` branch it then
   reaches, attn_forward_new_pt2_0, is written against a diffusers `Attention`
   module (self.group_norm / self.to_q), but sgm hands it a
   `MemoryEfficientAttnBlock` (self.q / self.k / self.v / self.proj_out). It
   fails with `'MemoryEfficientAttnBlock' object has no attribute 'group_norm'`.

So instead of rerouting the dispatch, swap the single op inside the function that
already has the right tensor plumbing. q/k/v arrive as (B, HW, C) single-head,
which is precisely what torch's scaled_dot_product_attention expects, making it a
true drop-in for xformers.ops.memory_efficient_attention.
"""
import pathlib
import sys

TARGET = pathlib.Path("/app/supir_src/SUPIR/utils/tilevae.py")

OLD = """    out = xformers.ops.memory_efficient_attention(
        q, k, v, attn_bias=None, op=self.attention_op)"""

NEW = """    # PATCHED (ai_services): xformers is deliberately not installed. q/k/v are
    # (B, HW, C) single-head here, so native SDPA is a drop-in replacement.
    out = F.scaled_dot_product_attention(q, k, v)"""

src = TARGET.read_text()
if OLD not in src:
    sys.exit(
        f"FATAL: expected xformers call not found in {TARGET}. "
        "Upstream changed — re-check SUPIR_SHA and this patch."
    )
TARGET.write_text(src.replace(OLD, NEW))

# Confirm nothing else in the tiled-VAE path still reaches for xformers at runtime.
remaining = [
    ln
    for ln in TARGET.read_text().splitlines()
    if "xformers.ops" in ln and not ln.strip().startswith("#")
]
if remaining:
    print(f"[patch_tilevae] note: {len(remaining)} other xformers.ops call(s) remain "
          "(unreached by the encode/decode task queue): " + "; ".join(r.strip() for r in remaining))
print("[patch_tilevae] ok — tiled VAE attention now uses torch SDPA")
