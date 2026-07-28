"""Make SeedVR2 run on Blackwell (SM_120) without flash-attn or apex.

Applied at image build time to the vendored checkout. Two upstream hard
dependencies block this box, and neither needs to.

--------------------------------------------------------------------------
1. flash-attn  ->  torch SDPA
--------------------------------------------------------------------------
models/dit{,_v2}/attention.py does a top-level `from flash_attn import
flash_attn_varlen_func`, and NaMMSRTransformerBlock hard-wires
`self.attn = FlashAttentionVarlen()` — it is not config-selectable, so the
3B config (dit_v2.nadit / block_type mmdit_sr) always lands on it.

Official flash-attn still has no SM_120 support; only community forks do.
Rather than depend on an unmaintained fork, this substitutes a pure-torch
implementation of the ONE function actually used.

The call site (dit_v2/nablocks/attention/mmattn.py:123) passes exactly:
    q, k, v            (total_tokens, nheads, head_dim), bfloat16, PACKED
    cu_seqlens_q/k     int32 (batch+1,), cumulative lengths with a leading 0
    max_seqlen_q/k     int
(plus `deterministic`, injected by the FlashAttentionVarlen wrapper).

"Varlen" means several variable-length sequences are concatenated into one
flat tensor and delimited by cu_seqlens, so that attention never crosses a
boundary. Slicing per sequence and calling SDPA on each reproduces that
exactly. A block-diagonal mask over the packed tensor would also work but
costs O(L^2) memory; slicing costs none. For single-image inference there is
one sequence, so this is a single SDPA call with no loop overhead.

Slower than flash-attn (no kernel fusion, no IO-awareness) and it allocates
more for long sequences — an honest trade for not compiling a fork.

--------------------------------------------------------------------------
2. apex  ->  stock torch / diffusers norms
--------------------------------------------------------------------------
The 3B config asks for `fusedrms`/`fusedln`, whose branches import
apex.normalization.Fused{RMS,Layer}Norm. apex ships py3.9/3.10 wheels only,
against this image's 3.12.

models/dit{,_v2}/normalization.py ALREADY imports
diffusers.models.normalization.RMSNorm and torch nn.LayerNorm for its
non-fused `rms`/`layer` branches. Fused and non-fused are numerically
equivalent and both expose a single `weight` parameter, so checkpoint keys
load unchanged — the fused variants are a speed optimisation, not a
different layer.

Patched at the import site rather than by rewriting configs_3b/main.yaml, so
the config stays byte-identical to upstream and the fallback is automatic
wherever apex is genuinely present.
"""
import pathlib
import sys

SRC = pathlib.Path("/app/seedvr_src")

# ---------------------------------------------------------------- attention
ATTN_OLD = "from flash_attn import flash_attn_varlen_func"

ATTN_NEW = '''# PATCHED (ai_services): flash-attn has no SM_120 build; use torch SDPA.
try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    import torch as _torch
    import torch.nn.functional as _F

    def flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q=None, max_seqlen_k=None,
        dropout_p=0.0, softmax_scale=None, causal=False, **_ignored,
    ):
        """Pure-torch stand-in for flash_attn_varlen_func.

        q/k/v are packed (total_tokens, nheads, head_dim); cu_seqlens delimits
        the concatenated sequences. Attention must not cross those boundaries,
        so slice per sequence and run SDPA on each. Returns the same packed
        layout the caller expects.
        """
        cq = cu_seqlens_q.tolist()
        ck = cu_seqlens_k.tolist()
        outs = []
        for i in range(len(cq) - 1):
            qi = q[cq[i]:cq[i + 1]].transpose(0, 1).unsqueeze(0)   # 1 h lq d
            ki = k[ck[i]:ck[i + 1]].transpose(0, 1).unsqueeze(0)   # 1 h lk d
            vi = v[ck[i]:ck[i + 1]].transpose(0, 1).unsqueeze(0)
            oi = _F.scaled_dot_product_attention(
                qi, ki, vi,
                dropout_p=dropout_p,
                is_causal=causal,
                scale=softmax_scale,
            )
            outs.append(oi.squeeze(0).transpose(0, 1))             # lq h d
        return _torch.cat(outs, dim=0)
'''

# ------------------------------------------------------------- normalization
NORM_OLD_LN = """        if norm_type == "fusedln":
            from apex.normalization import FusedLayerNorm
"""
NORM_NEW_LN = """        if norm_type == "fusedln":
            # PATCHED (ai_services): apex ships py3.9/3.10 wheels only. The
            # non-fused nn.LayerNorm is numerically equivalent and shares the
            # same single `weight` parameter, so checkpoints load unchanged.
            try:
                from apex.normalization import FusedLayerNorm
            except ImportError:
                from torch.nn import LayerNorm as FusedLayerNorm
"""

NORM_OLD_RMS = """        if norm_type == "fusedrms":
            from apex.normalization import FusedRMSNorm
"""
NORM_NEW_RMS = """        if norm_type == "fusedrms":
            # PATCHED (ai_services): see fusedln above. diffusers' RMSNorm is
            # already imported at the top of this module for the `rms` branch.
            try:
                from apex.normalization import FusedRMSNorm
            except ImportError:
                def FusedRMSNorm(normalized_shape, elementwise_affine=True, eps=1e-5):
                    return RMSNorm(
                        dim=normalized_shape,
                        eps=eps,
                        elementwise_affine=elementwise_affine,
                    )
"""


def patch(path: pathlib.Path, pairs: list[tuple[str, str]]) -> None:
    """Apply each (old, new) exactly once. Runs on a fresh clone at build time, so
    it is deliberately NOT idempotent — it hard-fails rather than silently skipping.

    (An earlier version tried to detect "already patched" by comparing the first
    line of `new` against the file. For the norm patches that first line is the
    unchanged `if norm_type == ...:` guard, so the check matched every time and
    skipped the patch while still reporting success. Hence the post-conditions
    below: a patch that does not apply must break the build, not the container.)
    """
    if not path.exists():
        sys.exit(f"FATAL: {path} missing — upstream layout changed?")
    src = path.read_text()
    for old, new in pairs:
        if old not in src:
            sys.exit(f"FATAL: expected snippet not found in {path}:\n{old[:160]}")
        src = src.replace(old, new, 1)
    path.write_text(src)
    print(f"[patch_seedvr2] patched {path.relative_to(SRC)}")


def assert_guarded(path: pathlib.Path, module: str) -> None:
    """Every import of `module` must sit under a try/except ImportError."""
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        if f"import {module}" in line and not line.strip().startswith("#"):
            window = "\n".join(lines[max(0, i - 4) : i])
            if "try:" not in window:
                sys.exit(
                    f"FATAL: unguarded `{module}` import at {path.relative_to(SRC)}:{i + 1}\n"
                    f"  {line.strip()}\n"
                    "The patch did not take — the container would fail at model build."
                )


for variant in ("dit", "dit_v2"):
    attn = SRC / "models" / variant / "attention.py"
    norm = SRC / "models" / variant / "normalization.py"
    patch(attn, [(ATTN_OLD, ATTN_NEW)])
    patch(norm, [(NORM_OLD_LN, NORM_NEW_LN), (NORM_OLD_RMS, NORM_NEW_RMS)])
    assert_guarded(attn, "flash_attn")
    assert_guarded(norm, "apex")

print("[patch_seedvr2] ok — no flash-attn, no apex required (imports verified guarded)")
