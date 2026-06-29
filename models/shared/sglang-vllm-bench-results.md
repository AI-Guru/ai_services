# SGLang vs vLLM benchmark campaign (2026-06-29, RTX PRO 6000 SM120)
# test_chat.py --runs 3 --warmup --no-think ; test_tools.py
# Note: --no-think NOT honored by these reasoning models; tok/s/TTFT include reasoning tokens.

## Qwen 3.6 27B (FP8 both sides, port 11436)
| Engine | Quant | avg tok/s | avg TTFT | Tools | Notes |
|---|---|---|---|---|---|
| SGLang (MTP/NEXTN) | FP8 | 72.1 | 22979 ms | 4/4 | Output CLEAN — README "NEXTN garbage" bug is now FIXED in current :latest |
| vLLM | FP8 | 66.2 | 28381 ms | 4/4 | clean |
# Qwen verdict: SGLang+MTP ~9% faster single-stream (72.1 vs 66.2), both 4/4 tools, both clean output.

## Gemma 4 26B-A4B
| Engine | Quant | avg tok/s | avg TTFT | Tools | Notes |
|---|---|---|---|---|---|
| vLLM | NVFP4 | 178.8 | 63 ms | 0/4 | Gemma 4 non-reasoning (no think). Tools 0/4 — vLLM compose has NO --tool-call-parser; config gap not engine limit |
| SGLang | NVFP4 | 157.1 | 24 ms | 4/4 | OVERTURNS research: CUTLASS NVFP4 MoE on SM120 works clean in current :latest (flashinfer_cutlass). Output verified coherent. |
| SGLang | BF16 | — | — | — | INFEASIBLE on this host: 30GB RAM, single 50GB safetensors shard -> mmap ENOMEM at load. Same reason vLLM uses NVFP4. NVFP4 is the viable SGLang Gemma path. |
# Gemma verdict: NVFP4 is the only viable 26B path on this host. SGLang NVFP4 works (157 tok/s, tools 4/4); vLLM NVFP4 faster throughput (179) but compose lacks tool parser. BF16 blocked by host RAM (both engines).

## Nemotron-3-Nano-30B-A3B (FP8 both sides)
| Engine | Quant | avg tok/s | avg TTFT | Tools | Notes |
|---|---|---|---|---|---|
| SGLang | FP8 | 224.1 | 789 ms | 4/4 | Required FIX: triton MoE runner OOMs SM120 smem (144KB > 99KB) in fused expert kernel -> use --moe-runner-backend flashinfer_cutlass. Output clean. |
| vLLM | FP8 | 271.5 | 646 ms | 4/4 | FLASHINFER attn. clean. |
# Nemotron verdict: vLLM ~21% faster (271 vs 224) single-stream; SGLang forced onto flashinfer_cutlass MoE (slower path) by SM120 smem limit. Both 4/4 tools, both clean.


## CONCURRENCY SWEEP — Nemotron Nano 30B FP8 (256 out tok/req, ignore_eos, no-think), 2026-06-29
# aggregate generation tok/s vs in-flight requests
| conc | SGLang agg tok/s | SGLang p50 lat |
|---|---|---|
| 1 | 220.9 | 1.16s |
| 4 | 522.8 | 1.96s |
| 8 | 863.0 | 2.37s |
| 16 | 1312.6 | 3.12s |
| 32 | 1926.9 | 4.25s |
| 64 | 3011.6 | 5.43s |

# FULL sweep both engines (agg gen tok/s):
| conc | SGLang | vLLM | winner |
|---|---|---|---|
| 1   | 220.9 | 261.3 | vLLM +18% |
| 4   | 522.8 | 581.4 | vLLM +11% |
| 8   | 863.0 | 903.0 | vLLM +5% |
| 16  | 1312.6 | 1344.6 | tie |
| 32  | 1926.9 | 1988.4 | tie |
| 64  | 3011.6 | 3031.6 | tie |
| 96  | 3643.7 | 3856.4 | vLLM +6% |
| 128 | 4076.6 | 4566.9 | vLLM +12% |
| 192 | 4082.9 | 5583.2 | vLLM +37% |
| 256 | 4496.0 | 6356.3 | vLLM +41% |
# SGLang plateaus ~4100 from c=128; vLLM scales to 6356 (still climbing at 256). 0 errors either side.
# CAVEATS: (1) SM120 forces SGLang onto slower flashinfer_cutlass MoE; (2) SGLang plateau may be an
# untuned default (max-running-requests / mem-fraction-static 0.85); (3) workload is pure-decode with a
# ~60-tok SHARED prompt -> does NOT exercise RadixAttention prefix-sharing (SGLang's signature win).
