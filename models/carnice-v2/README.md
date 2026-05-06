# Carnice-V2-27B

[`kai-os/Carnice-V2-27b`](https://huggingface.co/kai-os/Carnice-V2-27b) — a full-merge BF16 SFT of `Qwen/Qwen3.6-27B` for Hermes-style agent traces. Apache 2.0. Same hybrid architecture as base Qwen3.6-27B (16 full Gated Attention layers + 48 Gated DeltaNet layers, mtp_num_hidden_layers=1 in config).

The model card describes it as *"intended for agentic Hermes-style use; validate with your own agent harness before relying on it for production behavior."* SFT was assistant-token-loss-only on 6 554 windowed examples (1 508 Carnice + 1 015 DJLougen Hermes + 950 Lambda GLM-5.1 Hermes). Reported lift over base: IFEval prompt-strict 85 → 90 %, instruction-strict 90 → 93.3 %, eval perplexity 1.83 → 1.51.

Native weights are **~51 GB BF16 safetensors**. Community quants exist as GGUF (kai-os official), NVFP4-TEXT-MTP (sakamakismile), MLX (Tranquil-Flow), and INT4 AutoRound (gilbertb / wasifb).

## Compose variants

| File | Engine | Format | Weight VRAM | Speculative | Status |
|------|--------|--------|------------:|-------------|--------|
| `docker-compose.vllm-bf16-rtx.yml` | vLLM nightly | BF16 (native) | 50.2 GiB | none | ✅ works |
| `docker-compose.vllm-bf16-mtp-rtx.yml` | vLLM nightly | BF16 (native) + MTP | n/a | qwen3_next_mtp | ❌ **does not start** — see below |
| `docker-compose.vllm-nvfp4-mtp-rtx.yml` | vLLM nightly | NVFP4 group=16 + MTP | 18.4 GiB | qwen3_next_mtp | ✅ works — **fastest** |
| `docker-compose.llama-q4-rtx.yml` | llama.cpp server-cuda | GGUF Q4_K_M | ~15 GiB | none | ✅ works |
| `docker-compose.llama-q8-rtx.yml` | llama.cpp server-cuda | GGUF Q8_0 | ~27 GiB | none | ✅ works |

All composes serve on **port 11475** as `carnice-v2`.

The vLLM BF16 composes need `--max-num-seqs 256` (default 1024) — the hybrid Mamba/DeltaNet cache budget runs out at default with 50 GB BF16 weights. Both Q4 and Q8 GGUF composes use the **full 262 144-token native context**.

## Measured throughput (test_chat.py, --runs 3 --warmup --no-think)

Single-user single-stream on RTX PRO 6000 Blackwell (96 GB GDDR7, ~1.6 TB/s). Container started fresh per variant, weights cached after first pull.

| Variant | Avg tok/s | Per-run tok/s | TTFT avg | Notes |
|---|---:|---|---:|---|
| **NVFP4 + MTP** (sakamakismile) | **83.7** | 82.7 / 84.3 / 84.1 | 17.6 s | Fastest viable. NVFP4 group=16 + the baked-in MTP head. Matches base Qwen3.6-27B-NVFP4-MTP at 83.5 tok/s — the Carnice SFT didn't degrade MTP acceptance. |
| llama.cpp Q4_K_M | 70.7 | 70.7 / 70.7 / 70.6 | 16.3 s | After warm-up. First cold run reads ~31 tok/s due to the llama.cpp `failed to truncate tokens` chat-template warm-up artifact; re-running gives steady-state 70.7. |
| llama.cpp Q8_0 | 47.6 | 47.4 / 47.7 / 47.7 | 30.2 s | Higher quality reference; 1.7× the bandwidth of Q4, ~1/1.5 the throughput. |
| vLLM BF16 vanilla | 28.3 | 28.3 / 28.3 / 28.3 | 45.5 s | Bandwidth ceiling: 1.6 TB/s ÷ 50 GB ≈ 32 tok/s theoretical, we hit ~88 % of it. Highest fidelity by construction. |
| vLLM BF16 + MTP | — | — | — | **Does not start** — see "Known issues" below. |

### Headline

**Use `docker-compose.vllm-nvfp4-mtp-rtx.yml`** — 83.7 tok/s with full MTP speculation, ~3× the BF16 ceiling, ~1.2× the Q4_K_M GGUF, and the smallest weight footprint on disk (18.3 GB).

## Known issue: kai-os BF16 ships without MTP head weights

`kai-os/Carnice-V2-27b`'s `config.json` declares `mtp_num_hidden_layers=1` (inherited from the base), but the BF16 safetensors **do not contain the MTP head parameters**. vLLM nightly tries to load them when `--speculative-config method=qwen3_next_mtp` is set and bails:

```
ValueError: Following weights were not initialized from checkpoint:
  {'model.pre_fc_norm_hidden.weight', 'model.layers.0.mlp.gate_up_proj.weight',
   'model.layers.0.self_attn.qkv_proj.weight', 'model.norm.weight',
   'model.fc.weight', 'model.pre_fc_norm_embedding.weight', ...}
```

This is most likely a side-effect of the Unsloth LoRA-merge step — the LoRA adapter only covered the main model, and the merge didn't preserve (or never had) the MTP head. **The community NVFP4-MTP repackage by `sakamakismile` rebundles the MTP head weights correctly**, which is why the NVFP4 variant works while the direct BF16 path doesn't. The `docker-compose.vllm-bf16-mtp-rtx.yml` is kept in the repo as documentation of the failure mode.

If a future Carnice release fixes this (re-trains the MTP head post-merge or includes the original Qwen3.6-27B MTP weights in the safetensors), the BF16 + MTP path would unlock — expected ~45–50 tok/s based on the +41 % MTP gain we measured on base Qwen3.6-27B FP8.

## Notes

- **No vision input.** Even though base Qwen3.6-27B is multimodal, neither the kai-os GGUF repo nor `sakamakismile/...-NVFP4-TEXT-MTP` ships an `mmproj-*.gguf` file. The native BF16 still has the vision tower in its safetensors but the composes pass `--language-model-only`.
- **DFlash drafter not tested.** `z-lab/Qwen3.6-27B-DFlash` is trained against base Qwen3.6-27B logits; pairing it with a Carnice target (BF16 or NVFP4) would likely give some acceptance loss vs. the base config, and the BF16 path's missing MTP head suggests the safer route is the NVFP4-MTP recipe. A future test could pair NVFP4 + DFlash; not done here.
- **`max-num-seqs` constraint.** The vLLM BF16 compose needs `--max-num-seqs 256` (down from the default 1024) because the hybrid model's per-sequence Mamba state cache competes with the 50 GB weight footprint. The NVFP4 compose works at the same 256 setting; default 1024 might also work given the smaller weights — not retried.
- **Q4 first-run anomaly.** llama.cpp 's first chat completion after server start emitted a `failed to truncate tokens with position >= 10` log line and ran at 31.6 tok/s; subsequent runs all settled at 70.7. The recorded number is the warm steady-state.

## Quick start (recommended path)

```bash
# Pull the NVFP4-MTP variant (~18 GB) and start
docker compose -f docker-compose.vllm-nvfp4-mtp-rtx.yml up -d

# Wait for healthy then test
curl http://localhost:11475/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "carnice-v2",
    "messages": [{"role":"user","content":"Write a short Python function that returns the nth Fibonacci number iteratively."}],
    "max_tokens": 400
  }'
```

API endpoint: `http://localhost:11475/v1`, model name: `carnice-v2`.

## References

- Collection: <https://huggingface.co/collections/kai-os/carnice-v2>
- Base model: <https://huggingface.co/kai-os/Carnice-V2-27b>
- GGUF (kai-os official): <https://huggingface.co/kai-os/Carnice-V2-27b-GGUF>
- NVFP4-MTP (sakamakismile): <https://huggingface.co/sakamakismile/Carnice-V2-27b-NVFP4-TEXT-MTP>
- Base architecture (Qwen3.6-27B): <../qwen3.6/>
