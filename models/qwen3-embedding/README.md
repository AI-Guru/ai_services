# Qwen3-Embedding & Reranker — Self-Hosted Inference

Runs the Qwen3-Embedding and Qwen3-Reranker families via vLLM on the RTX PRO 6000,
exposing OpenAI-compatible **embeddings** and **reranking/scoring** APIs. These are
the RAG building blocks (vectorize → retrieve → rerank) — the one capability the
rest of the fleet (all text-generation) doesn't cover.

Apache 2.0, **not gated** (no `HF_TOKEN` needed). Both families are built on
`Qwen3ForCausalLM` backbones; vLLM serves them with the **pooling** runner.

## Models

| Model | Role | Params | Output | Context | Port |
|-------|------|--------|--------|---------|------|
| **Qwen3-Embedding-0.6B** | Embedding | 0.6B | 1024-d vector | 32K* | 11463 |
| **Qwen3-Embedding-4B** | Embedding | 4B | 2560-d vector | 40K* | 11464 |
| **Qwen3-Embedding-8B** | Embedding | 8B | 4096-d vector | 40K* | 11465 |
| **Qwen3-Reranker-0.6B** | Reranker | 0.6B | 0–1 score | 32K | 11466 |
| **Qwen3-Reranker-4B** | Reranker | 4B | 0–1 score | 40K | 11467 |
| **Qwen3-Reranker-8B** | Reranker | 8B | 0–1 score | 40K | 11468 |

\* The configs cap `--max-model-len` at **16384** so they start with a small KV
footprint and co-host alongside a primary chat model. Raise it toward the model's
native max (32K/40K) if you embed/rerank longer documents — you'll also need to
raise `--gpu-memory-utilization` (see Notes).

## Requirements

- NVIDIA GPU, ≥5 GB free for 0.6B, ≥12 GB for 4B, ≥22 GB for 8B (RTX PRO 6000 96 GB)
- Docker + NVIDIA Container Toolkit
- `vllm/vllm-openai:v0.22.0-cu129-ubuntu2404` (cu129 runs fine on Blackwell SM_120)

These are small and **co-hostable**: `--gpu-memory-utilization` is 0.05 / 0.12 / 0.22
(0.6B / 4B / 8B), so an embedder + reranker pair runs comfortably next to a chat model.

## Which variant should I use?

```
RAG pipeline?
├─ Vectorize documents/queries → an EMBEDDING model
│  ├─ Cheapest / co-host ───── embed-0.6b (1024-d, ~5 GB)
│  ├─ Balanced ─────────────── embed-4b  (2560-d, ~12 GB)
│  └─ Best quality ─────────── embed-8b  (4096-d, ~22 GB)
│
└─ Re-order retrieved candidates → a RERANKER (cross-encoder)
   ├─ Cheapest / co-host ───── rerank-0.6b
   ├─ Balanced ─────────────── rerank-4b
   └─ Best quality ─────────── rerank-8b
```

Typical setup: a small/medium **embedder** for first-stage retrieval, then a
**reranker** to re-score the top-K. Pair sizes to taste (e.g. embed-4b + rerank-0.6b).

## Quick Start

```bash
# Start the lightweight embedder + reranker pair
cd models/qwen3-embedding
docker compose -f docker-compose.vllm-embed-0.6b-rtx.yml up -d
docker compose -f docker-compose.vllm-rerank-0.6b-rtx.yml up -d

# Health
docker inspect --format='{{.State.Health.Status}}' qwen3-embedding-0-6b
```

### Embeddings — `POST /v1/embeddings`

```bash
curl http://localhost:11463/v1/embeddings -H 'Content-Type: application/json' -d '{
  "model": "qwen3-embedding-0.6b",
  "input": "The capital of China is Beijing."
}'
# -> data[0].embedding : 1024 floats
```

**Instructions improve retrieval (~1–5%).** Add a one-line instruct to *queries*
(not documents). The recommended format is:

```
Instruct: {task description}
Query:{query}
```

e.g. `"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:What is the capital of China?"`.
Smoke-tested separation: cos(query, relevant)=0.76 vs cos(query, irrelevant)=0.17.

### Reranking — `POST /rerank` (Cohere-style) or `POST /v1/score`

```bash
# /rerank: returns results sorted by relevance_score
curl http://localhost:11466/rerank -H 'Content-Type: application/json' -d '{
  "model": "qwen3-reranker-0.6b",
  "query": "What is the capital of China?",
  "documents": ["Gravity attracts two bodies.", "The capital of China is Beijing.", "Paris is the capital of France."]
}'

# /v1/score: score text_1 against each of text_2
curl http://localhost:11466/v1/score -H 'Content-Type: application/json' -d '{
  "model": "qwen3-reranker-0.6b",
  "text_1": "What is the capital of China?",
  "text_2": ["The capital of China is Beijing.", "Gravity attracts two bodies."]
}'
```

Smoke-tested ranking: Beijing 0.976 > Paris 0.809 > gravity 0.719 — correct order
(the absolute scores are uncalibrated; use them only for *ranking*).

## Service Variants

| File | Model | Runner | gpu-mem-util | max-len | Port |
|------|-------|--------|--------------|---------|------|
| `vllm-embed-0.6b-rtx.yml` | Qwen3-Embedding-0.6B | pooling | 0.05 | 16384 | 11463 |
| `vllm-embed-4b-rtx.yml` | Qwen3-Embedding-4B | pooling | 0.12 | 16384 | 11464 |
| `vllm-embed-8b-rtx.yml` | Qwen3-Embedding-8B | pooling | 0.22 | 16384 | 11465 |
| `vllm-rerank-0.6b-rtx.yml` | Qwen3-Reranker-0.6B | pooling + classify | 0.05 | 16384 | 11466 |
| `vllm-rerank-4b-rtx.yml` | Qwen3-Reranker-4B | pooling + classify | 0.12 | 16384 | 11467 |
| `vllm-rerank-8b-rtx.yml` | Qwen3-Reranker-8B | pooling + classify | 0.22 | 16384 | 11468 |

## Notes

- **vLLM 0.22 flag change**: the old `--task embed` / `--task score` flags were
  **removed**. Embeddings use `--runner pooling`; the reranker also uses
  `--runner pooling` (vLLM auto-resolves `--convert classify`).
- **Reranker requires an architecture override.** Qwen3-Reranker ships as a
  `Qwen3ForCausalLM` that natively scores by comparing "yes"/"no" token logits over
  the full 151669-token vocab — slow and incompatible with the score API. The configs
  pass `--hf-overrides '{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}'`
  to collapse that into a single relevance logit. Do **not** remove this override.
- **MRL (custom embedding dimensions)**: enabled. The embed configs pass
  `--hf-overrides '{"is_matryoshka":true}'`, so you can request a shorter vector
  with the OpenAI `dimensions` field (e.g. `"dimensions": 256`) — useful to cut
  vector-store size/search cost. Without the override vLLM rejects `dimensions`
  with *"does not support matryoshka representation"* (it isn't auto-detected from
  the model config). Verified: `dimensions:256` returns a 256-d vector.
- **Longer context**: raise `--max-model-len` (up to 32K for 0.6B, 40K for 4B/8B) and
  bump `--gpu-memory-utilization` accordingly. At util 0.05 the 0.6B tops out around
  ~26K tokens; the default 16384 leaves comfortable KV headroom.
- **Not gated**: `HF_TOKEN` is accepted but unnecessary — Qwen3-Embedding/Reranker
  are public Apache 2.0.
- **Image**: pinned to `v0.22.0-cu129-ubuntu2404` (no cu130 build exists for v0.22.0).
