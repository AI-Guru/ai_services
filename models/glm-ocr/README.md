# GLM-OCR (zai-org / Z.ai)

0.9B document-parsing vision-LM. CogViT vision encoder (~0.4B) + token-downsampling
connector + GLM-0.5B decoder. Page image → Markdown + JSON (layout bboxes, reading
order), with tables (HTML/TEDS), formulas (LaTeX), charts, seals, and handwritten KIE.

- **Model:** [`zai-org/GLM-OCR`](https://huggingface.co/zai-org/GLM-OCR) · [repo](https://github.com/zai-org/GLM-OCR)
- **Released:** ~2026-03-11 · **License:** weights MIT / code Apache-2.0
- **Reported accuracy** (Z.ai figures — verify against the card): OmniDocBench v1.5
  **94.62** (claimed #1 open-or-closed), OCRBench-Text 94.0, UniMERNet 96.5, olmOCR-bench 75.2
- **Headline speed mechanism:** native Multi-Token-Prediction (MTP) head reused as a
  self-speculative drafter — supported first-party on **both** vLLM and SGLang.

## Compose files

| File | Engine | MTP | Endpoint | Served name |
|------|--------|-----|----------|-------------|
| `docker-compose.sglang-glm-ocr-mtp-rtx.yml` | SGLang (`lmsysorg/sglang:latest`) | NEXTN | `:11438/v1` | `glm-ocr` |
| `docker-compose.vllm-glm-ocr-mtp-rtx.yml`   | vLLM (`v0.19.0-cu130`)             | mtp   | `:8010/v1` | `glm-ocr` |

```bash
# SGLang + MTP
docker compose --env-file ../../.env -f docker-compose.sglang-glm-ocr-mtp-rtx.yml up -d
# vLLM + MTP
docker compose --env-file ../../.env -f docker-compose.vllm-glm-ocr-mtp-rtx.yml up -d
```

Weights are ~1.8 GiB (BF16), trivial on the 96 GB card — utilization is set low
(`0.20`); raise it for more concurrent KV / a larger running-request pool.

## Benchmarking

For decode-speed / TTFT (spec-on vs spec-off), the shared harness works as usual:

```bash
python3 ../shared/test_chat.py --base-url http://localhost:11438/v1 --model glm-ocr --runs 3 --warmup
```

For a *real* OCR throughput number, send document images (not a text prompt) and
measure pages/s — the pure-text path understates the vision cost. To isolate the MTP
gain, run each engine twice: once as-is, once with the `--speculative-*` /
`--speculative-config` flags removed.

## Caveats (see the compose headers)

- The exact NEXTN step/topk/draft-token values and the vLLM `{"method":"mtp"}` string
  were taken from docs that had minor conflicts — **confirm against the live README**
  before trusting a benchmark.
- If a container crash-loops on an unknown-architecture / `model_type` error, the
  pinned image predates native GLM-OCR registration (SGLang `glm_ocr.py` merged
  2026-01-27) — bump the image tag or add `--trust-remote-code`. Then stop the
  crash-looping container so it isn't burning GPU on restarts.

## Related

DeepSeek-OCR (the throughput/optical-compression counterpart) already lives at
[`../deepseek-ocr/docker-compose.vllm-deepseek-ocr-rtx.yml`](../deepseek-ocr/docker-compose.vllm-deepseek-ocr-rtx.yml).
