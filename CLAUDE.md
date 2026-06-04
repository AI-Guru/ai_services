# ai_services — operating notes for Claude

Self-hosted LLM inference services. Each model family lives under `models/<family>/`
as a set of `docker-compose.<engine>-<variant>.yml` files that serve an
OpenAI-compatible API. Benchmarks live in `models/shared/` (`test_chat.py`,
`test_tools.py`, `test_scenarios.py`) and results are written up in each family's
`README.md` (+ `comparison*.html`).

## Watch for crash-looping containers

Every compose file uses `restart: unless-stopped`. **A container that fails to
start will silently crash-loop forever** — `docker inspect` reports
`State.Running=true` during each restart attempt, so a naive
"wait until Running==false" or "wait until healthy" loop will hang until timeout
instead of reporting the failure.

When you launch or wait on a container, treat it as crashed if **any** of these hold:
- `State.Status` is `restarting` or `exited`
- `RestartCount` has increased above the value you captured at launch
- `State.Health.Status` is `unhealthy`

Reference waiter (fails fast on a crash-loop instead of hanging):

```bash
C=<container-name>
base=$(docker inspect --format='{{.RestartCount}}' "$C")
while true; do
  st=$(docker inspect --format='{{.State.Status}}'        "$C" 2>/dev/null)
  h=$( docker inspect --format='{{.State.Health.Status}}' "$C" 2>/dev/null)
  rc=$(docker inspect --format='{{.RestartCount}}'        "$C" 2>/dev/null)
  [ -z "$st" ] && { echo "GONE"; break; }
  [ "$h" = healthy ] && { echo "HEALTHY"; break; }
  if [ "$st" = restarting ] || [ "$st" = exited ] || [ "${rc:-0}" -gt "${base:-0}" ]; then
    echo "CRASH-LOOP: status=$st RestartCount=$rc"
    docker logs "$C" 2>&1 | grep -iE 'error|valueerror|runtimeerror|assert|no module or parameter|out of memory' | tail -10
    break
  fi
  sleep 10
done
```

When a model fails to load, get the real cause from
`docker logs <container> | grep -iE 'error|traceback|valueerror|assert|out of memory'`
(logs persist across restarts for the same container). Then **stop the
crash-looping container** (`docker compose -f <file> down`) so it stops burning
GPU/CPU on restart attempts. Don't leave a known-broken container looping.

## GPU is a single shared card

One RTX PRO 6000 Blackwell (96 GB GDDR7, SM_120). Models are benchmarked
**one at a time** — vLLM grabs `--gpu-memory-utilization 0.90` of the whole card,
so a second model won't fit alongside a running one. Before launching:
- `nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader`
- `docker ps` — a model may already be resident (and wired into LibreChat on
  its port). Stop it first (`docker stop <name>`), run your benchmark, then
  `docker start <name>` to restore it. Confirm before stopping a live-served model.

## Benchmark convention

Match the existing READMEs: `python3 test_chat.py --base-url http://localhost:<port>/v1
--model <served-name> --runs 3 --warmup --no-think`. Report avg tok/s and TTFT
(note the warm-run TTFT separately from the cold first run).

## Compose / checkpoint conventions

- Filenames: `docker-compose.<engine>-<size><-quant><-feature>-<hw>.yml`
  (engine: `vllm`/`llama`/`sglang`/`trtllm`; quant: `fp8`/`nvfp4`/`bf16`/`q4`/`q8`;
  hw: `rtx`/`spark`/`vulkan`/`rocm`, omitted = generic CUDA).
- Prefer **first-party checkpoints** (Google base, RedHatAI, NVIDIA). If only a
  third-party community quant exists, **say so explicitly** in the compose header
  and the README row — community quants are unvetted and sometimes fail to load
  (e.g. extra/renamed weight keys the vLLM build doesn't expect).
- vLLM Gemma 4 images are arch-specific: the older `gemma4-0505-cu130` image
  serves `model_type: gemma4` checkpoints (26B, 31B); the newer
  `gemma4-unified-x86_64-cu130` image is required for `model_type: gemma4_unified`
  checkpoints (12B). Check `config.json` `model_type` before picking the image.
- Secrets: `HF_TOKEN` lives in each family's `.env` (gitignored, not readable).
  Never print or commit it.
