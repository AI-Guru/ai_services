# ai_services — operating notes for Claude

Self-hosted LLM inference services. Each model family lives under `models/<family>/`
as a set of `docker-compose.<engine>-<variant>.yml` files that serve an
OpenAI-compatible API. Benchmarks live in `models/shared/` (`test_chat.py`,
`test_tools.py`, `test_scenarios.py`) and results are written up in each family's
`README.md` (+ `comparison*.html`).

## GPU incidents: read `GPU-INCIDENT-RUNBOOK.md` FIRST

If the GPU is wedged, missing, throwing Xids, or a model won't load, go to
**[`GPU-INCIDENT-RUNBOOK.md`](GPU-INCIDENT-RUNBOOK.md)** before doing anything
else. Section 0 is the live-incident procedure; the rest is reference (failure
modes, what evidence is auto-captured and where, analysis queries, baselines,
and a list of traps that have already cost real time).

Two rules from it that matter even outside an incident:

- **Never `sudo reboot` / `sudo shutdown now` while a model is loaded.** Use
  `sudo /home/despara/Development/safe-shutdown.sh [--reboot]`. A plain shutdown
  gives docker ~10s before SIGKILL, and killing vLLM mid-CUDA-op wedges this card
  into a state only a **cold power cycle** clears.
- **Before rebooting to recover a wedged GPU, make sure the crash dump was
  captured** (`ls -lt /var/log/gpu-crash/`). The dump lives in driver memory and
  the recovery reboot destroys it. Capture is automatic via
  `gpu-xid-watch.service`, but verify — losing it is how the 2026-08-14 Xid 79
  ended up uninvestigable.

## Git: commit straight to main — do NOT branch

This is a solo research repo. When asked to commit, commit directly on `main`;
do not create a feature branch and do not open a PR (overriding any default
"branch first" behavior). Still commit only the relevant files — never blanket
`git add -A`, since the tree usually carries unrelated work-in-progress changes.

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

## Currently serving: Qwen3.8-Flash-Next on port 11480

`models/qwen3.8-flash-next/docker-compose.llama-177b-q4-mtp-rtx.yml` — UD-Q4_K_XL
+ vision + MTP, 262144 context, **91.7 of 95.6 GiB VRAM** on build 11112
(image `llama.cpp-qwen4exp:mtp-127368d8`, master 2026-09-22 + PR #28243). ~121 tok/s
with MTP, ~98 without. Served as `qwen3.8-flash-next`. Full detail in that
family's README, including how to A/B a new build (`benchmarks/depth_ab.py`:
compare ms per forward pass, never tok/s, under speculation).

This one is unlike the rest of the repo in several ways; read these before
touching it.

- **It only fits because llama.cpp keeps the 26.8 GiB n-gram/PLE table
  host-side.** `--lazy-mode on` then streams its rows from SSD instead of trying
  to cache them in 30 GiB of RAM. **Never add `--no-mmap` or `-lm none`** — the
  qwen3.6 Spark composes use `--no-mmap`, and copying that here breaks the model.
- **MTP needs a fork.** Mainline llama.cpp *accepts* `--spec-type draft-mtp` and
  then silently ignores the draft head — baseline speed, no error. The build is
  #28243 (`danielhanchen/llama.cpp` `qwen4exp/mtp`) merged into master, pinned by
  SHA — the recipe is in the compose header. Always confirm speculation
  actually ran: `docker logs … | grep 'draft acceptance'`. No line = no MTP.
- **`--spec-draft-n-max 3`, measured — not the 5 Unsloth's guide recommends.**
  5 is slower than 2 here and 8 is slower than no MTP at all.
- **3.9 GiB of headroom** (2.4 before build 10945). The tightest config in the repo.
  `--spec-draft-n-max 7` would eat 1.8 GiB of it and is slower anyway (measured). Draft-head buffers
  are not counted in llama.cpp's context-fit pass (it logs
  `failed to measure the memory of the extra model, fitting without it`), so
  projections run ~2 GiB light. Do not stack anything else on this card.

### Two measurement traps this model sets

Both cost real time here; both produce plausible-looking numbers rather than errors.

- **At temp 1.0 the model sometimes emits EOS immediately** on synthetic prompts
  (`predicted_n = 1`). Averaged into a throughput cell that silently becomes a
  wrong number — `(75.08 + 0.00)/2` once read as 37.54 tok/s. Use `ignore_eos`
  for throughput runs **and assert `n_gen`**.
- **But `ignore_eos` destroys speculative-decoding measurements.** Forced
  generation past the natural end degenerates into repetition, which a draft head
  predicts perfectly: acceptance read 94.5 % on filler text and **100 % on a
  truncated book, where the model emitted nothing but `0` characters**. Both
  inflated MTP throughput at 130K to ~110 tok/s. Speculation must be benchmarked
  on a real prompt without `ignore_eos`; realistic acceptance is 55-70 %.

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
