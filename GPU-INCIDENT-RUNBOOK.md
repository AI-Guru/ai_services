# GPU incident runbook — RTX PRO 6000 Blackwell

Operational guide for when the GPU wedges, disappears, or misbehaves. Written
after the **2026-08-14 Xid 79** incident, whose crash dump was lost because
nothing was set up to capture it. Everything here exists so that does not repeat.

**Read section 0 first during an incident. Everything after it is reference.**

---

## 0. The GPU is wedged RIGHT NOW

### 0.1 Triage — run this first

```bash
nvidia-smi                                   # does the device even exist?
journalctl -k -b | grep -E 'NVRM: Xid'       # what did the driver say?
docker ps -a --format '{{.Names}}\t{{.Status}}'
ls -lt /var/log/gpu-crash/ | head            # was a dump auto-captured?
```

### 0.2 Decide which failure you have

| What you see | Mode | Go to |
|---|---|---|
| `Unable to determine the device handle for GPU0 … Unknown Error` / `No devices were found`, and `Xid 79, GPU has fallen off the bus` | **Bus drop** | [1.1](#11-xid-79--gpu-fell-off-the-bus) |
| `nvidia-smi` runs but shows ghost memory with "No running processes", `ERR!` in fan/util fields; container won't stop ("PID … is zombie") | **SIGKILL wedge** | [1.2](#12-xid-154-wedge-after-a-sigkilled-cuda-process) |
| Device fine, but a model container restarts forever | **Crash loop** | [1.3](#13-crash-looping-container) |
| Device fine, container exits once with a Python traceback | **Ordinary load failure** | [1.4](#14-ordinary-model-load-failure) |

### 0.3 Before you reboot — CAPTURE THE EVIDENCE

The GPU crash dump lives in driver memory and **is destroyed by the reboot that
recovers the card**. It survives the fault itself and stays readable until the
kernel module unloads, so you have as long as you like — but only until reboot.

This should be automatic (`gpu-xid-watch.service` fires within seconds, and
`safe-shutdown.sh` collects as a backstop). Confirm, and force it if not:

```bash
ls -lt /var/log/gpu-crash/                                   # expect a fresh dir
sudo /home/despara/Development/gpu-crash-collect.sh --force --full   # if not
```

### 0.4 Reboot safely — NEVER plain `reboot`

```bash
sudo /home/despara/Development/safe-shutdown.sh --reboot
```

`sudo reboot` lets systemd stop docker, which force-kills containers after ~10s.
Killing vLLM mid-CUDA-op is exactly what wedges this card into a state only a
cold power cycle clears. The script stops GPU containers with a 90s grace,
verifies the GPU drained, and only then reboots.

---

## 1. Failure modes

### 1.1 Xid 79 — GPU fell off the bus

**Signature** (seen 2026-08-14 and 2026-09-05):

```
NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus.
NVRM: Xid (PCI:0000:01:00): 154, GPU recovery action changed from 0x0 (None) to 0x2 (OS Reboot)
nvidia-modeset: ERROR: GPU:0: Error while waiting for GPU progress   (repeats every 5s)
```

The card stops answering on PCIe. The display freezes and `nvidia-smi` reports
the device is simply gone. Expect a huge NVRM cascade (~250k lines in 6s).

**Recovery: read the Xid 154 recovery-action code — it tells you what is needed.**

| Recovery action | Meaning | Fix |
|---|---|---|
| `0x2 (OS Reboot)` | warm reboot suffices | `safe-shutdown.sh --reboot` |
| no 154 line / reboot doesn't clear it | full-chip reset needed | **cold power cycle**: power off, cut mains ~10s, boot |

This card returns **"Not Supported"** for `nvidia-smi --gpu-reset` — there is no
software reset path.

**Cause: unknown.** See [section 6](#6-incident-history-and-what-we-know).

### 1.2 Xid 154 wedge after a SIGKILLed CUDA process

Distinct from 1.1. Triggered by killing a process holding CUDA contexts mid-op —
classically `sudo shutdown now` while vLLM is running.

**Signature:** ghost memory with no processes listed, `ERR!` fields, unkillable
D/Z-state process, `docker stop` reports "PID … is zombie and can not be killed",
a plain shutdown hangs ~3 min and the board may power itself back on.

**Recovery: cold power cycle.** A warm reboot has been observed *not* to clear
this one. Prevent it by always using `safe-shutdown.sh`.

### 1.3 Crash-looping container

Every model compose uses `restart: unless-stopped`, so **a container that fails
to start crash-loops forever**. `docker inspect` reports `State.Running=true`
during each restart attempt, so naive "wait until healthy" loops hang until
timeout instead of reporting the failure.

Treat as crashed if **any** hold: `State.Status` is `restarting`/`exited`;
`RestartCount` rose above its launch value; `State.Health.Status` is `unhealthy`.
See `CLAUDE.md` for the reference waiter loop.

Get the real cause, then **stop the container** so it stops burning GPU on
restarts:

```bash
docker logs <container> 2>&1 | grep -iE 'error|traceback|valueerror|assert|out of memory'
docker compose -f <file> down
```

### 1.4 Ordinary model load failure

Not a GPU fault. Usual causes: wrong vLLM image for the checkpoint's
`model_type`, community quant with unexpected weight keys, insufficient VRAM.
See `CLAUDE.md` → compose/checkpoint conventions.

---

## 2. What is captured automatically

### 2.1 The capture chain

| Component | Trigger | Notes |
|---|---|---|
| `gpu-xid-watch.service` | follows the kernel log, fires within seconds of a severe Xid | runs `gpu-crash-collect.sh --full` |
| `safe-shutdown.sh` step 1 | every shutdown/reboot | backstop, collects before teardown destroys the dump |
| latch `/run/gpu-crash-collected` | — | one collection per boot; `/run` is tmpfs so it resets on boot |

**Severity filter:** collects for Xid `48 63 64 74 79 92 93 94 95 154`.
Deliberately **excludes 13 and 31** — routine application MMU faults and
graphics exceptions (this box logged ~10 from `python3` across Aug 11-12 alone).

### 2.2 What lands in `/var/log/gpu-crash/<timestamp>-xid<codes>/`

| File | What it is |
|---|---|
| `nvidia-bug-report-safe.gz` | `--safe-mode` report; the artifact NVIDIA support/RMA asks for |
| `nvidia-bug-report-full.gz` | richer report (watcher path only, not on shutdown) |
| `nvidia-debugdump.zip` | targeted GPU state dump |
| `kernel-current-boot.log.gz` | full kernel log for the boot |
| `summary.txt` | Xid codes + all Xid lines + driver version |
| `manifest.txt` | per-step exit code, timeout, and payload size |
| `nvidia-smi-q.out`, `lspci.out`, `docker-ps.out` | state snapshots |

**Always read `manifest.txt` first.** `rc=124` means a step hit its timeout;
`SUSPECT_EMPTY` or `MISSING` means an artifact is not trustworthy.

**Expect degraded content after a bus drop.** The driver's own dump attempt
partially fails when the card is gone (`prbEncStartAlloc: Can't allocate memory`,
`GPU lost from the bus`). The host-side half — PCIe config space, driver state,
kernel log — is intact and is what an RMA conversation runs on.

### 2.3 Manual capture

```bash
sudo /home/despara/Development/gpu-crash-collect.sh            # if a severe Xid is in this boot's log
sudo /home/despara/Development/gpu-crash-collect.sh --force --full   # unconditional
sudo /home/despara/Development/gpu-crash-collect.sh --check    # exit 0 if a severe Xid is present
```

---

## 3. Post-incident analysis

### 3.1 Checklist

1. `ls -lt /var/log/gpu-crash/` — grab the newest dir, read `summary.txt` then `manifest.txt`.
2. `journalctl -k -b -1 | grep -E 'NVRM: Xid'` — the previous boot is where the fault is.
3. Find the **first** error, not the cascade. The cascade is ~250k lines of aftermath.
4. Check for corroborating hardware evidence — usually absent, which is itself informative:
   ```bash
   journalctl -k -b -1 | grep -iE 'aer|pcieport|machine check|mce:|thermal|throttl'
   ```
5. Query Prometheus for the minutes *before* the fault (see 3.2).
6. Compare against the stored baselines (section 5).

### 3.2 Prometheus queries that matter

Grafana `localhost:3000`, Prometheus `localhost:9090`, 30-day retention.

```promql
# Did the PSU assert the power-brake pin? Non-zero => power delivery implicated.
increase(nvidia_clocks_event_seconds_total{reason="hw_power_braking"}[6h])

# PCIe correctable errors — the direct probe for link instability.
increase(pcie_aer_errors_total{type="TOTAL_ERR_COR"}[6h])

# Link speed / width over time — did it downgrade or fail to return?
pcie_link_speed_gts
pcie_link_width

# Thermal
increase(nvidia_clocks_event_seconds_total{reason="hw_thermal_slowdown"}[6h])
nvidia_smi_temperature_gpu

# Power draw against the 600 W cap
nvidia_smi_power_draw_watts

# Exporters alive? A card that falls off the bus takes nvidia_gpu down with it.
up{job=~"nvidia_gpu|gpu_events"}
```

> **Why cumulative counters matter.** `nvidia_clocks_event_seconds_total` accrues
> in microseconds, so a 50 ms brake event is retained permanently. A 5s sample of
> `power.draw` would miss it entirely. For transients, prefer the counter.

### 3.3 Reproduce under instrumentation

```bash
python3 /home/despara/Development/gpu-loadtest.py \
    --duration 1800 --concurrency 8 --label <what-changed>
```

Duty-cycles load (120s on / 60s off) and reports AER deltas, brake counters,
link distribution, rail correlation, power and thermals. Reports land in
`~/gpu-loadtests/`. Note it needs the model container serving — several composes
use `restart: "no"` and will **not** come back after a reboot on their own.

---

## 4. Instrumentation inventory

### 4.1 Health check — run after any reboot

```bash
docker ps --filter name=exporter --format '{{.Names}}|{{.Status}}'   # expect 2
systemctl is-active gpu-xid-watch.service                            # expect active
lsmod | grep -c nct6775                                              # expect >0
curl -s localhost:9090/api/v1/targets | grep -c '"health":"up"'      # expect 2
cat /proc/cmdline                                                    # expect pcie_aspm.policy=performance
```

### 4.2 Components

| Path | Role |
|---|---|
| `gpu-dashboard/docker-compose.yml` | Grafana + Prometheus + both exporters |
| `gpu-dashboard/gpu-events-exporter.py` | :9836 — cumulative event counters, PCIe AER, link state, board rails |
| `nvidia-gpu-exporter` (container) | :9835 — instantaneous nvidia-smi metrics |
| `gpu-ops/safe-shutdown.sh` | graceful GPU release + crash-capture backstop |
| `gpu-ops/gpu-crash-collect.sh` | the collector |
| `gpu-ops/gpu-xid-watch.sh` | journal follower |
| `gpu-ops/gpu-crash-install.sh` | installs + verifies the capture chain |
| `gpu-ops/gpu-loadtest.py` | instrumented soak harness |
| `gpu-ops/gpu-aspm-toggle.sh` | ASPM kernel-parameter management |
| `gpu-ops/gpu-sensors-probe.sh` | loads `nct6775`, exposes board rails |
| `gpu-ops/systemd/` | unit file sources (installed copies live in `/etc/systemd/system/`) |
| `/etc/systemd/system/gpu-xid-watch.service` | the watcher unit |
| `/etc/systemd/system/gpu-safe-shutdown.service` | runs safe-shutdown at shutdown (`TimeoutStopSec=360`) |
| `/etc/modules-load.d/nct6775.conf` | persists the sensor module |

> **All scripts live in [`gpu-ops/`](gpu-ops/) and are version controlled.** The
> historical `/home/despara/Development/<script>` paths — which the systemd units,
> the scripts' cross-references, and this runbook all still use — are now symlinks
> into `gpu-ops/`. Edit in the repo; the live tool changes immediately.
>
> **If this repo is ever moved or renamed the symlinks break and the capture chain
> silently stops.** Re-point them with the loop in
> [`gpu-ops/README.md`](gpu-ops/README.md), and verify with §4.1.

### 4.3 Known gap

**There is still no alerting.** Prometheus has no `rule_files` and Grafana has no
alert rules, so nothing notifies you. The highest-value rule is trivial — when
the card falls off the bus, `nvidia-smi` fails and the exporter target drops:

```promql
up{job="nvidia_gpu"} == 0
```

---

## 5. Reference baselines

Two 30-minute soaks, 8 workers, 512-token generations, 120s/60s duty cycle,
Qwen3.8-27B NVFP4. Use these to judge whether a future run is abnormal.

| Metric | 2026-08-15 ASPM **on** | 2026-08-15 ASPM **off** |
|---|---|---|
| requests / failures | 1594 / 0 | 1582 / 0 |
| completion tokens | 816,128 | 809,984 |
| mean / peak power | 330.7 W / 474.9 W | 326.9 W / 471.9 W |
| mean / peak temp | 69.2 °C / 79 °C | 68.7 °C / 80 °C |
| link speed distribution | `{2.5: 67, 5.0: 3, 32.0: 287}` | `{2.5: 66, 5.0: 3, 32.0: 287}` |
| link transitions | 22 | 22 |
| PCIe correctable errors | +0 | +0 |
| HW power braking | +0 | +0 |
| GPU-invisible samples | 0 | 0 |

Under load the link holds **32 GT/s x16 in 100% of busy samples**. Idle drops to
Gen1 are normal.

---

## 6. Incident history and what we know

### Occurrences

| When | Uptime at fault | Engine | Peak power | ASPM | Dump captured |
|---|---|---|---|---|---|
| 2026-08-14 18:36:05 | 10.8 h | vLLM | not recorded (no telemetry) | **enabled** | no — lost to the reboot |
| 2026-09-05 16:59:15 | 8.5 h | llama.cpp | **603 W** (cap 600 W) | **disabled** | yes, automatically |

Exactly two in all retained journals, 22 days apart.

### What the second event ruled OUT

- **ASPM is not the cause.** The faulting boot ran with
  `pcie_aspm.policy=performance` and `_OSC: OS now controls [... AER
  PCIeCapability ...]`. The mitigation was active and the card fell off anyway.
- **Not software-stack specific.** August was vLLM, September was llama.cpp.
- **Not thermal.** 80 C at the fault — identical to the peak of a soak that ran
  clean for 30 minutes.
- **Not progressive link degradation.** Zero PCIe correctable errors in the hour
  before, and the kernel log is completely silent for 1h44m before the Xid.

### What it pointed TO: power

The 90 seconds before the fault, from Prometheus:

```
16:57:15   17.5 W   (idle)
16:57:30  282.7 W
16:58:00  319.4 W
16:58:15  388.1 W
16:58:30  451.5 W
16:58:45  594.8 W
16:59:00  598.4 W
16:59:15  541.0 W  <- Xid 79, link width 16 -> 63 within one 5s scrape
peak in the preceding hour: 603.12 W
```

Idle to the 600 W cap in ~90 s, sustained ~598 W, then the bus drop. There is no
precursor of any other kind.

**Crucially, this is a regime our testing never entered.** Both 30-minute soaks
peaked at **474.9 W** — about 125 W short — and both ran clean. That is the most
likely reason the load test never reproduced the fault, and it means "the soak
was clean" was never the reassurance it appeared to be.

Still circumstantial: the `hw_power_braking` counter, which would have settled
it, was silently absent for the whole hour (see the exporter trap in section 7).
That is fixed, so a third occurrence should be decisive.

### Next experiment

Cap the board power and see whether faults stop:

```bash
sudo nvidia-smi -pl 500      # not persistent across reboot
```

This is now a well-motivated test rather than a guess: faults occur at ~600 W,
and an hour of load at <=475 W has never produced one. The cost is throughput on
the workloads that actually reach the cap.

## 6b. Original notes on the 2026-08-14 event

**Timeline.** Boot 07:48. At 18:04:19 a vLLM container started. At **18:36:05**
— 32 minutes in — Xid 79, then Xid 154 with recovery action `0x2 (OS Reboot)`.
Rebooted 18:38:55; the card came back completely clean (`Recovery Action: None`,
no Channel/TPC repair pending, no ECC errors).

**No corroborating evidence.** No PCIe AER errors, no MCE, no thermal throttle.
The card just stopped answering. First Xid 79 in the retained journals (back to
2026-07-31); all prior Xids are benign 13/31 from `python3`.

**No telemetry existed** for the crash window — `nvidia-gpu-exporter` was absent,
so Grafana was blind. That is fixed.

**Leading hypothesis, unproven.** The link reported `ASPM L1 Enabled` on a Gen5
x16 path with `Retimer+ 2Retimers+`, on a consumer ASUS TUF GAMING B850-PLUS WIFI
board driving a 600 W card. ASPM was disabled on 2026-08-15 via
`pcie_aspm.policy=performance` — **a mitigation on mechanism fit, not a proven
fix**. The A/B soak showed literally no difference (22 link transitions both
ways), so it costs nothing to keep and proves nothing.

**Still untested:** eliminating P-state-driven link retraining with
`nvidia-smi --lock-gpu-clocks`, at a large idle-power cost. Only worth it if
evidence points there.

---

## 7. Traps — all found the hard way

| Trap | Reality |
|---|---|
| `pcie_aspm=off` | **Do not use.** Disables Linux's ASPM *driver*, so the kernel declines PCIe control at `_OSC`; firmware keeps ASPM enabled AND the OS loses AER/DPC/hotplug/PME. AER counters then read a healthy-looking zero while measuring nothing. Use `pcie_aspm.policy=performance` — note the dot. |
| "ASPM causes the Gen1 idle drops" | No. That is the NVIDIA driver's P-state link-speed scaling, unchanged by ASPM. Verify ASPM via `lspci -vv` → `LnkCtl: ASPM Disabled`, never via link speed. |
| `restart: unless-stopped` on GPU containers | `safe-shutdown.sh` stops every container with `DeviceRequests`, and Docker remembers a manual stop across reboots — so they never come back. Exporters use `restart: always` for this reason. |
| Editing a bind-mounted single file (`prometheus.yml`) | Replaces the inode and silently breaks the mount. The container keeps serving the old config and `POST /-/reload` reports success while changing nothing. **Recreate the container**, don't reload. |
| `nvidia-debugdump` exit code | Exits **0** while writing a 44-byte empty zip when it lacks permissions. Check payload size, never just `rc`. |
| `nvidia-bug-report.sh --output-file X` | Produces `X.gz`, **not** `X.log.gz`. |
| Board voltage rails | Unlabelled, undivided, ~8 mV ADC steps (~53 mV referred to 12 V). The +12 V rail feeding the card appears unmonitored. `hw_power_braking` is the only real power-delivery probe. |
| `grep -c … \|\| echo 0` | On zero matches grep prints `0` *then* exits 1, so the fallback appends a second line and arithmetic breaks. `grep -c` already prints 0. |
| Exporter "healthy" but emitting nothing | `gpu-events-exporter` did not check `nvidia-smi`'s **return code**: a failing call left stdout empty, the parse produced nothing, and `err` stayed 0 — so `gpu_events_exporter_up` read 1 while no clock-event counters were emitted at all. It was dark for the entire hour before the 2026-09-05 fault, losing `hw_power_braking`. Fixed, plus `gpu_events_clock_counters_parsed` now exposes the condition. **Alert on that being 0, not just on `up`.** |
| Exporter crash-looping silently | Single-threaded `HTTPServer` died with `BrokenPipeError` when a scrape timed out mid-write (8 restarts). Now `ThreadingHTTPServer` with the write wrapped. |
| "The soak was clean" | Both soaks peaked at 474.9 W; the fault happens at ~600 W. A clean run proves nothing if it never entered the regime where the fault lives. Check peak power against the real workload before trusting a negative result. |
| ECC | **Disabled** on this card, so there is no memory-error visibility. Enabling costs ~6% VRAM (~5.8 GB of 96). |
