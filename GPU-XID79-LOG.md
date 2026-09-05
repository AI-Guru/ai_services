# Xid 79 investigation log

Running log of the hunt for the cause of `Xid 79, GPU has fallen off the bus` on
the RTX PRO 6000 Blackwell. **Status: unresolved. Leading hypothesis is power
delivery.**

This is the *investigation* record — what we tried, what it showed, what is dead
and what is still alive. For "the GPU is wedged, what do I do right now", see
[`GPU-INCIDENT-RUNBOOK.md`](GPU-INCIDENT-RUNBOOK.md).

**Append, don't rewrite.** Entries are dated and immutable; when a conclusion is
overturned, add a new entry saying so rather than editing the old one. The
hypothesis ledger is the one section that gets updated in place.

---

## Hypothesis ledger

| # | Hypothesis | Status | Basis |
|---|---|---|---|
| H1 | PCIe ASPM L1 entry/exit failure | **DEAD** | Fault #2 occurred with ASPM verifiably disabled (E3, I2) |
| H2 | Software stack / vLLM bug | **DEAD** | Fault #1 was vLLM, fault #2 was llama.cpp (I2) |
| H3 | Thermal | **DEAD** | 80 °C at fault #2 — identical to the peak of a soak that ran clean (E1, E4, I2) |
| H4 | Progressive link degradation (signal integrity) | **DEAD** | Zero AER correctable errors before either fault; no precursor at all (I1, I2) |
| H5 | **Power delivery** (PSU / 12V-2x6 / VRM under transient) | **ALIVE — leading, now with physical evidence** | Fault #2 at 598 W sustained, 603 W peak vs a 600 W cap; ~1h at ≤475 W has never faulted (I2, E1, E4). White flaking at the 12V-2x6 retention latch (P1) — possible partial seating |
| H6 | PCIe retimer / Gen5 marginality | **ALIVE — weak** | Link reports `Retimer+ 2Retimers+`, but H4's absence of AER argues against |
| H7 | Driver / GSP firmware bug | **ALIVE — untested** | Both faults on driver 610.43.02; no way to test without a version change |
| H8 | P-state link retraining (Gen5↔Gen1) | **ALIVE — untested** | 22 retrains per 30 min survive fine, but the fault could be a rare loss |

---

## Incidents

### I1 — 2026-08-14 18:36:05 (first known)

| | |
|---|---|
| Uptime at fault | 10.8 h (boot 07:48) |
| Workload | vLLM, container started 18:04:19 — ~32 min before the fault |
| Power at fault | **unknown** — no telemetry existed |
| ASPM | enabled (`LnkCtl: ASPM L1 Enabled`) |
| Recovery | Xid 154 → `0x2 (OS Reboot)`; warm reboot at 18:38:55 cleared it fully |
| Crash dump | **lost** — nothing was capturing, the recovery reboot destroyed it |

Evidence: no PCIe AER, no MCE, no thermal throttle. ~250k NVRM lines in 6 s
(journald did *not* rate-limit — the log is complete). First Xid 79 in journals
going back to 2026-07-31; all prior Xids were benign 13/31 from `python3`.

Two things were discovered while investigating rather than caused by the fault:
`nvidia-gpu-exporter` was **absent entirely**, so Grafana had been blind for an
unknown period; and nothing was capturing crash dumps.

### I2 — 2026-09-05 16:59:15 (recurrence, 22 days later)

| | |
|---|---|
| Uptime at fault | 8.5 h (boot 08:29:03, kernel 7.0.0-30 — was -28 for I1) |
| Workload | llama.cpp `llama-server`, `llama.cpp-qwen4exp:master`, exited 139 |
| Power at fault | **598 W sustained, 603.12 W peak in the preceding hour (cap 600 W)** |
| Temp / util / fan | 80 °C, 99 %, P1, fan 69 % |
| ASPM | **disabled** — `pcie_aspm.policy=performance` active, `_OSC: OS now controls [... AER PCIeCapability ...]` |
| Xid line | `79, pid=118717, name=nvidia-smi, GPU has fallen off the bus` |
| Recovery | Xid 154 → `0x2 (OS Reboot)`; rebooted 17:01, clean |
| Crash dump | **captured automatically** → `/var/log/gpu-crash/20260905-165915-xid79-154` |

Power run-up from Prometheus:

```
16:57:15   17.5 W   (idle)
16:57:30  282.7 W
16:58:00  319.4 W
16:58:15  388.1 W
16:58:30  451.5 W
16:58:45  594.8 W
16:59:00  598.4 W
16:59:15  541.0 W   <- Xid 79
```

`pcie_link_width` was a steady 16 through 16:59:15 and read 63 (sysfs "link
down") at 16:59:20 — the drop is inside one 5 s scrape. Zero AER correctable
errors throughout. The kernel log is **completely silent for 1h44m** before the
Xid. `lspci` at capture time shows `!!! Unknown header type 7f` — all-1s config
space, the device is simply gone.

Board rails showed no measurable droop, but only oscillated across the 8 mV ADC
step, which is ~106 mV referred to 12 V — below the resolution needed to see a
realistic sag. **Inconclusive, not negative.**

> **The `pid=nvidia-smi` attribution is not causal.** It names the process
> holding an RM call when the drop was detected. Two exporters poll `nvidia-smi`
> every 5 s, so it is very likely to be whatever is mid-call at any moment.

---

## Physical inspection

### P1 — 2026-09-05 — 12V-2x6 connector reseated

Connector removed, inspected and reseated. **No discoloration or melting on the
contacts** — so no evidence of the classic terminal-overheating failure yet.

**But: small white flakes at the retention latch ("clicky lever"), described as
white dust or paper.**

This matters. The likeliest benign explanations are dust, paper fibre, or mould
release residue. The concerning explanation is **stress whitening / crazing of
the latch polymer** — the housings are typically glass-filled nylon or PBT, which
whitens and can shed at flex points under mechanical fatigue or heat. The latch
is precisely such a flex point.

Why the distinction is critical: a weakened or cracked latch may not hold the
plug **fully mated**. Partial seating on a 12V-2x6 carrying ~50 A at 600 W is the
documented root cause of connector melting, and it produces exactly the fault
signature seen here — an instantaneous power interruption with no AER, no thermal
event, no software correlate, and nothing a driver could log.

**This is the best physical evidence to date and it is consistent with H5.** It
is not confirmation: nobody has yet established that the flakes are connector
polymer rather than debris.

Discriminating tests (pending):
- Is there a matching whitened/crazed area on the latch hinge under bright light?
- Does the flaking recur after cleaning? Recurrence implies shedding.
- Does the latch still click firmly and resist a gentle tug?
- Flake texture: paper/dust smears and disintegrates; polymer is harder, waxy.
- Check the PSU-side connector and whether this is a native cable or an adapter
  (adapters fail far more often).

**Consequence for the plan:** deliberately driving the card to ~600 W to force a
reproduction — previously listed as a way to shorten the feedback loop — is now
**withdrawn as unsafe**. If the latch is compromised, that experiment risks an
actual connector melt rather than a clean bus drop.

## Experiments

### E1 — 2026-08-15 — baseline soak, ASPM ON

`gpu-loadtest.py`, 30 min, 8 workers, 512-token generations, duty-cycled
120 s/60 s. **1594 requests, 0 failures**, 816,128 tokens. Mean 330.7 W /
**peak 474.9 W**. Mean 69.2 °C / peak 79 °C. Link `{2.5: 67, 5.0: 3, 32.0: 287}`,
**22 transitions**, all successful. AER +0, brake +0, thermal +0.

**Result: null.** No anomaly of any kind.

### E2 — 2026-08-15 — `pcie_aspm=off` — FAILED, and harmful

Did not disable ASPM. It disables Linux's ASPM *driver*, so the kernel declined
PCIe control at `_OSC`; firmware kept ownership, BIOS-configured ASPM L1 survived
untouched, **and the OS lost AER/DPC/hotplug/PME**. Reverted.

### E3 — 2026-08-15 — `pcie_aspm.policy=performance` — SUCCEEDED

`LnkCtl: ASPM Disabled`, `_OSC` ownership restored. Note the dot:
`pcie_aspm.policy=` is a module parameter that keeps the driver; `pcie_aspm=` is
the driver on/off switch.

Also disproved a predicted observable: the idle Gen5→Gen1 downtraining is the
**driver's P-state link-speed scaling**, not ASPM, and persists with ASPM off.

### E4 — 2026-08-15 — repeat soak, ASPM OFF

Identical parameters. 1582 req / 0 fail, mean 326.9 W / **peak 471.9 W**, peak
80 °C, link `{2.5: 66, 5.0: 3, 32.0: 287}`, **22 transitions again**, AER +0.
Throughput within 0.8 % of E1.

**Result: statistically identical to E1.** Disabling ASPM changes nothing
measurable — costs nothing to keep, proves nothing.

### E5 — 2026-09-05 — power cap at 450 W

```bash
sudo gpu-ops/gpu-power-cap.sh 450
```

Chosen over 500 W because 475 W is the highest level soak-tested clean (E1/E4) —
500 W would be untested territory — and because 450 W is a defined 12V-2x6
sense-pin tier. Takes connector current from ~43.8 A to ~31.2 A, i.e. **~51 % of
the contact I²R heating**.

Motivation strengthened by P1: this is now protecting a physically suspect
connector, not just probing a hypothesis.

Success criterion is absence of recurrence — **slow and weak evidence** given the
22-day gap between I1 and I2. Release conditions and the lever are documented in
[`GPU-INCIDENT-RUNBOOK.md` section 8](GPU-INCIDENT-RUNBOOK.md#8-mitigations-in-force--and-how-to-release-them).

---

### E6 — 2026-09-05 — throughput cost of the 450 W cap (single stream)

`test_chat.py`, qwen3.8-flash-next (llama.cpp q4) on :11480, 3 runs + warmup:

| | |
|---|---|
| Throughput | **96.5 tok/s** avg (95.3 / 96.8 / 97.3) |
| TTFT | 3617 ms avg — dominated by reasoning (e.g. 2693 ms of a 2833 ms TTFT) |
| Peak power | **348.0 W**, mean-busy 328.2 W, peak SM 2827 MHz, peak 63 C |
| Samples within 5 % of the 450 W cap | **0** |

**The cap is not binding for single-stream chat.** ~102 W of headroom to the
limit, so this workload is unaffected and the cap costs nothing here.

**This does NOT show the cap is free in general.** Single-stream inference peaks
around 348 W; the fault regime is ~600 W, which is reached under concurrency. The
cost lands entirely on high-concurrency/batch serving, and is unmeasured.

Note: `--no-think` was a **no-op** — the model still emitted reasoning
(`think 2693 ms (1231 chars)`), so the tok/s figure includes thinking tokens.
Same pattern as other families in this repo.

## What the investigation itself got wrong

Worth keeping, because these were process failures, not hardware ones.

**The soak never entered the fault regime.** E1 and E4 both peaked at 474.9 W;
I2 faulted at ~600 W — roughly 125 W higher. "The soak was clean" read as
reassurance for three weeks, when it only ever meant "clean at 79 % of the power
where the fault lives." *Check that a reproduction attempt reaches the observed
fault conditions before treating a negative as informative.*

**The decisive metric was silently dark.** `hw_power_braking` — the one signal
that would settle H5 — had no data for the entire hour before I2, while the same
exporter's other metrics flowed normally. Cause: `read_nvidia_events()` never
checked `nvidia-smi`'s return code, so a failing call produced empty output, an
empty parse, and `gpu_events_exporter_up = 1`. Coverage was only 57 hourly points
across 30 days with 11 gaps > 2 h. The exporter had also crashed 8× on
`BrokenPipeError`. Both fixed 2026-09-05; `gpu_events_clock_counters_parsed` now
exposes the silent-empty case. **Alert on that being 0, not just on `up`.**

**A mitigation was applied on mechanism fit and treated as progress.** H1 was
plausible and cheap to act on, but E4 showed no measurable change and I2 killed
it outright. No harm done — it costs nothing to keep — but three weeks passed
believing the most likely cause was handled.

---

## Open questions

0. **Are the white flakes connector polymer or debris?** The single highest-value
   open question — see P1. A photograph under good light would likely settle it.
1. **Was the power brake ever asserted?** Unknown for both faults — the counter
   was dark. Now instrumented; a third occurrence should answer it.
2. **Does the 12 V rail sag under load?** Unmeasurable at 8 mV ADC resolution.
   Would need a clamp meter on the 12V-2x6, or a rail the board actually labels.
3. **Was I1 also at ~600 W?** Unknown — no telemetry. If it was, H5 strengthens
   considerably; there is no way to recover this.
4. **Is it load-transient or sustained-power?** I2 shows both — a 90 s ramp from
   idle to the cap, then ~30 s at ~598 W. Nothing distinguishes them yet.

## Next actions

- [x] E5: capped to 450 W, persisted via `gpu-power-cap.service`
- [ ] Identify the white flakes (P1) — highest-value open question
- [ ] Replace the 12V-2x6 cable if the latch is compromised
- [ ] Alerting — still none. `up{job="nvidia_gpu"} == 0` and
      `gpu_events_clock_counters_parsed == 0` are the two cheap ones
- [x] ~~Re-run the soak at ~600 W to force a reproduction~~ — **WITHDRAWN 2026-09-05
      as unsafe** after P1. Do not drive this card to the cap until the latch is
      cleared.
- [ ] Check for a BIOS newer than 1402 (AGESA PCIe fixes)
- [ ] Consider enabling ECC (~6 % VRAM) for memory-error visibility

## Evidence index

| What | Where |
|---|---|
| Crash dumps | `/var/log/gpu-crash/<timestamp>-xid<codes>/` |
| I2 dump | `/var/log/gpu-crash/20260905-165915-xid79-154` |
| Soak reports + raw CSV | `~/gpu-loadtests/` |
| Metrics (30 d retention) | Prometheus `localhost:9090`, Grafana `localhost:3000` |
| Tooling | [`gpu-ops/`](gpu-ops/) |
| Operational procedure | [`GPU-INCIDENT-RUNBOOK.md`](GPU-INCIDENT-RUNBOOK.md) |
