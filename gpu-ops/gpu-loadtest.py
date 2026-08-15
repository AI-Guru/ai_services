#!/usr/bin/env python3
"""
gpu-loadtest.py — sustained GPU soak with full fault-signal instrumentation.

Purpose is NOT benchmarking. It drives the card hard while recording the exact
signals that discriminate the Xid 79 hypotheses, then reports the deltas:

  - PCIe AER correctable errors     -> link instability (the ASPM/retimer theory)
  - HW Power Braking counter        -> PSU asserting the power-brake pin
  - board voltage rails vs power    -> identifies the +12V rail by anti-correlation,
                                       and shows any droop under load
  - link speed / width over time    -> a link that downgrades or fails to return
  - temperature / throttle reasons  -> the thermal theory

Reads sysfs and nvidia-smi directly rather than scraping the exporter, so it is
independent of it (and cross-validates it).

Usage:
  ./gpu-loadtest.py --duration 1800 --concurrency 8 \
      --url http://localhost:11484/v1 --model qwen3.8-27b --label baseline-aspm-on

Deliberately NOT reusing models/shared/stresstest.sh: that is request-count based
and hardcoded for a different model. A soak needs to be duration-based.
"""
import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

GPU_BDF = os.environ.get("GPU_BDF", "0000:01:00.0")
PCI_ROOT = f"/sys/bus/pci/devices/{GPU_BDF}"

SMI_FIELDS = [
    "power.draw", "temperature.gpu", "fan.speed", "clocks.sm",
    "utilization.gpu", "memory.used", "pcie.link.gen.current",
]

stop_flag = threading.Event()
# Duty-cycle gate. A flat soak makes power a step function, which gives the rail
# correlation exactly one transition to work with — the 60s smoke run scored the
# CPU rails highest simply because they rose once alongside the GPU. Cycling the
# load produces many independent power swings, so a rail feeding the card shows
# repeated anti-correlation that CPU rails do not. It also stresses harder: the
# di/dt of repeated 10W<->500W transitions is what strains PSU and link, more so
# than a constant draw.
load_gate = threading.Event()
stats = {"ok": 0, "fail": 0, "tokens": 0, "cycles": 0}
stats_lock = threading.Lock()


# ---------------------------------------------------------------- sampling ---
def read_smi():
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(SMI_FIELDS)}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip().splitlines()[0]
        vals = [v.strip() for v in out.split(",")]
        return {f.replace(".", "_"): (float(v) if re.match(r"^[\d.]+$", v) else None)
                for f, v in zip(SMI_FIELDS, vals)}
    except Exception:
        return {f.replace(".", "_"): None for f in SMI_FIELDS}


def read_events():
    """Cumulative clock-event counters, in microseconds."""
    counters = {}
    try:
        out = subprocess.run(["nvidia-smi", "-q", "-d", "PERFORMANCE"],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return counters
    in_counters = False
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("Clocks Event Reasons Counters"):
            in_counters = True
            continue
        if in_counters:
            key, _, val = s.partition(":")
            m = re.match(r"^(\d+)\s*us$", val.strip())
            if m:
                counters[re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")] = int(m.group(1))
            elif s:
                break
    return counters


def read_aer():
    totals = {}
    for sev in ("correctable", "fatal", "nonfatal"):
        try:
            with open(f"{PCI_ROOT}/aer_dev_{sev}") as fh:
                for line in fh:
                    p = line.split()
                    if len(p) == 2 and p[0].startswith("TOTAL_ERR") and p[1].isdigit():
                        totals[sev] = int(p[1])
        except OSError:
            pass
    return totals


def read_link():
    speed = width = None
    try:
        with open(f"{PCI_ROOT}/current_link_speed") as fh:
            m = re.match(r"([\d.]+)", fh.read().strip())
            speed = float(m.group(1)) if m else None
        with open(f"{PCI_ROOT}/current_link_width") as fh:
            width = int(fh.read().strip())
    except (OSError, ValueError):
        pass
    return speed, width


def read_rails():
    rails = {}
    try:
        chips = os.listdir("/sys/class/hwmon")
    except OSError:
        return rails
    for c in chips:
        base = f"/sys/class/hwmon/{c}"
        try:
            with open(f"{base}/name") as fh:
                if not fh.read().strip().startswith("nct"):
                    continue
            for e in sorted(os.listdir(base)):
                if e.startswith("in") and e.endswith("_input"):
                    with open(f"{base}/{e}") as fh:
                        rails[e[:-len("_input")]] = int(fh.read().strip()) / 1000.0
        except (OSError, ValueError):
            continue
    return rails


def snapshot():
    ev, aer = read_events(), read_aer()
    speed, width = read_link()
    return {
        "time": datetime.now(timezone.utc).isoformat(),
        "smi": read_smi(), "events": ev, "aer": aer,
        "link_speed_gts": speed, "link_width": width, "rails": read_rails(),
    }


def sampler(path, interval, rail_names):
    fields = (["ts", "elapsed"] + [f.replace(".", "_") for f in SMI_FIELDS]
              + ["link_speed_gts", "link_width",
                 "aer_correctable", "aer_fatal", "aer_nonfatal",
                 "ev_hw_power_braking_us", "ev_sw_power_capping_us",
                 "ev_hw_thermal_slowdown_us", "ev_sw_thermal_slowdown_us"]
              + rail_names + ["gpu_visible"])
    t0 = time.time()
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        while not stop_flag.is_set():
            smi, ev, aer, rails = read_smi(), read_events(), read_aer(), read_rails()
            speed, width = read_link()
            row = {"ts": datetime.now(timezone.utc).isoformat(),
                   "elapsed": round(time.time() - t0, 1),
                   "link_speed_gts": speed, "link_width": width,
                   "aer_correctable": aer.get("correctable"),
                   "aer_fatal": aer.get("fatal"),
                   "aer_nonfatal": aer.get("nonfatal"),
                   "ev_hw_power_braking_us": ev.get("hw_power_braking"),
                   "ev_sw_power_capping_us": ev.get("sw_power_capping"),
                   "ev_hw_thermal_slowdown_us": ev.get("hw_thermal_slowdown"),
                   "ev_sw_thermal_slowdown_us": ev.get("sw_thermal_slowdown"),
                   # If nvidia-smi stops answering, the card has gone. Record it
                   # rather than dying — the crash-capture chain handles the dump.
                   "gpu_visible": 0 if smi.get("power_draw") is None else 1}
            row.update(smi)
            row.update(rails)
            w.writerow(row)
            fh.flush()
            stop_flag.wait(interval)


# ------------------------------------------------------------------- load ---
PROMPT = ("Write a detailed technical explanation of how PCIe link training and "
          "ASPM power states interact on modern GPUs. Be thorough and specific.")


def worker(url, model, max_tokens):
    endpoint = url.rstrip("/") + "/chat/completions"
    while not stop_flag.is_set():
        if not load_gate.is_set():          # idle phase of the duty cycle
            load_gate.wait(1)
            continue
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": max_tokens, "temperature": 0.9,
        }).encode()
        req = urllib.request.Request(
            endpoint, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read())
            n = data.get("usage", {}).get("completion_tokens", 0)
            with stats_lock:
                stats["ok"] += 1
                stats["tokens"] += n
        except Exception:
            with stats_lock:
                stats["fail"] += 1
            stop_flag.wait(2)


# --------------------------------------------------------------- analysis ---
def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 10:
        return None
    xs2, ys2 = [p[0] for p in pairs], [p[1] for p in pairs]
    if len(set(xs2)) < 2 or len(set(ys2)) < 2:
        return None
    try:
        return statistics.correlation(xs2, ys2)
    except Exception:
        return None


def analyse(csv_path, before, after, rail_names, args, elapsed):
    rows = list(csv.DictReader(open(csv_path)))

    def col(name, cast=float):
        out = []
        for r in rows:
            v = r.get(name, "")
            try:
                out.append(cast(v)) if v not in ("", "None") else out.append(None)
            except ValueError:
                out.append(None)
        return out

    power = col("power_draw")
    power_ok = [p for p in power if p is not None]
    temp = [t for t in col("temperature_gpu") if t is not None]
    speeds = [s for s in col("link_speed_gts") if s is not None]

    L = []
    L.append(f"# GPU load test — {args.label}")
    L.append("")
    L.append(f"- started: {before['time']}")
    L.append(f"- duration: {elapsed:.0f}s (requested {args.duration}s), "
             f"concurrency {args.concurrency}")
    L.append(f"- model: `{args.model}` at `{args.url}`")
    L.append(f"- requests: {stats['ok']} ok, {stats['fail']} failed, "
             f"{stats['tokens']} completion tokens")
    L.append(f"- samples: {len(rows)}")
    L.append("")

    L.append("## Verdict")
    L.append("")
    verdicts = []
    aer_delta = {k: (after["aer"].get(k, 0) - before["aer"].get(k, 0))
                 for k in set(before["aer"]) | set(after["aer"])}
    cor_delta = aer_delta.get("correctable", 0)
    verdicts.append(
        f"- **PCIe correctable errors: +{cor_delta}** — "
        + ("LINK IS ACCUMULATING ERRORS under load. Supports the ASPM/retimer "
           "hypothesis; compare against a run with `pcie_aspm=off`."
           if cor_delta else
           "none accumulated. No positive evidence of link instability in this run."))
    brake = (after["events"].get("hw_power_braking", 0)
             - before["events"].get("hw_power_braking", 0))
    verdicts.append(
        f"- **HW power braking: +{brake / 1e6:.6f}s** — "
        + ("PSU ASSERTED THE POWER-BRAKE PIN. Power delivery is implicated."
           if brake else "never asserted. No evidence for the power-delivery theory."))
    thermal = (after["events"].get("hw_thermal_slowdown", 0)
               - before["events"].get("hw_thermal_slowdown", 0))
    verdicts.append(
        f"- **HW thermal slowdown: +{thermal / 1e6:.6f}s** — "
        + ("thermal limit reached." if thermal else "never engaged."))
    fell_off = sum(1 for r in rows if r.get("gpu_visible") == "0")
    if fell_off:
        verdicts.append(f"- **GPU became invisible in {fell_off} sample(s)** — "
                        "the card dropped out. Check /var/log/gpu-crash/.")
    L += verdicts
    L.append("")

    L.append("## Load profile")
    L.append("")
    if power_ok:
        L.append(f"- power draw: mean {statistics.mean(power_ok):.1f} W, "
                 f"peak {max(power_ok):.1f} W, of a 600 W cap")
    if temp:
        L.append(f"- temperature: mean {statistics.mean(temp):.1f} C, peak {max(temp):.0f} C")
    if speeds:
        dist = {s: speeds.count(s) for s in sorted(set(speeds))}
        L.append(f"- link speed distribution (GT/s): {dist}")
        L.append(f"- link width: {sorted(set(w for w in col('link_width') if w))}")
    L.append("")

    L.append("## Rail identification (correlation with GPU power draw)")
    L.append("")
    L.append("A rail feeding the card should correlate NEGATIVELY with power draw —")
    L.append("it sags as the card pulls harder. The most negative is the +12V candidate.")
    L.append("")
    L.append("| rail | mean (raw V) | min | max | corr vs power |")
    L.append("|---|---|---|---|---|")
    scored = []
    for rn in rail_names:
        vals = [v for v in col(rn) if v is not None]
        if not vals:
            continue
        r = pearson(power, col(rn))
        scored.append((r if r is not None else 0, rn, vals, r))
    for _, rn, vals, r in sorted(scored):
        L.append(f"| {rn} | {statistics.mean(vals):.4f} | {min(vals):.3f} | "
                 f"{max(vals):.3f} | {'n/a' if r is None else f'{r:+.3f}'} |")
    L.append("")
    best = sorted(scored)[0] if scored else None
    if best and best[3] is not None and best[3] < -0.5:
        rn, vals, mean_v = best[1], best[2], statistics.mean(best[2])
        droop = max(vals) - min(vals)
        L.append(f"**Strongest anti-correlation: `{rn}` ({best[3]:+.3f}), "
                 f"mean {mean_v:.3f} V raw, droop {droop * 1000:.0f} mV.**")
        # Only read this as a 12V rail if the raw value is consistent with one
        # behind the usual ~6.6:1 divider. A winner sitting at ~3.3V is an
        # undivided 3.3V rail, and applying the 12V narrative to it (as an
        # earlier version of this script did) invents a finding that is not there.
        if 1.5 <= mean_v <= 2.5:
            L.append(f"Consistent with a divided +12 V rail (~{mean_v * 6.6:.2f} V "
                     f"actual); droop would be ~{droop * 6.6:.2f} V. Sag beyond "
                     f"~0.6 V would be a real concern.")
        else:
            L.append(f"**Not a +12 V candidate** — {mean_v:.3f} V raw is an "
                     f"undivided rail (3.3 V class), so it does not feed the card's "
                     f"12 V input. Correlation here most likely reflects overall "
                     f"system load, not GPU supply droop. The +12 V rail feeding "
                     f"the GPU appears not to be monitored on this board.")
        L.append("")
        L.append("_Caveat: the super-I/O ADC quantises to ~8 mV, which is ~53 mV "
                 "referred to a 12 V rail. Small regulation issues are below the "
                 "noise floor regardless of which rail is watched._")
    else:
        L.append("_No rail correlated strongly with power draw. Either none of the "
                 "monitored inputs feeds the card, or the load did not vary enough._")
    L.append("")

    L.append("## Counter deltas (before -> after)")
    L.append("")
    L.append("| counter | before | after | delta |")
    L.append("|---|---|---|---|")
    for k in sorted(set(before["events"]) | set(after["events"])):
        b, a = before["events"].get(k, 0), after["events"].get(k, 0)
        L.append(f"| clocks_event.{k} (us) | {b} | {a} | +{a - b} |")
    for k in sorted(set(before["aer"]) | set(after["aer"])):
        b, a = before["aer"].get(k, 0), after["aer"].get(k, 0)
        L.append(f"| aer.{k} | {b} | {a} | +{a - b} |")
    L.append("")
    L.append(f"Raw samples: `{csv_path}`")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=int, default=1800)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--url", default="http://localhost:11484/v1")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--interval", type=float, default=5.0)
    # Duty cycle. Defaults give 10 load/idle swings across a 30-minute run —
    # enough independent transitions for the rail correlation to mean something.
    # Set --cycle-off 0 for a flat soak.
    ap.add_argument("--cycle-on", type=int, default=120)
    ap.add_argument("--cycle-off", type=int, default=60)
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--out", default="/var/log/gpu-crash/loadtest")
    args = ap.parse_args()

    # Prefer co-locating with the crash artifacts, but never require root just to
    # write a report — nothing else in this test needs privileges.
    try:
        os.makedirs(args.out, exist_ok=True)
        probe = os.path.join(args.out, ".w")
        open(probe, "w").close()
        os.unlink(probe)
    except OSError:
        args.out = os.path.expanduser("~/gpu-loadtests")
        os.makedirs(args.out, exist_ok=True)
        print(f"(not writable as this user; using {args.out})", flush=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = os.path.join(args.out, f"{stamp}-{args.label}")
    csv_path, report_path = base + ".csv", base + ".md"

    try:
        with urllib.request.urlopen(args.url.rstrip("/") + "/models", timeout=15) as r:
            json.loads(r.read())
        print(f"endpoint OK: {args.url}", flush=True)
    except Exception as exc:
        raise SystemExit(f"endpoint not reachable ({exc}) — is the model serving?")

    rail_names = sorted(read_rails().keys())
    print(f"monitoring {len(rail_names)} board rails: {', '.join(rail_names)}", flush=True)

    before = snapshot()
    print(f"baseline: AER {before['aer']}, brake "
          f"{before['events'].get('hw_power_braking', 0)}us, "
          f"link {before['link_speed_gts']}GT/s x{before['link_width']}", flush=True)

    t_sampler = threading.Thread(target=sampler, args=(csv_path, args.interval, rail_names))
    t_sampler.start()
    workers = [threading.Thread(target=worker, args=(args.url, args.model, args.max_tokens))
               for _ in range(args.concurrency)]
    for w in workers:
        w.start()
    print(f"load started: {args.concurrency} workers for {args.duration}s", flush=True)

    t0 = time.time()
    try:
        while time.time() - t0 < args.duration:
            for phase, secs in (("LOAD", args.cycle_on), ("IDLE", args.cycle_off)):
                if time.time() - t0 >= args.duration:
                    break
                if secs <= 0:
                    load_gate.set()
                    continue
                load_gate.set() if phase == "LOAD" else load_gate.clear()
                deadline = min(time.time() + secs, t0 + args.duration)
                while time.time() < deadline and not stop_flag.is_set():
                    time.sleep(min(30, max(1, deadline - time.time())))
                    with stats_lock:
                        smi = read_smi()
                        print(f"  [{time.time() - t0:6.0f}s] {phase} "
                              f"ok={stats['ok']} fail={stats['fail']} tok={stats['tokens']} "
                              f"power={smi.get('power_draw')}W "
                              f"temp={smi.get('temperature_gpu')}C", flush=True)
            with stats_lock:
                stats["cycles"] += 1
    except KeyboardInterrupt:
        print("interrupted — stopping early", flush=True)
    load_gate.set()   # let workers exit promptly rather than sitting in the gate

    elapsed = time.time() - t0
    stop_flag.set()
    for w in workers:
        w.join(timeout=310)
    t_sampler.join(timeout=60)

    after = snapshot()
    report = analyse(csv_path, before, after, rail_names, args, elapsed)
    with open(report_path, "w") as fh:
        fh.write(report)
    print("\n" + report)
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
