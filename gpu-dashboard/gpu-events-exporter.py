#!/usr/bin/env python3
"""
gpu-events-exporter — the signals nvidia_gpu_exporter cannot give you.

Why this exists alongside the nvidia-smi exporter:

  1. CUMULATIVE EVENT COUNTERS. `nvidia-smi -q -d PERFORMANCE` exposes
     "Clocks Event Reasons Counters" in microseconds since driver load. These
     ACCUMULATE, so a 50ms power-brake event lands in them permanently — whereas
     a 5s-interval sampler of power.draw misses it entirely. For diagnosing a
     transient (the leading hypothesis for the 2026-08-14 Xid 79), the counter
     is structurally better than the sample. None of these are available via
     `nvidia-smi --query-gpu=`, which is why the other exporter cannot see them.

     The one to watch is HW Power Braking: it ticks when the PSU asserts the
     power-brake pin. Non-zero = the power-delivery hypothesis is confirmed.

  2. PCIe AER ERROR COUNTERS from sysfs. Correctable errors (BadTLP, BadDLLP,
     Replay Timer Timeout) accumulate on a marginal link long before it drops.
     They are the direct probe for the ASPM/retimer link-instability hypothesis.

  3. LINK SPEED/WIDTH from sysfs, to catch a link that silently downgrades or
     fails to return from a low-power state.

Serves Prometheus text format on :9836/metrics. Stdlib only, no dependencies.
"""
import os
import re
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

GPU_BDF = os.environ.get("GPU_BDF", "0000:01:00.0")
PORT = int(os.environ.get("PORT", "9836"))
PCI_ROOT = f"/sys/bus/pci/devices/{GPU_BDF}"

# Written for a single-GPU host, which is what this box is. With more than one
# card the nvidia-smi section would need per-GPU splitting.


def _slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def read_nvidia_events():
    """Parse the two 'Clocks Event Reasons' blocks out of nvidia-smi."""
    states, counters = {}, {}
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-q", "-d", "PERFORMANCE"],
            capture_output=True, text=True, timeout=20,
        )
    except Exception:
        return states, counters, 1
    # MUST check returncode. The 2026-09-05 Xid 79 was diagnosed blind because
    # this did not: nvidia-smi was failing (very likely contention -- two
    # exporters plus a saturated GPU), returncode was non-zero, stdout was empty,
    # the parse produced nothing, and err stayed 0. The exporter reported
    # gpu_events_exporter_up=1 for the whole hour before the fault while silently
    # emitting no clock-event counters at all -- including hw_power_braking, the
    # one metric that would have settled the power-delivery question.
    if proc.returncode != 0:
        return states, counters, 1
    out = proc.stdout

    section = None
    for line in out.splitlines():
        stripped = line.strip()
        if stripped.startswith("Clocks Event Reasons Counters"):
            section = "counters"
            continue
        if stripped.startswith("Clocks Event Reasons"):
            section = "states"
            continue
        if not stripped or ":" not in stripped:
            continue
        # A new top-level block (e.g. "Sparse Operation Mode") ends the section.
        key, _, val = stripped.partition(":")
        key, val = key.strip(), val.strip()
        if section == "states":
            if val in ("Active", "Not Active"):
                states[_slug(key)] = 1 if val == "Active" else 0
            elif not val.endswith("us"):
                section = None
        elif section == "counters":
            m = re.match(r"^(\d+)\s*us$", val)
            if m:
                counters[_slug(key)] = int(m.group(1))
            else:
                section = None
    return states, counters, 0


def read_aer():
    """AER counters from sysfs: files hold 'ErrName <count>' per line."""
    result = {}
    for severity in ("correctable", "fatal", "nonfatal"):
        path = f"{PCI_ROOT}/aer_dev_{severity}"
        try:
            with open(path) as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) == 2 and parts[1].isdigit():
                        result[(severity, parts[0])] = int(parts[1])
        except OSError:
            continue  # AER not exposed (kernel lacks control of it) — skip
    return result


def read_hwmon_rails():
    """Motherboard voltage rails from the super-I/O chip (NCT6799 on this board).

    Reported RAW: the driver supplies no labels or scaling for the voltage
    inputs (only for temperatures), because each rail sits behind a
    board-specific resistor divider the driver cannot know about. So `in16`
    reads 1.840 V rather than the ~12.14 V it most likely represents (~6.6:1).

    Exporting them unscaled is deliberate — inventing a divider would bake a
    guess into the data. Identify the +12V rail empirically instead: under GPU
    load the rail feeding the card is the one that droops in anti-correlation
    with power draw. Once identified, apply the scale factor in the query.
    """
    rails = {}
    prefixes = tuple(p.strip() for p in
                     os.environ.get("HWMON_CHIPS", "nct").split(",") if p.strip())
    try:
        chips = sorted(os.listdir("/sys/class/hwmon"))
    except OSError:
        return rails
    for chip_dir in chips:
        base = f"/sys/class/hwmon/{chip_dir}"
        try:
            with open(f"{base}/name") as fh:
                chip = fh.read().strip()
        except OSError:
            continue
        if not chip.startswith(prefixes):
            continue
        try:
            entries = sorted(os.listdir(base))
        except OSError:
            continue
        for entry in entries:
            if not (entry.startswith("in") and entry.endswith("_input")):
                continue
            name = entry[: -len("_input")]
            try:
                with open(f"{base}/{entry}") as fh:
                    millivolts = int(fh.read().strip())
            except (OSError, ValueError):
                continue
            label = ""
            try:
                with open(f"{base}/{name}_label") as fh:
                    label = fh.read().strip()
            except OSError:
                pass
            rails[(chip, name, label)] = millivolts / 1000.0
    return rails


def read_link():
    """Current negotiated link speed (GT/s) and width."""
    speed = width = None
    try:
        with open(f"{PCI_ROOT}/current_link_speed") as fh:
            m = re.match(r"([\d.]+)\s*GT/s", fh.read().strip())
            if m:
                speed = float(m.group(1))
    except OSError:
        pass
    try:
        with open(f"{PCI_ROOT}/current_link_width") as fh:
            width = int(fh.read().strip())
    except OSError:
        pass
    return speed, width


def render():
    started = time.time()
    lines = []
    add = lines.append

    states, counters, err = read_nvidia_events()

    add("# HELP nvidia_clocks_event_active Clock event reason currently asserted (1/0).")
    add("# TYPE nvidia_clocks_event_active gauge")
    for reason, val in sorted(states.items()):
        add(f'nvidia_clocks_event_active{{reason="{reason}"}} {val}')

    add("# HELP nvidia_clocks_event_seconds_total Cumulative time in each clock event "
        "reason since driver load. Catches transients shorter than the scrape interval.")
    add("# TYPE nvidia_clocks_event_seconds_total counter")
    for reason, usec in sorted(counters.items()):
        add(f'nvidia_clocks_event_seconds_total{{reason="{reason}"}} {usec / 1e6:.6f}')

    aer = read_aer()
    add("# HELP pcie_aer_errors_total PCIe Advanced Error Reporting counters from sysfs.")
    add("# TYPE pcie_aer_errors_total counter")
    for (severity, etype), val in sorted(aer.items()):
        add(f'pcie_aer_errors_total{{bdf="{GPU_BDF}",severity="{severity}",'
            f'type="{etype}"}} {val}')

    rails = read_hwmon_rails()
    if rails:
        add("# HELP board_voltage_raw_volts Super-I/O voltage input, RAW (undivided). "
            "Divided rails read low: ~1.84V here is most likely +12V at ~6.6:1. "
            "Identify by anti-correlation with GPU power draw under load.")
        add("# TYPE board_voltage_raw_volts gauge")
        for (chip, name, label), volts in sorted(rails.items()):
            add(f'board_voltage_raw_volts{{chip="{chip}",input="{name}",'
                f'label="{label}"}} {volts:.3f}')

    speed, width = read_link()
    if speed is not None:
        add("# HELP pcie_link_speed_gts Current negotiated PCIe link speed (GT/s).")
        add("# TYPE pcie_link_speed_gts gauge")
        add(f'pcie_link_speed_gts{{bdf="{GPU_BDF}"}} {speed}')
    if width is not None:
        add("# HELP pcie_link_width Current negotiated PCIe link width (lanes).")
        add("# TYPE pcie_link_width gauge")
        add(f'pcie_link_width{{bdf="{GPU_BDF}"}} {width}')

    add("# HELP gpu_events_exporter_up Whether the nvidia-smi query succeeded.")
    add("# TYPE gpu_events_exporter_up gauge")
    add(f"gpu_events_exporter_up {0 if err else 1}")
    # Distinct from _up: the query can succeed and still yield nothing usable.
    # Alert on this being 0, not just on _up.
    add("# HELP gpu_events_clock_counters_parsed Number of clock-event counters "
        "parsed. 0 means the counters are silently missing -- treat as an outage.")
    add("# TYPE gpu_events_clock_counters_parsed gauge")
    add(f"gpu_events_clock_counters_parsed {len(counters)}")
    add("# HELP gpu_events_exporter_scrape_seconds Time taken to build this response.")
    add("# TYPE gpu_events_exporter_scrape_seconds gauge")
    add(f"gpu_events_exporter_scrape_seconds {time.time() - started:.4f}")

    return ("\n".join(lines) + "\n").encode()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.rstrip("/") not in ("", "/metrics"):
            self.send_error(404)
            return
        try:
            body = render()
        except Exception as exc:  # never let a scrape take the exporter down
            body = f"# scrape failed: {exc}\ngpu_events_exporter_up 0\n".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # Prometheus gave up mid-write. Harmless, but it used to propagate
            # and take the whole (single-threaded) server down with it.
            pass

    def log_message(self, *args):
        pass  # a scrape every 5s would otherwise flood the container log


if __name__ == "__main__":
    print(f"gpu-events-exporter listening on :{PORT} for {GPU_BDF}", flush=True)
    ThreadingHTTPServer(("", PORT), Handler).serve_forever()
