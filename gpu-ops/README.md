# gpu-ops — GPU incident tooling

Operational scripts for the RTX PRO 6000. **These are incident-response tools:
if one is broken or missing you find out at the worst possible moment**, which is
why they live in git rather than loose in a home directory.

For *when* and *why* to run any of this, see
[`../GPU-INCIDENT-RUNBOOK.md`](../GPU-INCIDENT-RUNBOOK.md). This file only covers
how the files are laid out and installed.

## This directory is the source of truth

The historical paths under `/home/despara/Development/` are now **symlinks** into
here:

```
/home/despara/Development/safe-shutdown.sh -> ai_services/gpu-ops/safe-shutdown.sh
/home/despara/Development/gpu-crash-collect.sh -> ai_services/gpu-ops/gpu-crash-collect.sh
...etc
```

Editing a file here changes the live tool immediately — no copy step, no drift.
The systemd units, the cross-references inside the scripts, and the runbook all
still use the `/home/despara/Development/...` paths and resolve through the
symlinks, so nothing needed rewriting.

**The one failure mode to know:** if this repo is moved or renamed, every symlink
breaks and the capture chain silently stops working. Re-point them with:

```bash
cd /home/despara/Development
for f in safe-shutdown.sh gpu-crash-collect.sh gpu-xid-watch.sh \
         gpu-crash-install.sh gpu-sensors-probe.sh gpu-aspm-toggle.sh gpu-loadtest.py; do
  ln -sfn "ai_services/gpu-ops/$f" "$f"
done
```

## Contents

| File | Purpose |
|---|---|
| `safe-shutdown.sh` | Release the GPU gracefully, then power off/reboot. **Use instead of `reboot`.** Also collects a crash dump first if this boot logged a severe Xid. |
| `gpu-crash-collect.sh` | Capture the GPU crash dump + kernel log + PCIe state before a reboot destroys it. |
| `gpu-xid-watch.sh` | Follows the kernel log; fires the collector seconds after a severe Xid. Run by `gpu-xid-watch.service`. |
| `gpu-crash-install.sh` | Installs and verifies the whole capture chain. Safe to re-run. |
| `gpu-loadtest.py` | Instrumented soak: AER, power-brake and thermal counters, link speed, board rails. |
| `gpu-aspm-toggle.sh` | Manage the PCIe ASPM kernel parameter. `--verify` / `--runtime` / `--revert`. |
| `gpu-sensors-probe.sh` | Load `nct6775` for motherboard voltage rails; persists it. |
| `systemd/` | Unit files. **Copies** — the live ones are in `/etc/systemd/system/`. |

## Installing / reinstalling

The systemd units are *not* symlinked (systemd is happier owning real files in
`/etc`), so after editing anything under `systemd/` you must reinstall:

```bash
sudo /home/despara/Development/gpu-crash-install.sh
```

That installs both units, reloads systemd, enables the watcher, and runs a full
verification including a real forced collection. `--test-only` re-verifies
without reinstalling; `--no-test` installs without the ~1 minute collection.

It deliberately does **not** restart `gpu-safe-shutdown.service` — that unit's
`ExecStop` *is* `safe-shutdown.sh`, so restarting it would stop every GPU
container on the box. A `daemon-reload` is sufficient.

## Verifying after a reboot

```bash
systemctl is-active gpu-xid-watch.service     # active
docker ps --filter name=exporter              # 2 containers
lsmod | grep -c nct6775                       # >0
cat /proc/cmdline                             # pcie_aspm.policy=performance
```

## Backups

A pre-move tarball of all scripts and units is at
`/home/despara/gpu-ops-backup-<timestamp>.tar.gz`. Git history is the real
backup from here on.
