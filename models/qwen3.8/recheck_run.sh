#!/usr/bin/env bash
# Steady-state recheck of the two cells where a 60s window closes while vLLM is
# still draining the prefill backlog ("Running: 6, Waiting: 58" at 10K x 64).
set -uo pipefail
cd "$(dirname "$0")"
export GLLM_TIMEOUT=1800
MAXSEC=300 ./gllm.sh nvfp4-w4a4-steady pp10k "32,64"
MAXSEC=300 ./gllm.sh nvfp4-w4a4-steady pp8k  "32,64"
echo "SCALE|nvfp4-w4a4-steady|end_status=$(docker inspect --format='{{.State.Status}}' prod 2>/dev/null)|done" >> benchmarks/SCALING.log
