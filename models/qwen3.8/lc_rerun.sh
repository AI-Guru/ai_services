#!/usr/bin/env bash
# Long-context levels, re-run against an already-running container with a
# request timeout a 100K-prompt request can actually finish inside, and a window
# long enough for high-concurrency levels to reach steady state. Written to the
# "lc2" label; the first attempt is kept under "nvfp4-w4a4-lc*" as the record of
# what a 60s client timeout does to these levels.
set -uo pipefail
cd "$(dirname "$0")"
export GLLM_TIMEOUT=1800
MAXSEC=420 ./gllm.sh nvfp4-w4a4-lc2 lc100k "1,4,8,16"
MAXSEC=420 ./gllm.sh nvfp4-w4a4-lc2 lc150k "1,4,8,10"
echo "SCALE|nvfp4-w4a4-lc2|end_status=$(docker inspect --format='{{.State.Status}}' prod 2>/dev/null)|done" >> benchmarks/SCALING.log
