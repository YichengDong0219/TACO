#!/usr/bin/env bash
# Start one PI0.5 policy server per GPU, then run the 12-task extraction queue.
#
#   GPUS="0 1 2 3" ./run_extraction.sh
#
# Environment overrides:
#   PYTHON      python of the pinned env (default: python)
#   GPUS        space separated GPU ids, one server each (default: 0)
#   PORT_BASE   first server port (default: 18080)
#   SESSION     tmux session name (default: taco)
#   START_ONLY  1 = start the servers and stop, without running the queue
set -euo pipefail

PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
GPUS="${GPUS:-0}"
PORT_BASE="${PORT_BASE:-18080}"
SESSION="${SESSION:-taco}"
START_ONLY="${START_ONLY:-0}"

mkdir -p "$PKG/logs"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found: the policy servers must outlive this shell." >&2
  exit 1
fi

# 1) One server per GPU. Two servers on one card is wasteful: a single client
#    already saturates the card for this workload.
tmux has-session -t "$SESSION" 2>/dev/null || tmux new-session -d -s "$SESSION" -n control

SERVER_LIST=""
index=0
for gpu in $GPUS; do
  port=$((PORT_BASE + index))
  index=$((index + 1))
  SERVER_LIST="${SERVER_LIST:+$SERVER_LIST,}127.0.0.1:$port"
  tmux new-window -d -t "$SESSION" -n "srv$port" \
    "CUDA_VISIBLE_DEVICES=$gpu exec $PYTHON '$PKG/scripts/pi05_policy_server.py' \
       --policy-path '$PKG/pi05_robotwin_lerobot' \
       --tokenizer-path '$PKG/tokenizer/paligemma-3b-pt-224' \
       --host 127.0.0.1 --port $port > '$PKG/logs/server_$port.log' 2>&1"
  echo "server $port -> GPU $gpu"
done

echo "waiting for the policy weights to load (about 60-90 s) ..."
sleep 90

for port in $(echo "$SERVER_LIST" | tr ',' ' ' | sed 's/.*://'); do
  if grep -q '"cfn_loaded"' "$PKG/logs/server_$port.log" 2>/dev/null; then
    echo "  server $port ready"
  else
    echo "  server $port NOT ready yet -- check $PKG/logs/server_$port.log" >&2
  fi
done

if [[ "$START_ONLY" == "1" ]]; then
  echo "START_ONLY=1: servers are up on $SERVER_LIST, queue not started."
  exit 0
fi

# 2) The extraction queue. --max-concurrent 0 means "one job per policy server",
#    which is what keeps every server busy without stacking clients on one lock.
exec "$PYTHON" "$PKG/scripts/robotwin_multitask.py" \
  --artifacts "$PKG" \
  --policy "$PKG/pi05_robotwin_lerobot" \
  --tokenizer "$PKG/tokenizer/paligemma-3b-pt-224" \
  --train-manifests "$PKG/robotwin_train_episodes_12tasks" \
  --policy-server "$SERVER_LIST" \
  extract-queue --max-attempts 2 --max-concurrent 0
