#!/usr/bin/env bash
# Run a named subset of tasks through the extraction, serially, on this machine.
#
#   TASKS="click_bell turn_switch" SERVER=127.0.0.1:18090 PARALLEL=1 ./run_tasks.sh
#
# Why not `extract-queue`: the queue walks the hardcoded 12-task list and skips
# only what already has feature.pt, so it cannot be pointed at a subset. Tasks are
# run heaviest-first and the next starts as soon as one finishes, which is the same
# dynamic behaviour the queue gives, just over a chosen set.
#
# Each task is launched exactly the way extract_queue launches it, so an
# already-present feature.pt is reused instead of recomputed (see extract()).
set -euo pipefail

PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/home/dongyicheng/miniconda3/envs/taco/bin/python}"
SERVER="${SERVER:-127.0.0.1:18090}"
PARALLEL="${PARALLEL:-1}"
TASKS="${TASKS:-}"
GPU="${GPU:-0}"

[[ -n "$TASKS" ]] || { echo "set TASKS=\"task1 task2\"" >&2; exit 2; }

# Heaviest first (measured chunk counts), so the long tail cannot strangle the end.
order=(stack_blocks_two place_fan stamp_seal scan_object move_pillbottle_pad \
       place_mouse_pad place_bread_skillet move_playingcard_away pick_dual_bottles \
       press_stapler turn_switch click_bell)
sorted=""
for t in "${order[@]}"; do
  for want in $TASKS; do [[ "$t" == "$want" ]] && sorted="$sorted $t"; done
done

mkdir -p "$PKG/logs"
echo "tasks (heaviest first):$sorted"
echo "server=$SERVER parallel=$PARALLEL"

running=0
for task in $sorted; do
  feat="$PKG/features/$task/feature.pt"
  if [[ -f "$feat" ]]; then
    echo "  $task: feature.pt already present, skipping compute"
  fi
  while (( running >= PARALLEL )); do wait -n; running=$((running-1)); done
  (
    TACO_POLICY_SERVER="$SERVER" \
    PYTHONPATH="$PKG/third_party/lerobot/src:$PKG/cfn" \
    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="$GPU" \
    "$PYTHON" "$PKG/scripts/robotwin_multitask.py" \
      --artifacts "$PKG" \
      --policy "$PKG/pi05_robotwin_lerobot" \
      --train-manifests "$PKG/robotwin_train_episodes_12tasks" \
      --policy-server "$SERVER" \
      extract --task "$task" --gpu "$GPU" \
      > "$PKG/logs/extract_$task.log" 2>&1
    code=$?
    if [[ $code -ne 0 ]]; then
      echo "  $task: FAILED (exit $code) -- see logs/extract_$task.log" >&2
    else
      echo "  $task: done -> $(du -h "$PKG/features/$task/feature.pt" 2>/dev/null | cut -f1)"
    fi
  ) &
  running=$((running+1))
done
wait
echo "all requested tasks finished"
