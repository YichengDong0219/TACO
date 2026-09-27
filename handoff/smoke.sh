#!/usr/bin/env bash
# Prove THIS machine's policy server is numerically equivalent to the cluster's.
#
# Starts one policy server, sends a fixed probe, and compares the returned action
# chunk against what the cluster's servers return for the identical probe.
# Identical output means the environment (python / torch / cuDNN / kernels) really
# is the same, which is what "aligned with the running experiment" has to mean.
#
#   PORT=18090 GPU=0 ./smoke.sh
#
# The probe is deterministic: a fresh session id restarts the server-side RNG from
# the seeded snapshot, so the same payload always produces the same action chunk.
set -euo pipefail

PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PKG
# The aligned env lives at the same path on both machines.
PYTHON="${PYTHON:-/home/dongyicheng/miniconda3/envs/taco/bin/python}"
PORT="${PORT:-18090}"
GPU="${GPU:-0}"
SESSION="${SESSION:-taco-smoke}"

echo "== 1. GPU headroom (one server needs about 9.9 GB)"
nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader,nounits |
  awk -F', ' '{printf "  gpu%s: free %.1f GB%s\n", $1, $3/1024, ($3<11000 ? "   <-- too small for one server" : "")}'

echo "== 2. start the server on gpu$GPU, port $PORT"
mkdir -p "$PKG/logs"
tmux kill-window -t "$SESSION:srv" 2>/dev/null || true
tmux has-session -t "$SESSION" 2>/dev/null || tmux new-session -d -s "$SESSION" -n ctl
tmux new-window -d -t "$SESSION" -n srv \
  "CUDA_VISIBLE_DEVICES=$GPU exec $PYTHON '$PKG/scripts/pi05_policy_server.py' \
     --policy-path '$PKG/pi05_robotwin_lerobot' \
     --tokenizer-path '$PKG/tokenizer/paligemma-3b-pt-224' \
     --host 127.0.0.1 --port $PORT > '$PKG/logs/smoke_server.log' 2>&1"
echo "  waiting ~90 s for the 7.2 GB weights to load ..."
sleep 90

echo "== 3. probe and compare with the cluster"
PYTHONPATH="$PKG/cfn" "$PYTHON" - "$PORT" <<'PY'
import hashlib, os, sys, uuid

sys.path.insert(0, os.path.join(os.environ["PKG"], "cfn"))
import numpy as np
from cfn.pi05_rpc import PI05RPCClient

# The exact probe the cluster's servers were measured with.
EXPECT_SHA = "3f57c7a3f8c252e6"
EXPECT_SUM = 239.859314

state = np.zeros(14, dtype=np.float32)
state[:4] = [0.5, -0.25, 1.0, 0.0]
image = np.full((240, 320, 3), 128, dtype=np.uint8)

client = PI05RPCClient(f"127.0.0.1:{sys.argv[1]}")
print("  health:", client.health())
action = np.asarray(client.request(
    "action", mode="baseline", session_id="probe-" + uuid.uuid4().hex,
    instruction="press the bell", images=[image, image, image], state=state,
))
client.close()
sha = hashlib.sha256(action.tobytes()).hexdigest()[:16]
total = float(action.sum())
print(f"  action shape={action.shape}  sum={total:.6f}  sha={sha}")
print(f"  cluster reference       sum={EXPECT_SUM:.6f}  sha={EXPECT_SHA}")
if sha == EXPECT_SHA:
    print("  RESULT: IDENTICAL -- this machine reproduces the cluster bit for bit.")
elif abs(total - EXPECT_SUM) < 1e-3:
    print("  RESULT: matches to 1e-3 but not bit-identical (kernel/cuDNN ordering).")
    print("          Workable, but the features cannot be assumed bit-equal.")
else:
    print("  RESULT: DIFFERENT -- do NOT mix this machine's features with the cluster's.")
    raise SystemExit(1)
PY
