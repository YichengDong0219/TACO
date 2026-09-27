# TACO — PI0.5 feature extraction handoff

Extract PI0.5 internal-representation features for the 12-task RoboTwin2.0
campaign. The features are the training input for the CFN; this package is
everything needed to produce them, with no RoboTwin simulator and no assets.

## What is in here

| Path | Size | Role |
|---|---|---|
| `datasets/robotwin12_official_v1/<task>/` | 6.2 GB | LeRobot datasets — **the only input data** |
| `pi05_robotwin_lerobot/` | 6.8 GB | PI0.5 cotrain policy (converted export) |
| `tokenizer/paligemma-3b-pt-224/` | 17 MB | Pinned tokenizer (`35e4f464…`) |
| `third_party/lerobot/` | 109 MB | LeRobot checkout the extraction runs on — **patched, see below** |
| `scripts/` | 348 KB | `collect.py` (extraction), `robotwin_multitask.py` (driver), `pi05_policy_server.py` |
| `cfn/` | 64 KB | `PI05RPCClient` |
| `robotwin_train_episodes_12tasks/` | 436 KB | Manifests (hashed into each output `manifest.json`) |
| `environment.yml` | — | Pinned conda env |

Output: `features/<task>/feature.pt` (9–37 MB each) + `manifest.json`.
Expected total: 12 files, ~150–200 MB.

**No RoboTwin checkout is needed.** The extraction reads the LeRobot datasets and
calls the policy server over RPC; it never builds a simulator environment.
(RoboTwin is only required for data *collection* and for *evaluation*.)

## Requirements

- GPUs (RTX 4090 class, ≥ 12 GB each) — one policy server per GPU.
- The pinned python env: `conda env create -f environment.yml`
  (python 3.10.21, torch 2.7.0+cu126, torchvision 0.22.0, numpy 1.26.4,
  opencv 4.11.0, h5py 3.16.0).
- `tmux`, so the servers outlive your shell.
- CPU: each extraction client runs `--num_workers=8` dataloaders, so
  **8–10 cores per concurrent job**. Over-subscribing cores slows every stream
  down; see "Tuning" below.

## Quick start

```bash
cd taco-feature-extraction
conda activate <env>            # or: conda env create -f environment.yml
tmux new -s taco                # servers must survive disconnects

# one server per GPU, then the extraction queue across all 12 tasks
GPUS="0 1 2 3" ./run_extraction.sh
```

`run_extraction.sh` starts one policy server per GPU (ports 18080, 18081, …),
waits for the weights to load, then runs `extract-queue`, which walks the 12
tasks and writes `features/<task>/feature.pt`.

Check progress at any time:

```bash
cat extract_queue_status.json            # pending / active / failures
ls features/*/feature.pt | wc -l         # finished tasks, expect 12
tail -f extract_logs/<task>/job.log      # outer bar = task chunks
```

Validate the package before committing hours to it:

```bash
python check_env.py                      # layout + patch + prompt-shape checks
```

## Acceptance criteria

1. `python check_env.py` passes (includes the prompt-state check below).
2. `features/<task>/feature.pt` exists for **all 12** tasks.
3. Every `features/<task>/manifest.json` carries `feature_sha256`,
   `policy_sha256` and `manifest_sha256`.
4. `extract_queue_status.json` ends with `pending=[]`, `active=[]`,
   `failures=[]`.

## Do not replace `third_party/lerobot`

This checkout is **patched** at two sites, both aligning the eval-time input with
what the checkpoint was trained on. Using an unpatched LeRobot silently degrades
every result:

- `policies/pi05/processor_pi05.py` — the state written into the prompt must be
  the **un-padded** state. The original pads to `max_state_dim=32` first, which
  appends 18 zero-pad bins ("128") to the prompt the model conditions on.
  openpi tokenizes before padding (`TokenizePrompt` precedes
  `PadStatesAndActions`), so training only ever saw 14 numbers.
- `policies/pi05/modeling_pi05.py` — `resize_with_pad_torch` must pad float
  inputs with **0.0**, not −1.0. The pad band is 25 % of every camera frame
  (320×240 → 224×168, 56 of 224 rows); with −1.0 the caller's `* 2 - 1` pushes
  it to −3.0, far outside the trained range.

`check_env.py` asserts both, and prints the prompt the pipeline actually builds
(it must contain exactly 14 numbers in `State: …`).

## Tuning

- **Concurrency** is `--max-concurrent 0` = one extraction job per server. Keep
  it at one per server: the server serialises every request on one lock, so extra
  clients on the same server add latency without throughput.
- **Server count** scales throughput roughly linearly (measured: 4 servers =
  1.17 chunks/min, 6 = 1.38 on a 20-core box), but only up to what the CPU can
  feed. Watch `uptime`: once load approaches your core count, more servers stop
  helping.
- **Per-server pinning**: don't put two servers on one GPU unless you have no
  choice — one client already saturates a card for this workload, so the second
  server mostly buys latency.
- Job order is slowest-first for free (`EXTRACT_SLOWEST_FIRST` in
  `scripts/robotwin_multitask.py`, measured chunk counts), and each job is handed
  to whichever server is free, so the tail does not stall.

## Timing

99 task-hours of server-bound work in total, dominated by `stack_blocks_two`
(295 chunks ≈ 13.8 h on its own) and `scan_object` / `place_fan` / `stamp_seal`
(~135 chunks each). On 4–6 servers expect **16–25 hours**; a single task run
(`extract --task click_bell`, the cheapest at 73 chunks) finishes in ~5 h.
