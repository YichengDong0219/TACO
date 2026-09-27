# RoboTwin 12-task multi-task CFN

This branch uses the existing PI0.5 LeRobot checkpoint and RoboTwin assets. It
does not use a published TACO checkpoint or any DSRL artifact.

## Fixed inputs

These are the values in `taco.toml`; a different machine overrides them there,
in the untracked `taco.local.toml`, in the environment (`TACO_POLICY` and
friends), or with the matching command-line flag.

- Policy: `/home/dongyicheng/dsrl/pi05_robotwin_lerobot`
- Assets: `/home/dongyicheng/dsrl/RoboTwin-assets/assets`
- RoboTwin: `third_party/RoboTwin-official` at `6dde5715`, from
  `robotwin-Platform/RoboTwin`
- Train manifests: `robotwin_train_episodes_12tasks` (30 episodes per task)
- Eval manifests: `robotwin_eval_episodes_12tasks` (100 episodes per task)
- Artifacts: `artifacts/robotwin12_official_v1`

Each manifest seed and instruction is used exactly as written. A failed expert
episode or invalid scene aborts collection; it is never replaced by another
seed.

## Setting up a machine

The repository carries the pipeline, the vendored LeRobot, the manifests and
the CFN, but three of its inputs live outside it: the RoboTwin checkout, the
asset tree, and the PI0.5 checkpoint. Nothing in the Python pipeline fetches
them — `bootstrap` asserts they are already in place — so they are obtained
here:

```bash
vendor/bootstrap.sh env        # conda env from environment.yml + the three
                               # editable installs (lerobot, transformers, cfn)
vendor/bootstrap.sh robotwin   # clone RoboTwin at the pinned revision, fetch
                               # ~30 GB of assets, symlink them into the checkout
vendor/bootstrap.sh tokenizer  # PaliGemma tokenizer, which `serve` needs and
                               # cannot fetch itself (HF_HUB_OFFLINE=1)
vendor/bootstrap.sh check      # verify the layout and every load-bearing patch
```

`environment.yml` deliberately omits `lerobot`, `transformers` and `cfn`: as
releases they would be an unpatched LeRobot, which degrades every result
silently. They must come from this working tree, which is what `env` installs.

The verbs are separate because they fail differently and are worth re-running
separately: `env` needs conda, `robotwin` needs network and disk, `tokenizer`
needs Hugging Face auth (the PaliGemma repo is gated), and `check` needs none of
them and is safe to run at any time.

**The CFN is buildable here; the policy is not.** `collect → convert → extract →
train` produce a CFN from the manifests and the asset tree alone, but `extract`,
`serve` and `eval` all need the fine-tuned PI0.5 at `paths.policy` (6.8 GB),
which is a local conversion — `openpi_export.json` in it records its origin as
`/newhome/xiexinyi/vla/openpi/checkpoints/RoboTwin-lerobot_v30-aloha_agilex-joint-0/29999`
under the `rlinf_lerobot_pi05_v1` format. It is in no repository and on no hub,
so a machine without that artifact can run the collection and conversion halves
of the pipeline and nothing downstream of them.

The RoboTwin revision is pinned in `scripts/robotwin_multitask.py` as
`ROBOTWIN_REVISION` and asserted by `bootstrap`. Environments, experts and task
definitions all come from the checkout, so a different revision is a different
benchmark: it is refused unless `--allow-revision-mismatch` is passed, and the
revision each run actually saw is written to its `run_manifest.json` either
way. The upstream TACO release also vendors a `third_party/Robotwin`, but it is
an older RoboTwin with the pre-2.0 `script/` layout and cannot run `convert` or
`eval`, which need `data/decode_image_bit.py` and
`scripts/eval_policy_xpolicylab.py` from the official checkout. It was removed
along with the three single-task driver scripts that existed only to reach into
it; they are superseded by `scripts/robotwin_data/`. Both are recoverable:

```bash
git checkout e3000ba -- third_party/Robotwin \
    scripts/eval/eval_robotwin2_torch_pi05_taco.sh \
    scripts/robotwin_data/task_dataset_collection.sh \
    scripts/robotwin_data/data_trans/rt2-hdf5_2_hdf5_2_lerobot.sh
```

## Official RoboTwin, no video

Environments and experts run from an unmodified official checkout; TACO keeps
only the manifests, the CFN, and the RPC policy client. The official checkout is
never edited — `bootstrap` only symlinks the shared asset tree into it.

The pipeline never writes or reads video. Episode cameras come from the HDF5
that `create_xpolicylab_hdf5` writes, and are decoded with RoboTwin's
`data.decode_image_bit`, which is the only decoder that tells the legacy
channel-reversed byte format apart from the marked standard format that
self-collected episodes now use. `cv2.imdecode` alone returns BGR for the marked
format, so it must not be used here. LeRobot datasets are written with
`use_videos=False`, so frames stay PNG.

## Commands

Run commands in the existing `taco` Conda environment. The unattended campaign
is `pipeline`; the rest are the individual steps and manual tools:

```bash
python scripts/robotwin_multitask.py validate
python scripts/robotwin_multitask.py bootstrap
python scripts/robotwin_multitask.py serve --gpu 1 --port 18081
python scripts/robotwin_multitask.py --policy-server 127.0.0.1:18080 smoke --gpu 3
python scripts/robotwin_multitask.py collect-queue
python scripts/robotwin_multitask.py --policy-server 127.0.0.1:18080,127.0.0.1:18081 extract-queue
python scripts/robotwin_multitask.py train --gpu 1 --epochs 16
python scripts/robotwin_multitask.py --policy-server 127.0.0.1:18080,127.0.0.1:18081 load-cfn
python scripts/robotwin_multitask.py --policy-server 127.0.0.1:18080,127.0.0.1:18081 queue
python scripts/robotwin_multitask.py summarize
```

`smoke` runs one `click_bell` episode through the same collectors, converter,
feature extractor and evaluator the full run uses, and stops with `COMPLETE`
only after both baseline and taco evaluation episodes finish.

## The unattended run

`pipeline` chains the stages in the order they depend on each other and is
resumable at every boundary, so it can be left alone:

1. `collect-queue` — collect and convert every task. A device is booked only
   while an environment is running on it, and the raw HDF5 is kept.
2. `extract-queue` — extract PI0.5 features, one job per policy server.
3. `train` — 16 epochs over every task that produced features. A missing task is
   recorded in `training_shortfall.json` and reported, not fatal; the run fails
   only if nothing at all was extracted.
4. `load-cfn`, then `queue` and `summarize`.

Every stage skips work that already exists on disk, so an interrupted run
resumes where it stopped. Collection additionally re-verifies that a previously
written episode still opens and still deserializes before trusting it, because a
process killed mid-write leaves a file that otherwise looks finished.

## CFN sampling

The 12 feature tensors are loaded in stable path order and concatenated. Every
epoch shuffles the concatenated frame set with seed 42. There is no task-level
reweighting or replacement sampler, so longer trajectories naturally
contribute more frames. `train_config.json` records the strategy as
`shuffled_concatenation` and includes the actual frame count for every task.

The formal checkpoints are saved after epochs 4, 8, 12, and 16. Evaluation
uses `model_epoch16.pt` by default.

## Scheduling and outputs

GPUs are shared with other users, so admission is decided by the free memory
`nvidia-smi` reports rather than by whether a device is empty: a card carrying
someone else's process is still usable while it has room for our reservation
plus `--safety-mib`. Nothing another user runs is ever terminated or preempted,
and the status files record which devices carry foreign processes. The eval
queue launches at most `--max-jobs-per-gpu` (default 1) jobs per device and
reserves `--eval-reserve-mib` for each. A job is complete only after all 100
fixed episodes have been written and `COMPLETE` exists.

`collect-queue` fits up to `--max-envs-per-gpu` (default 4) environments per
device, reserving `--env-reserve-mib` for each. A measured environment peaks
around 5.4 GB, which is what the reservation is sized against.

`extract-queue` runs one extraction per policy server by default
(`--max-concurrent`), because a server serializes its requests on one GPU.

`--preferred-gpus` gives a device order to either queue. The listed devices are
filled first and the rest are used only once those are full, so latency-critical
evaluation can take reserved cards while bulk collection fills the shared ones.
Devices are still never over-committed: the ordering changes which card is
chosen, not whether a card fits.

The policy server scores TACO's 50 noise candidates in slices of
`CANDIDATES_PER_SLICE`. Scoring all 50 at once left the caching allocator
holding ~21 GiB for the life of the process, against ~7 GiB right after load,
which does not fit beside a co-tenant on a 48 GB card.

`load-cfn` pushes a trained checkpoint into an already-running server, which
`queue` runs automatically before dispatching taco jobs, because training
happens after the server was started.

Raw RoboTwin HDF5 and conversion staging are removed only after the LeRobot
dataset and PI0.5 feature tensor have both passed validation. LeRobot datasets,
features, CFN checkpoints, manifests, and evaluation summaries are retained.
