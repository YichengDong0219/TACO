#!/usr/bin/env python3
"""Reproducible multi-task TACO pipeline for the fixed RoboTwin manifests."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = (
    "click_bell",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "pick_dual_bottles",
    "place_bread_skillet",
    "place_fan",
    "place_mouse_pad",
    "press_stapler",
    "scan_object",
    "stack_blocks_two",
    "stamp_seal",
    "turn_switch",
)
# Evaluation is scheduled slowest-first. A failed episode runs to its task's
# step_lim, so a task's cost is dominated by step_lim and its failure rate, not
# by its position in TASKS: the three expensive tasks below cost 3x what the
# cheapest ones do. Leaving them last parks the tail of a campaign on a machine
# that is otherwise idle (measured 2026-09-24: the final job ran ~1 h alone
# while seven slots sat empty), so they are launched first and the short tasks
# fill in behind them.
EVAL_SLOWEST_FIRST = (
    "stack_blocks_two",     # step_lim 800
    "scan_object",          # step_lim 500
    "place_bread_skillet",  # step_lim 500
    "place_mouse_pad",
    "place_fan",
    "stamp_seal",
    "pick_dual_bottles",
    "press_stapler",
    "turn_switch",
    "move_playingcard_away",
    "move_pillbottle_pad",
    "click_bell",
)
# Extraction is scheduled slowest-first too, but the cost ranking differs from
# evaluation's: an extraction job's cost is the number of chunks it walks in the
# dataset, which is not what makes an episode expensive. Measured on
# 2026-09-24 over the 12 datasets (outer progress-bar iterations): stack_blocks_two
# needs 295, the next task 137, the cheapest 73. Whichever server draws
# stack_blocks_two carries a fifth of the campaign on its own, so it should be
# handed out first.
EXTRACT_SLOWEST_FIRST = (
    "stack_blocks_two",     # 295 chunks
    "place_fan",            # 137
    "stamp_seal",           # 136
    "scan_object",          # 135
    "move_pillbottle_pad",  # 132
    "place_mouse_pad",      # 128
    "place_bread_skillet",  # 114
    "move_playingcard_away",  # 111
    "pick_dual_bottles",    # 111
    "press_stapler",        # 103
    "turn_switch",          # 86
    "click_bell",           # 73
)
PROTOCOL = "robotwin12_official_v1"
RAW_TAG = "demo_clean_official_v1"
DEFAULT_ARTIFACTS = ROOT / "artifacts" / PROTOCOL
DEFAULT_POLICY = Path("/home/dongyicheng/dsrl/pi05_robotwin_lerobot")
DEFAULT_ASSETS = Path("/home/dongyicheng/dsrl/RoboTwin-assets/assets")
DEFAULT_ROBOTWIN = ROOT / "third_party" / "RoboTwin-official"
DEFAULT_TOKENIZER = Path("/home/dongyicheng/dsrl/lerobot-dsrl-stage/tokenizer/paligemma-3b-pt-224")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path, expected_task: str, expected_count: int | None = None) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("task") != expected_task:
        raise ValueError(f"{path}: expected task {expected_task}, got {payload.get('task')}")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"{path}: expected a non-empty episode list")
    if expected_count is not None and len(episodes) != expected_count:
        raise ValueError(f"{path}: expected {expected_count} episodes")
    seeds = [item.get("episode_id") for item in episodes]
    if any(not isinstance(seed, int) for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError(f"{path}: seeds must be unique integers")
    for item in episodes:
        if not isinstance(item.get("instruction"), str) or not item["instruction"].strip():
            raise ValueError(f"{path}: missing instruction for seed {item.get('episode_id')}")
        for field in ("plan_success", "expert_success", "planned_joints_legal"):
            if item.get(field) is not True:
                raise ValueError(f"{path}: {field} is not true for seed {item['episode_id']}")
        if not isinstance(item.get("scene_info"), dict):
            raise ValueError(f"{path}: missing scene_info for seed {item['episode_id']}")
    ids_path = path.with_name("episode_ids.txt")
    ids = [int(value) for value in ids_path.read_text(encoding="utf-8").split()]
    if ids != seeds:
        raise ValueError(f"{ids_path}: IDs differ from episodes.json")
    return payload


def validate(args) -> int:
    audit = {"schema": "taco.robotwin12.manifests.v1", "tasks": {}}
    for task in TASKS:
        train_path = args.train_manifests / task / "episodes.json"
        eval_path = args.eval_manifests / task / "episodes.json"
        train = load_manifest(train_path, task, 30)
        evaluation = load_manifest(eval_path, task, 100)
        train_seeds = {item["episode_id"] for item in train["episodes"]}
        eval_seeds = {item["episode_id"] for item in evaluation["episodes"]}
        if train_seeds & eval_seeds:
            raise ValueError(f"{task}: train/eval seed overlap")
        audit["tasks"][task] = {
            "train_count": 30,
            "eval_count": 100,
            "train_sha256": sha256(train_path),
            "eval_sha256": sha256(eval_path),
            "overlap_count": 0,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


def bootstrap(args) -> int:
    robotwin = args.robotwin.resolve()
    asset_dir = robotwin / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted(args.assets.resolve().iterdir()):
        if not source.is_dir() or source.name == "__MACOSX":
            continue
        target = asset_dir / source.name
        if target.exists() or target.is_symlink():
            continue
        target.symlink_to(source, target_is_directory=True)
    missing = [
        name for name in ("embodiments", "objects", "background_texture")
        if not (asset_dir / name).exists()
    ]
    if missing:
        raise RuntimeError(f"Assets are not linked into {asset_dir}: {missing}")

    revision = subprocess.run(
        ["git", "-C", str(robotwin), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    remote = subprocess.run(
        ["git", "-C", str(robotwin), "remote", "get-url", "origin"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if "robotwin-Platform/RoboTwin" not in remote:
        raise RuntimeError(f"Not an official RoboTwin checkout: {remote}")
    print(json.dumps({
        "robotwin": str(robotwin), "repository": remote, "revision": revision,
        "assets": str(args.assets), "checkout_dirty": bool(subprocess.run(
            ["git", "-C", str(robotwin), "status", "--porcelain"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()),
    }))
    return 0


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _runtime_env(gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    conda_base_bin = Path(sys.executable).resolve().parents[3] / "bin"
    if (conda_base_bin / "ffmpeg").is_file():
        env["PATH"] = os.pathsep.join([str(conda_base_bin), env.get("PATH", "")])
    return env


def collect(args) -> int:
    manifest = args.train_manifests / args.task / "episodes.json"
    load_manifest(manifest, args.task, args.expected_episodes)
    env = _runtime_env(args.gpu)
    _run([
        sys.executable,
        str(ROOT / "scripts/robotwin_data/collect_manifest_official.py"),
        "--robotwin-root", str(args.robotwin),
        "--task", args.task,
        "--episodes-manifest", str(manifest.resolve()),
        "--output-dir", str(args.artifacts / "raw" / args.task / RAW_TAG),
    ], cwd=ROOT, env=env)
    return 0


def convert(args) -> int:
    manifest = args.train_manifests / args.task / "episodes.json"
    raw_dir = args.artifacts / "raw" / args.task / RAW_TAG
    output = args.artifacts / "datasets" / PROTOCOL / args.task
    if (output / "source_manifest.json").is_file():
        print(f"Dataset already converted: {output}")
        return 0
    # The converter cleans up after itself on error, but a process killed
    # mid-conversion cannot run that cleanup, and the leftover directory makes
    # every later attempt refuse to start.
    if output.exists():
        print(f"Removing incomplete dataset from an interrupted run: {output}", flush=True)
        shutil.rmtree(output)
    _run([
        sys.executable,
        str(ROOT / "scripts/robotwin_data/convert_manifest_to_lerobot.py"),
        "--raw-dir", str(raw_dir),
        "--episodes-manifest", str(manifest),
        "--output-root", str(args.artifacts / "datasets"),
        "--repo-id", f"{PROTOCOL}/{args.task}",
        "--robotwin-root", str(args.robotwin),
        "--fps", "15",
    ], cwd=ROOT)
    return 0


def extract(args) -> int:
    dataset_root = args.artifacts / "datasets" / PROTOCOL / args.task
    output_dir = args.artifacts / "features" / args.task
    feature_path = output_dir / "feature.pt"
    if output_dir.exists() and not feature_path.is_file():
        shutil.rmtree(output_dir)
    env = _runtime_env(args.gpu)
    env["HF_HUB_OFFLINE"] = "1"
    # collect.py resolves `cfn.pi05_rpc` and the pinned LeRobot checkout itself.
    env["PYTHONPATH"] = os.pathsep.join([
        str(ROOT / "third_party/lerobot/src"), str(ROOT / "cfn"), env.get("PYTHONPATH", "")
    ])
    if args.policy_server:
        env["TACO_POLICY_SERVER"] = args.policy_server
    if not feature_path.is_file():
        _run([
            sys.executable,
            str(ROOT / "scripts/collect_inernal_representation/pi05_robotwin2/collect.py"),
            f"--dataset.repo_id={PROTOCOL}/{args.task}",
            f"--dataset.root={dataset_root}",
            f"--output_dir={output_dir}",
            "--batch_size=32",
            "--num_workers=8",
            "--policy.type=pi05",
            f"--policy.pretrained_path={args.policy}",
            "--policy.push_to_hub=false",
            "--seed=42",
        ], cwd=ROOT, env=env)
    if not feature_path.is_file():
        raise RuntimeError(f"Feature extraction did not create {feature_path}")
    metadata = {
        "task": args.task,
        "feature_sha256": sha256(feature_path),
        "policy_sha256": sha256(args.policy / "model.safetensors"),
        "manifest_sha256": sha256(args.train_manifests / args.task / "episodes.json"),
    }
    (output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return 0


def collect_convert(args) -> int:
    """Collect expert demonstrations and convert them, keeping every artifact.

    The raw XPolicyLab HDF5 stays on disk: it is what a later re-conversion or a
    different dataset format would start from, and it costs a few hundred MB per
    task against a disk that has room for it.
    """
    collect(args)
    convert(args)
    dataset_dir = args.artifacts / "datasets" / PROTOCOL / args.task
    raw_dir = args.artifacts / "raw" / args.task / RAW_TAG
    usage = {
        "task": args.task,
        "dataset_bytes": sum(p.stat().st_size for p in dataset_dir.rglob("*") if p.is_file()),
        "raw_bytes": sum(p.stat().st_size for p in raw_dir.rglob("*") if p.is_file()),
        "free_bytes": shutil.disk_usage(args.artifacts).free,
    }
    (dataset_dir / "collection_usage.json").write_text(json.dumps(usage, indent=2) + "\n")
    return 0


def prepare(args) -> int:
    """Collect, convert and extract one task in a single process."""
    collect_convert(args)
    extract(args)
    (args.artifacts / "features" / args.task / "PREPARE_COMPLETE").write_text(
        "complete\n", encoding="utf-8"
    )
    return 0


def train(args) -> int:
    """Train the CFN on the features that exist.

    An unattended run should not lose a whole night because one task's features
    are missing, so the shortfall is recorded and reported rather than fatal —
    but a run with nothing to train on still fails.
    """
    available = [
        task for task in TASKS
        if (args.artifacts / "features" / task / "feature.pt").is_file()
    ]
    missing = [task for task in TASKS if task not in available]
    if not available:
        raise RuntimeError("No task produced features; nothing to train on")
    if missing:
        print(f"training without features for {len(missing)} task(s): {missing}", flush=True)
        (args.artifacts / "training_shortfall.json").write_text(
            json.dumps({"trained_tasks": available, "missing_tasks": missing}, indent=2) + "\n"
        )
    env = _runtime_env(args.gpu)
    _run([
        sys.executable,
        str(ROOT / "scripts/train_cfn/train_cfn.py"),
        "--output_dir", str(args.artifacts / "cfn"),
        "--feature_dir", str(args.artifacts / "features"),
        "--batch_size", "512",
        "--num_workers", "16",
        "--epochs", str(args.epochs),
        "--save_freq", "4",
        "--input_dim", "1024",
        "--cfn_output_dim", "20",
        "--cfn_hidden_dim", "1536",
        "--seed", "42",
        "--multi_feature_file",
        "--no-task_balanced",
    ], cwd=ROOT, env=env)
    return 0


def serve(args) -> int:
    env = _runtime_env(args.gpu)
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([
        str(ROOT / "third_party/lerobot/src"), str(ROOT / "cfn"), env.get("PYTHONPATH", "")
    ])
    command = [
        sys.executable, str(ROOT / "scripts/pi05_policy_server.py"),
        "--policy-path", str(args.policy), "--tokenizer-path", str(args.tokenizer),
        "--host", args.host, "--port", str(args.port),
    ]
    if args.cfn_checkpoint is not None:
        command += ["--cfn-checkpoint", str(args.cfn_checkpoint)]
    # Replace this process rather than spawning the server as a child: a child
    # outlives a killed parent, keeps holding the port, and silently serves the
    # old code to the next run.
    print("+", " ".join(command), flush=True)
    os.chdir(ROOT)
    os.execvpe(command[0], command, env)


def load_cfn(args) -> int:
    """Load a trained CFN into an already-running policy server.

    `serve --cfn-checkpoint` only covers a server started after training;
    training normally runs later, so the checkpoint has to be pushed into the
    running server before any taco evaluation can be served.
    """
    from cfn.pi05_rpc import PI05RPCClient

    servers = parse_policy_servers(args.policy_server)
    if not servers:
        raise RuntimeError("Loading a CFN requires --policy-server")
    checkpoint = args.cfn_checkpoint
    if checkpoint is None or not Path(checkpoint).is_file():
        raise FileNotFoundError(f"CFN checkpoint not found: {checkpoint}")
    for address in servers:
        client = PI05RPCClient(address)
        try:
            result = client.load_cfn(str(checkpoint))
        finally:
            client.close()
        print(json.dumps({"policy_server": address, **result}), flush=True)
    return 0


def smoke(args) -> int:
    """Run one episode end to end through the same code paths as the full pipeline."""
    if not args.policy_server:
        raise RuntimeError("Smoke test requires --policy-server pointing at a running PI0.5 server")
    task = "click_bell"
    smoke_root = args.artifacts / "smoke"
    complete = smoke_root / "COMPLETE"
    if complete.is_file():
        print(f"Smoke test already complete: {complete}")
        return 0

    smoke_args = argparse.Namespace(**vars(args))
    smoke_args.artifacts = smoke_root / "artifacts"
    smoke_args.train_manifests = smoke_root / "manifests"
    smoke_args.expected_episodes = 1
    smoke_args.task = task

    source = load_manifest(args.train_manifests / task / "episodes.json", task)
    payload = {**source, "episodes": source["episodes"][:1], "target_count": 1}
    manifest_dir = smoke_args.train_manifests / task
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "episodes.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (manifest_dir / "episode_ids.txt").write_text(
        f"{payload['episodes'][0]['episode_id']}\n", encoding="utf-8"
    )

    collect(smoke_args)
    convert(smoke_args)
    extract(smoke_args)

    features_root = smoke_args.artifacts / "features"
    cfn_dir = smoke_root / "cfn"
    cfn_checkpoint = cfn_dir / "model_epoch1.pt"
    if not cfn_checkpoint.is_file():
        _run([
            sys.executable, str(ROOT / "scripts/train_cfn/train_cfn.py"),
            "--output_dir", str(cfn_dir), "--feature_dir", str(features_root),
            "--batch_size", "512", "--num_workers", "4", "--epochs", "1",
            "--save_freq", "1", "--input_dim", "1024", "--cfn_output_dim", "20",
            "--cfn_hidden_dim", "1536", "--seed", "42", "--multi_feature_file",
            "--no-task_balanced",
        ], cwd=ROOT, env=_runtime_env(args.gpu))

    eval_args = argparse.Namespace(**vars(smoke_args))
    eval_args.eval_manifests = smoke_args.train_manifests
    eval_args.expected_episodes = 1
    eval_args.max_episodes = 1
    eval_args.cfn_checkpoint = cfn_checkpoint
    eval_args.output_dir = None
    eval_args.noise_seed = 42
    load_cfn(eval_args)
    for mode in ("baseline", "taco"):
        eval_args.mode = mode
        evaluate(eval_args)

    shutil.rmtree(smoke_args.artifacts / "raw", ignore_errors=True)
    complete.write_text("complete\n", encoding="utf-8")
    print(f"Smoke test complete: {complete}")
    return 0


def evaluate(args) -> int:
    if not args.policy_server:
        raise RuntimeError("Evaluation requires --policy-server pointing at a running PI0.5 server")
    manifest = args.eval_manifests / args.task / "episodes.json"
    load_manifest(manifest, args.task, args.expected_episodes)
    output = args.output_dir or args.artifacts / "eval" / args.mode / args.task
    env = _runtime_env(args.gpu)
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "cfn"), env.get("PYTHONPATH", "")])
    command = [
        sys.executable,
        str(ROOT / "scripts/robotwin_data/eval_manifest_official.py"),
        "--robotwin-root", str(args.robotwin),
        "--task", args.task,
        "--mode", args.mode,
        "--episodes-manifest", str(manifest),
        "--policy-server", args.policy_server,
        "--output-dir", str(output),
        "--max-episodes", str(args.max_episodes),
        "--noise-seed", str(args.noise_seed),
    ]
    if args.mode == "taco":
        command += ["--require-cfn-loaded"]
    _run(command, cwd=ROOT, env=env)
    revision = subprocess.run(
        ["git", "-C", str(args.robotwin), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    run_manifest = {
        "mode": args.mode,
        "task": args.task,
        "protocol": PROTOCOL,
        "policy_sha256": sha256(args.policy / "model.safetensors"),
        "cfn_sha256": sha256(args.cfn_checkpoint) if args.mode == "taco" else None,
        "episodes_manifest_sha256": sha256(manifest),
        # The one input two otherwise-identical runs may differ in, plus a hash
        # of the serving code that turns it into a candidate pool. Without these
        # two, seeded replicates are indistinguishable on disk — every other
        # field here is equal by construction.
        "noise_seed": args.noise_seed,
        "policy_server_sha256": sha256(ROOT / "scripts/pi05_policy_server.py"),
        "robotwin_revision": revision,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n")
    return 0


def _launchable_gpus(
    active_on_gpu: dict[int, int],
    reserved_mib: dict[int, int],
    *,
    jobs_per_gpu: int,
    reserve_mib: int,
    safety_mib: int,
) -> list[int]:
    """GPUs that can take one more job, judged on free memory alone.

    Other users' processes are respected, not avoided: a device stays eligible
    while nvidia-smi reports enough memory for our reservation plus the safety
    margin. Nothing another user runs is ever terminated or preempted, and a
    shared device is never over-committed past that margin. `reserved_mib` is
    subtracted on top of the live reading so a job that was just started counts
    before it has allocated anything.
    """
    free_mib = _gpu_free_memory()
    return sorted(
        gpu for gpu, free in free_mib.items()
        if active_on_gpu.get(gpu, 0) < jobs_per_gpu
        and free - reserved_mib.get(gpu, 0) - safety_mib >= reserve_mib
    )


def _active_counts(active: dict) -> dict[int, int]:
    counts: dict[int, int] = {}
    for entry in active.values():
        counts[entry[0]] = counts.get(entry[0], 0) + 1
    return counts


def _fill_groups(
    preferred: list[int], launchable: list[int], free_mib: dict[int, int]
) -> list[list[int]]:
    """Preferred devices first, everything else only once they are full.

    Heavy, latency-sensitive work takes the reserved cards while bulk work fills
    whatever else has room, spilling onto the reserved cards rather than idling.
    Without a preference the emptiest device goes first: filling by device index
    parks a handful of jobs on device 0 and leaves the rest of the machine idle.
    """
    head = [gpu for gpu in preferred if gpu in launchable]
    rest = sorted(
        (gpu for gpu in launchable if gpu not in head),
        key=lambda gpu: -free_mib.get(gpu, 0),
    )
    return [head, rest]


def parse_gpu_preference(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(part) for part in value.replace(",", " ").split() if part.strip()]


def parse_policy_servers(value: str | None) -> list[str]:
    if not value:
        return []
    return [part for part in value.replace(",", " ").split() if part]


def server_for(index: int, servers: list[str]) -> str | None:
    """Spread jobs over the running policy servers.

    Feature extraction and TACO's 50-candidate sweeps are what the campaign
    actually spends its time on, and a single server serializes every request on
    one GPU, so jobs are dealt out to servers round-robin.
    """
    if not servers:
        return None
    return servers[index % len(servers)]


def _gpu_free_memory() -> dict[int, int]:
    lines = subprocess.run(
        [
            "nvidia-smi", "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    return {int(index): int(free_mib) for index, free_mib in (line.split(",", 1) for line in lines)}


def _foreign_compute_gpus() -> set[int]:
    gpu_lines = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    uuid_to_index = {uuid.strip(): int(index) for index, uuid in (line.split(",", 1) for line in gpu_lines)}
    process_lines = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    foreign = set()
    for line in process_lines:
        if not line.strip():
            continue
        gpu_uuid, pid_text = (part.strip() for part in line.split(",", 1))
        try:
            if Path(f"/proc/{int(pid_text)}").stat().st_uid != os.getuid():
                foreign.add(uuid_to_index[gpu_uuid])
        except (FileNotFoundError, KeyError, ValueError):
            # Treat an uninspectable process conservatively as foreign.
            if gpu_uuid in uuid_to_index:
                foreign.add(uuid_to_index[gpu_uuid])
    return foreign


def _queue_failures(artifacts: Path, name: str, failures: list) -> None:
    if failures:
        (artifacts / f"{name}_failures.json").write_text(
            json.dumps(failures, indent=2) + "\n", encoding="utf-8"
        )
        print(f"{name}: {len(failures)} task(s) failed: {failures}", flush=True)


def collect_queue(args) -> int:
    """Collect and convert every task, filling whichever devices have room.

    A slot is held only for collection and conversion; feature extraction is a
    separate queue, so a device is not parked for hours on a job that is done
    with the GPU.
    """
    args.artifacts.mkdir(parents=True, exist_ok=True)
    lock_file = (args.artifacts / "collect_queue.lock").open("a+")
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pending = [
        task for task in TASKS
        if not (args.artifacts / "datasets" / PROTOCOL / task / "source_manifest.json").is_file()
    ]
    active: dict[int, tuple[int, str, subprocess.Popen, object]] = {}
    reserved_mib: dict[int, int] = {}
    failures: list[dict] = []
    attempts: dict[str, int] = {}
    preferred = parse_gpu_preference(args.preferred_gpus)

    def launchable_in(group: list[int]) -> list[int]:
        allowed = set(group)
        return [
            gpu for gpu in _launchable_gpus(
                _active_counts(active), reserved_mib,
                jobs_per_gpu=args.max_envs_per_gpu,
                reserve_mib=args.env_reserve_mib,
                safety_mib=args.safety_mib,
            )
            if gpu in allowed
        ]

    while pending or active:
        for pid, (gpu, task, process, log) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            del active[pid]
            reserved_mib[gpu] -= args.env_reserve_mib
            if code != 0:
                attempts[task] = attempts.get(task, 0) + 1
                if attempts[task] < args.max_attempts:
                    # One retry rides out a transient device or allocation
                    # failure without a human watching the run.
                    print(f"{task}: exit {code}, retrying ({attempts[task]})", flush=True)
                    pending.append(task)
                else:
                    failures.append(
                        {"task": task, "gpu": gpu, "exit_code": code, "attempts": attempts[task]}
                    )

        free_mib = _gpu_free_memory()
        for group in _fill_groups(preferred, sorted(free_mib), free_mib):
            while pending:
                targets = launchable_in(group)
                if not targets:
                    break
                gpu, task = targets[0], pending.pop(0)
                log_dir = args.artifacts / "collect_logs" / task
                log_dir.mkdir(parents=True, exist_ok=True)
                log = (log_dir / "job.log").open("a", encoding="utf-8")
                command = [
                    sys.executable, str(Path(__file__).resolve()),
                    "--artifacts", str(args.artifacts),
                    "--policy", str(args.policy),
                    "--robotwin", str(args.robotwin),
                    "--train-manifests", str(args.train_manifests),
                    "collect-convert", "--task", task, "--gpu", str(gpu),
                ]
                process = subprocess.Popen(
                    command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                )
                active[process.pid] = (gpu, task, process, log)
                reserved_mib[gpu] = reserved_mib.get(gpu, 0) + args.env_reserve_mib

        (args.artifacts / "collect_queue_status.json").write_text(
            json.dumps({
                "pending": pending,
                "active": [{"gpu": gpu, "task": task, "pid": pid}
                           for pid, (gpu, task, _, _) in active.items()],
                "foreign_gpus": sorted(_foreign_compute_gpus()),
                "failures": failures,
                "updated_at": time.time(),
            }, indent=2) + "\n"
        )
        if pending or active:
            time.sleep(15)
    _queue_failures(args.artifacts, "collect", failures)
    return 0


def extract_queue(args) -> int:
    """Extract PI0.5 features for every collected dataset.

    Extraction is served over RPC, so it needs no local GPU: one job per policy
    server keeps every server busy without piling clients onto a single lock.
    """
    args.artifacts.mkdir(parents=True, exist_ok=True)
    lock_file = (args.artifacts / "extract_queue.lock").open("a+")
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    servers = parse_policy_servers(args.policy_server)
    if not servers:
        raise RuntimeError("Feature extraction requires --policy-server")
    cost_order = {task: index for index, task in enumerate(EXTRACT_SLOWEST_FIRST)}
    pending = [
        task
        for task in sorted(TASKS, key=lambda name: cost_order.get(name, len(cost_order)))
        if not (args.artifacts / "features" / task / "feature.pt").is_file()
    ]
    slots = args.max_concurrent or len(servers)
    free_servers = list(servers)
    active: dict[int, tuple[str, str, subprocess.Popen, object]] = {}
    failures: list[dict] = []
    attempts: dict[str, int] = {}
    while pending or active:
        for pid, (server, task, process, log) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            del active[pid]
            free_servers.append(server)
            if code != 0:
                attempts[task] = attempts.get(task, 0) + 1
                if attempts[task] < args.max_attempts:
                    print(f"{task}: extraction exit {code}, retrying ({attempts[task]})", flush=True)
                    pending.append(task)
                else:
                    failures.append(
                        {"task": task, "server": server, "exit_code": code, "attempts": attempts[task]}
                    )

        while pending and len(active) < slots:
            task = pending.pop(0)
            # Hand each job to a server that is actually free. Pinning a task to
            # a fixed server by its index leaves the finish time at the mercy of
            # one pairing: measured 2026-09-24, the static mapping parked a
            # 406-chunk group on a card shared with another server and pushed the
            # projected wall from ~19 h to ~29 h, because every server then waits
            # on its own fixed share of work instead of on the whole queue.
            server = free_servers.pop(0)
            log_dir = args.artifacts / "extract_logs" / task
            log_dir.mkdir(parents=True, exist_ok=True)
            log = (log_dir / "job.log").open("a", encoding="utf-8")
            command = [
                sys.executable, str(Path(__file__).resolve()),
                "--artifacts", str(args.artifacts),
                "--policy", str(args.policy),
                "--train-manifests", str(args.train_manifests),
                "--policy-server", server,
                "extract", "--task", task, "--gpu", "0",
            ]
            process = subprocess.Popen(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            active[process.pid] = (server, task, process, log)

        (args.artifacts / "extract_queue_status.json").write_text(
            json.dumps({
                "pending": pending,
                "active": [{"server": server, "task": task, "pid": pid}
                           for pid, (server, task, _, _) in active.items()],
                "failures": failures,
                "updated_at": time.time(),
            }, indent=2) + "\n"
        )
        if pending or active:
            time.sleep(20)
    _queue_failures(args.artifacts, "extract", failures)
    return 0


def queue(args) -> int:
    args.artifacts.mkdir(parents=True, exist_ok=True)
    lock_file = (args.artifacts / "eval_queue.lock").open("a+")
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cost_order = {task: index for index, task in enumerate(EVAL_SLOWEST_FIRST)}
    pending = [
        (mode, task)
        for mode in ("baseline", "taco")
        for task in sorted(TASKS, key=lambda name: cost_order.get(name, len(cost_order)))
        if not (args.artifacts / "eval" / mode / task / "COMPLETE").is_file()
    ]
    if any(mode == "taco" for mode, _ in pending):
        # Training runs after the server was started, so the CFN is not in the
        # server yet; every pending taco job would otherwise fail its check.
        load_cfn(args)
    active: dict[int, tuple[int, str, str, subprocess.Popen, object]] = {}
    reserved_mib: dict[int, int] = {}
    preferred = parse_gpu_preference(args.preferred_gpus)

    def launch(gpu: int, mode: str, task: str) -> None:
        output = args.artifacts / "eval" / mode / task
        output.mkdir(parents=True, exist_ok=True)
        log = (output / "job.log").open("a", encoding="utf-8")
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--artifacts", str(args.artifacts),
            "--policy", str(args.policy),
            "--robotwin", str(args.robotwin),
            "--eval-manifests", str(args.eval_manifests),
        ]
        assigned = server_for(len(active), parse_policy_servers(args.policy_server))
        if assigned:
            command += ["--policy-server", assigned]
        command += [
            "eval", "--mode", mode, "--task", task, "--gpu", str(gpu),
            "--max-episodes", "100", "--expected-episodes", "100",
            "--noise-seed", str(args.noise_seed),
        ]
        if mode == "taco":
            command += ["--cfn-checkpoint", str(args.cfn_checkpoint)]
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        active[process.pid] = (gpu, mode, task, process, log)
        reserved_mib[gpu] = reserved_mib.get(gpu, 0) + args.eval_reserve_mib

    def launchable_in(group: list[int]) -> list[int]:
        allowed = set(group)
        return [
            gpu for gpu in _launchable_gpus(
                _active_counts(active),
                reserved_mib,
                jobs_per_gpu=args.max_jobs_per_gpu,
                reserve_mib=args.eval_reserve_mib,
                safety_mib=args.safety_mib,
            )
            if gpu in allowed
        ]

    while pending or active:
        for pid, (gpu, mode, task, process, log) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            del active[pid]
            reserved_mib[gpu] -= args.eval_reserve_mib
            if code != 0:
                raise RuntimeError(f"Evaluation failed: mode={mode} task={task} gpu={gpu} exit={code}")

        free_mib = _gpu_free_memory()
        candidates = sorted(free_mib)
        if args.allowed_gpus:
            allowed = set(args.allowed_gpus)
            candidates = [gpu for gpu in candidates if gpu in allowed]
        for group in _fill_groups(preferred, candidates, free_mib):
            while pending:
                targets = launchable_in(group)
                if not targets:
                    break
                mode, task = pending.pop(0)
                launch(targets[0], mode, task)

        state = {
            "pending": [{"mode": mode, "task": task} for mode, task in pending],
            "active": [{"gpu": gpu, "mode": mode, "task": task, "pid": pid} for pid, (gpu, mode, task, _, _) in active.items()],
            "updated_at": time.time(),
        }
        (args.artifacts / "eval_queue_status.json").write_text(json.dumps(state, indent=2) + "\n")
        if pending or active:
            time.sleep(30)
    return 0


def pipeline(args) -> int:
    """Run the campaign in the order the phases depend on each other.

    Collection fills every device that has room as room appears and is the only
    phase that runs until it is completely finished; training needs all twelve
    feature tensors, and evaluation needs the trained CFN loaded in the server.
    """
    args.preferred_gpus = args.prepare_preferred_gpus
    collect_queue(args)
    args.max_concurrent = args.max_concurrent_extracts
    extract_queue(args)
    args.gpu = args.train_gpu
    train(args)
    load_cfn(args)
    args.preferred_gpus = args.eval_preferred_gpus
    queue(args)
    return summarize(args)


def summarize(args) -> int:
    rows = []
    for task in TASKS:
        summaries = {}
        for mode in ("baseline", "taco"):
            path = args.artifacts / "eval" / mode / task / "summary.json"
            summaries[mode] = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "task": task,
            "baseline_success_rate": summaries["baseline"]["success_rate"],
            "taco_success_rate": summaries["taco"]["success_rate"],
            "delta": summaries["taco"]["success_rate"] - summaries["baseline"]["success_rate"],
        })
    output = args.artifacts / "eval" / "paired_summary.tsv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({
            "task": "macro",
            "baseline_success_rate": sum(row["baseline_success_rate"] for row in rows) / len(rows),
            "taco_success_rate": sum(row["taco_success_rate"] for row in rows) / len(rows),
            "delta": sum(row["delta"] for row in rows) / len(rows),
        })
    print(output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifests", type=Path, default=ROOT / "robotwin_train_episodes_12tasks")
    parser.add_argument("--eval-manifests", type=Path, default=ROOT / "robotwin_eval_episodes_12tasks")
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--robotwin", type=Path, default=DEFAULT_ROBOTWIN)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument(
        "--policy-server",
        help="Shared PI0.5 server address, for example 127.0.0.1:18080. The "
             "queues accept a comma-separated list and deal jobs out round-robin, "
             "since one server serializes every request on one GPU.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    command = subparsers.add_parser("validate")
    command.add_argument("--output", type=Path, default=DEFAULT_ARTIFACTS / "manifest_audit.json")
    command.set_defaults(func=validate)

    command = subparsers.add_parser("bootstrap")
    command.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    command.set_defaults(func=bootstrap)

    for name, func in (
        ("collect", collect),
        ("convert", convert),
        ("extract", extract),
        ("collect-convert", collect_convert),
        ("prepare", prepare),
    ):
        command = subparsers.add_parser(name)
        command.add_argument("--task", choices=TASKS, required=True)
        command.add_argument("--gpu", type=int, default=0)
        command.add_argument("--expected-episodes", type=int, default=30)
        command.set_defaults(func=func)

    command = subparsers.add_parser("train")
    command.add_argument("--gpu", type=int, default=0)
    command.add_argument("--epochs", type=int, default=16)
    command.set_defaults(func=train)

    command = subparsers.add_parser("serve")
    command.add_argument("--gpu", type=int, required=True)
    command.add_argument("--host", default="127.0.0.1")
    command.add_argument("--port", type=int, default=18080)
    command.add_argument("--cfn-checkpoint", type=Path)
    command.set_defaults(func=serve)

    command = subparsers.add_parser("load-cfn")
    command.add_argument("--cfn-checkpoint", type=Path, default=DEFAULT_ARTIFACTS / "cfn/model_epoch16.pt")
    command.set_defaults(func=load_cfn)

    command = subparsers.add_parser("collect-queue")
    command.add_argument("--max-attempts", type=int, default=2)
    command.add_argument("--max-envs-per-gpu", type=int, default=4)
    command.add_argument("--env-reserve-mib", type=int, default=6144)
    command.add_argument("--safety-mib", type=int, default=4096)
    command.add_argument(
        "--preferred-gpus",
        help="Devices to fill first, e.g. 1,2. Their order is the preference "
             "order; other devices are used once these are full.",
    )
    command.set_defaults(func=collect_queue)

    command = subparsers.add_parser("extract-queue")
    command.add_argument("--max-attempts", type=int, default=2)
    command.add_argument("--max-concurrent", type=int, default=0,
                         help="Concurrent extractions; defaults to one per policy server.")
    command.set_defaults(func=extract_queue)

    command = subparsers.add_parser("smoke")
    command.add_argument("--gpu", type=int, required=True)
    command.set_defaults(func=smoke)

    command = subparsers.add_parser("eval")
    command.add_argument("--mode", choices=("baseline", "taco"), required=True)
    command.add_argument("--task", choices=TASKS, required=True)
    command.add_argument("--gpu", type=int, required=True)
    command.add_argument("--max-episodes", type=int, default=100)
    command.add_argument("--expected-episodes", type=int, default=100)
    command.add_argument(
        "--noise-seed",
        type=int,
        default=42,
        help="Candidate-noise pool to score in taco mode. Two runs differing "
             "only in this are genuine replicates; two runs sharing it are the "
             "same deterministic run twice.",
    )
    command.add_argument("--cfn-checkpoint", type=Path, default=DEFAULT_ARTIFACTS / "cfn/model_epoch16.pt")
    command.add_argument("--output-dir", type=Path)
    command.set_defaults(func=evaluate)

    command = subparsers.add_parser("queue")
    command.add_argument("--cfn-checkpoint", type=Path, default=DEFAULT_ARTIFACTS / "cfn/model_epoch16.pt")
    command.add_argument("--noise-seed", type=int, default=42)
    command.add_argument("--max-jobs-per-gpu", type=int, default=1)
    command.add_argument("--eval-reserve-mib", type=int, default=6144)
    command.add_argument("--safety-mib", type=int, default=3072)
    command.add_argument(
        "--preferred-gpus",
        help="Devices to fill first, e.g. 1,2. Evaluation is on the critical "
             "path, so it takes the reserved cards before sharing others.",
    )
    command.add_argument(
        "--allowed-gpus",
        type=parse_gpu_preference,
        default=[],
        help="Restrict evaluation to these devices, e.g. 1,2. Unlike "
             "--preferred-gpus this never spills onto the other devices, so a "
             "campaign can be kept off cards that must stay free for something "
             "else. Defaults to every device.",
    )
    command.set_defaults(func=queue)

    command = subparsers.add_parser("summarize")
    command.set_defaults(func=summarize)

    command = subparsers.add_parser("pipeline")
    command.add_argument("--max-envs-per-gpu", type=int, default=4)
    command.add_argument("--env-reserve-mib", type=int, default=6144)
    command.add_argument("--safety-mib", type=int, default=4096)
    command.add_argument("--prepare-preferred-gpus", default=None,
                         help="Devices collection fills first; all are used when unset.")
    command.add_argument("--max-concurrent-extracts", type=int, default=0,
                         help="Concurrent extractions; defaults to one per policy server.")
    command.add_argument("--train-gpu", type=int, default=1)
    command.add_argument("--epochs", type=int, default=16)
    command.add_argument("--max-jobs-per-gpu", type=int, default=1)
    command.add_argument("--eval-reserve-mib", type=int, default=6144)
    command.add_argument("--eval-preferred-gpus", default=None,
                         help="Devices evaluation fills first, e.g. 1,2.")
    command.add_argument("--cfn-checkpoint", type=Path, default=DEFAULT_ARTIFACTS / "cfn/model_epoch16.pt")
    command.set_defaults(func=pipeline)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    for name in ("train_manifests", "eval_manifests", "artifacts", "robotwin", "policy", "tokenizer"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
