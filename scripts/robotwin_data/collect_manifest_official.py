#!/usr/bin/env python3
"""Collect exact-manifest demonstrations with an unmodified official RoboTwin checkout."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import yaml


class SceneMismatch(RuntimeError):
    """The seed produced a scene the manifest does not describe.

    Unlike a failed expert this cannot be recorded and moved past: the pinned
    instruction belongs to a different scene, so the episode would be wrong.
    """


def _assert_scene_info(expected: dict, actual: dict) -> None:
    mismatches = {
        key: (value, actual.get(key))
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise SceneMismatch(f"Scene info does not match manifest: {mismatches}")


def _expert_outcome(task_env) -> dict:
    legal, excess = task_env.planned_joints_legal()
    return {
        "plan_success": bool(task_env.plan_success),
        "expert_success": bool(task_env.check_success()),
        "planned_joints_legal": bool(legal),
        "joint_excess": float(excess),
    }


def _traj_is_readable(path: Path) -> bool:
    """A trajectory is reusable only if it still deserializes."""
    try:
        with path.open("rb") as stream:
            import pickle

            trajectory = pickle.load(stream)
        return bool(trajectory.get("left_joint_path") is not None)
    except Exception:
        return False


def _episode_is_complete(path: Path) -> bool:
    """Whether a previously written episode is usable as-is.

    Resuming trusts files on disk, so a run killed mid-write would otherwise
    leave a truncated HDF5 that later looks finished. Reading the datasets back
    is cheap next to collecting the episode again.
    """
    if not path.is_file():
        return False
    try:
        import h5py

        with h5py.File(path, "r") as episode:
            return (
                "/state/left_arm_joint_states" in episode
                and "/action/left_arm_joint_states" in episode
                and "/vision/cam_head/colors" in episode
            )
    except Exception:
        return False


def _merge_cache_to_hdf5(task_env, output: Path, instruction: str, frequency: int) -> None:
    from envs.utils.pkl2hdf5 import (
        append_data_to_structure,
        create_xpolicylab_hdf5,
        load_pkl_file,
        parse_dict_structure,
    )

    cache = Path(task_env.folder_path["cache"])
    pkl_paths = sorted(
        (path for path in cache.glob("*.pkl") if path.stem.isdigit()),
        key=lambda path: int(path.stem),
    )
    if not pkl_paths:
        raise FileNotFoundError(f"No frame cache found in {cache}")
    if [int(path.stem) for path in pkl_paths] != list(range(len(pkl_paths))):
        raise ValueError(f"Non-contiguous frame cache in {cache}")

    data = parse_dict_structure(load_pkl_file(pkl_paths[0]))
    for path in pkl_paths:
        append_data_to_structure(data, load_pkl_file(path))
    output.parent.mkdir(parents=True, exist_ok=True)
    create_xpolicylab_hdf5(data, output, [instruction], frequency)


def _configure_task(robotwin: Path, task_name: str, output: Path, episode_count: int):
    sys.path.insert(0, str(robotwin))
    sys.path.insert(0, str(robotwin / "scripts"))
    from envs import CONFIGS_PATH
    from scripts.collect_data import class_decorator, get_embodiment_config

    config_path = Path(CONFIGS_PATH) / "demo_clean.yml"
    args = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    args.update(
        {
            "task_name": task_name,
            "task_config": "demo_clean",
            "episode_num": episode_count,
            "use_seed": True,
            "save_path": str(output),
            "render_freq": 0,
            "eval_video_log": False,
        }
    )

    embodiment_types = yaml.safe_load(
        (Path(CONFIGS_PATH) / "_embodiment_config.yml").read_text(encoding="utf-8")
    )
    embodiment = args["embodiment"]
    if len(embodiment) != 1:
        raise ValueError("robotwin12_v1 expects one shared dual-arm embodiment")
    robot_file = embodiment_types[embodiment[0]]["file_path"]
    args.update(
        {
            "left_robot_file": robot_file,
            "right_robot_file": robot_file,
            "dual_arm_embodied": True,
            "embodiment_name": str(embodiment[0]),
            "left_embodiment_config": get_embodiment_config(robot_file),
            "right_embodiment_config": get_embodiment_config(robot_file),
        }
    )
    return class_decorator(task_name), args


def collect(robotwin: Path, task_name: str, manifest_path: Path, output: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("task") != task_name:
        raise ValueError(f"Manifest task mismatch: {manifest.get('task')} != {task_name}")
    episodes = manifest["episodes"]
    task_env, task_args = _configure_task(robotwin, task_name, output, len(episodes))
    traj_dir = output / "_traj_data"
    traj_dir.mkdir(parents=True, exist_ok=True)

    # The expert's scripted planner is not reproducible: the same seed yields a
    # different joint path on every attempt, and whether the executed path
    # satisfies the task's tolerance check flips with it. A failed expert is
    # therefore recorded and stepped over rather than allowed to abort the run;
    # only a scene that does not match the manifest is fatal, because there the
    # manifest's instruction belongs to a different scene.
    task_args["need_plan"] = True
    records: dict[int, dict] = {}
    for index, item in enumerate(episodes):
        seed = int(item["episode_id"])
        traj_path = traj_dir / f"episode{index}.pkl"
        record = {"episode_id": seed}
        if traj_path.is_file() and _traj_is_readable(traj_path):
            records[index] = {**record, "expert": "planned"}
            continue
        traj_path.unlink(missing_ok=True)
        try:
            task_env.setup_demo(now_ep_num=index, seed=seed, **task_args)
        except Exception as exc:
            records[index] = {**record, "expert": "env_error", "error": f"{type(exc).__name__}: {exc}"}
            task_env.close_env(clear_cache=((index + 1) % task_args["clear_cache_freq"] == 0))
            print(f"[{index}] seed={seed} environment failed to build: {exc}", flush=True)
            continue
        try:
            episode_info = task_env.play_once()
            outcome = _expert_outcome(task_env)
            if not all(outcome[key] for key in ("plan_success", "expert_success", "planned_joints_legal")):
                records[index] = {**record, "expert": "failed", **outcome}
                print(f"[{index}] seed={seed} expert did not succeed: {outcome}", flush=True)
                continue
            _assert_scene_info(item["scene_info"], episode_info["info"])
            task_env.save_traj_data(index)
            records[index] = {**record, "expert": "planned", **outcome}
        except SceneMismatch:
            raise
        except Exception as exc:
            records[index] = {**record, "expert": "error", "error": f"{type(exc).__name__}: {exc}"}
            print(f"[{index}] seed={seed} expert raised: {exc}", flush=True)
        finally:
            task_env.close_env(clear_cache=((index + 1) % task_args["clear_cache_freq"] == 0))

    planned = sorted(index for index, record in records.items() if record["expert"] == "planned")
    print(f"planned {len(planned)}/{len(episodes)} episodes", flush=True)
    (output / "seed.txt").write_text(
        " ".join(str(episodes[index]["episode_id"]) for index in planned) + "\n", encoding="utf-8"
    )

    task_args.update({"need_plan": False, "save_data": True, "render_freq": 0})
    scene_info_path = output / "scene_info.json"
    scene_info = json.loads(scene_info_path.read_text()) if scene_info_path.is_file() else {}
    for index in planned:
        item = episodes[index]
        seed = int(item["episode_id"])
        hdf5_path = output / "data" / f"episode_{index:07d}.hdf5"
        if _episode_is_complete(hdf5_path):
            records[index]["replay"] = "replayed"
            records[index].setdefault("replay_success", None)
            continue
        hdf5_path.unlink(missing_ok=True)
        try:
            task_env.setup_demo(now_ep_num=index, seed=seed, **task_args)
        except Exception as exc:
            records[index]["replay"] = "env_error"
            records[index]["replay_error"] = f"{type(exc).__name__}: {exc}"
            task_env.close_env(clear_cache=((index + 1) % task_args["clear_cache_freq"] == 0))
            print(f"[{index}] seed={seed} replay environment failed to build: {exc}", flush=True)
            continue
        trajectory = task_env.load_tran_data(index)
        task_args["left_joint_path"] = trajectory["left_joint_path"]
        task_args["right_joint_path"] = trajectory["right_joint_path"]
        task_env.set_path_lst(task_args)
        try:
            episode_info = task_env.play_once()
            _assert_scene_info(item["scene_info"], episode_info["info"])
            scene_info[f"episode_{index:07d}"] = episode_info
            instruction_dir = output / "instruction"
            instruction_dir.mkdir(parents=True, exist_ok=True)
            (instruction_dir / f"episode_{index:07d}.json").write_text(
                json.dumps({"seen": [item["instruction"]], "unseen": []}, indent=2) + "\n",
                encoding="utf-8",
            )
            task_env.close_env(clear_cache=((index + 1) % task_args["clear_cache_freq"] == 0))
            _merge_cache_to_hdf5(task_env, hdf5_path, item["instruction"], task_args["save_freq"])
            # The replay follows a recorded path, so its outcome is a property of
            # the episode, not a reason to discard it.
            records[index]["replay"] = "replayed"
            records[index]["replay_success"] = bool(task_env.check_success())
        except SceneMismatch:
            raise
        except Exception as exc:
            records[index]["replay"] = "error"
            records[index]["replay_error"] = f"{type(exc).__name__}: {exc}"
            print(f"[{index}] seed={seed} replay raised: {exc}", flush=True)
        finally:
            task_env.remove_data_cache()
        scene_info_path.write_text(
            json.dumps(scene_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    revision = subprocess.run(
        ["git", "-C", str(robotwin), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    collected = sorted(
        index for index, record in records.items()
        if (output / "data" / f"episode_{index:07d}.hdf5").is_file()
    )
    outcomes = Counter(record["expert"] for record in records.values())
    (output / "collection_manifest.json").write_text(
        json.dumps(
            {
                "robotwin_repository": "https://github.com/robotwin-Platform/RoboTwin.git",
                "robotwin_revision": revision,
                "episodes_manifest": str(manifest_path.resolve()),
                "manifest_episode_count": len(episodes),
                "episode_count": len(collected),
                "expert_outcomes": dict(sorted(outcomes.items())),
                "collected_episodes": collected,
                "episodes": {str(index): records[index] for index in sorted(records)},
                "raw_video_generated": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"collected {len(collected)}/{len(episodes)} episodes: {dict(sorted(outcomes.items()))}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episodes-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    os.chdir(args.robotwin_root.resolve())
    collect(
        args.robotwin_root.resolve(),
        args.task,
        args.episodes_manifest.resolve(),
        args.output_dir.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
