#!/usr/bin/env python3
"""Evaluate baseline or TACO selection in the official RoboTwin environment."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from cfn.pi05_rpc import RemotePI05Model


def _wilson(successes: int, total: int) -> list[float]:
    if not total:
        return [0.0, 0.0]
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return [center - margin, center + margin]


def _observation(model: RemotePI05Model, instruction: str, observation: dict) -> None:
    model.set_language(instruction)
    model.update_observation_window(
        [
            observation["observation"]["head_camera"]["rgb"],
            observation["observation"]["right_camera"]["rgb"],
            observation["observation"]["left_camera"]["rgb"],
        ],
        np.asarray(observation["joint_action"]["vector"]),
    )


def evaluate(args) -> dict:
    robotwin = args.robotwin_root.resolve()
    sys.path.insert(0, str(robotwin))
    sys.path.insert(0, str(robotwin / "scripts"))
    from scripts.eval_policy_xpolicylab import class_decorator, load_task_args

    manifest = json.loads(args.episodes_manifest.read_text(encoding="utf-8"))
    if manifest.get("task") != args.task:
        raise ValueError("Evaluation manifest task mismatch")
    episodes = manifest["episodes"][: args.max_episodes]

    task_args, _ = load_task_args(
        {
            "task_name": args.task,
            "task_config": "demo_clean",
            "ckpt_setting": "base",
            "policy_name": "pi05",
        }
    )
    task_args.update({"eval_mode": True, "eval_video_log": False, "render_freq": 0})
    task_env = class_decorator(args.task)
    model = RemotePI05Model(args.policy_server, args.mode, args.noise_seed)
    if args.require_cfn_loaded and not model.client.health().get("cfn_loaded"):
        raise RuntimeError("TACO evaluation requires a CFN checkpoint loaded on the policy server")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("episodes.jsonl", "summary.json", "summary.tsv", "COMPLETE"):
        (args.output_dir / name).unlink(missing_ok=True)

    rows = []
    for index, item in enumerate(episodes):
        started = time.time()
        success = False
        error = None
        length = 0
        try:
            task_env.setup_demo(
                now_ep_num=index,
                seed=int(item["episode_id"]),
                is_test=True,
                **task_args,
            )
            # RoboTwin only fills `info["info"]` inside the expert's play_once,
            # which evaluation never runs, so the manifest's scene_info cannot
            # be re-derived here. Collection asserts it against the same seeds
            # on the same checkout instead, and the seed alone fixes the scene.
            task_env.set_instruction(instruction=item["instruction"])
            model.reset_obsrvationwindows()
            while not (task_env.eval_success or task_env.take_action_cnt >= task_env.step_lim):
                observation = task_env.get_obs()
                _observation(model, item["instruction"], observation)
                actions = model.get_action()[: model.pi0_step]
                for action in actions:
                    task_env.take_action(action)
                    length = int(task_env.take_action_cnt)
                    if task_env.eval_success or task_env.take_action_cnt >= task_env.step_lim:
                        break
                if task_env.eval_success:
                    success = True
                    break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                task_env.close_env(clear_cache=((index + 1) % task_args["clear_cache_freq"] == 0))
            except Exception as close_exc:
                error = error or f"close_env: {type(close_exc).__name__}: {close_exc}"

        row = {
            "method": args.mode,
            "task": args.task,
            "seed": int(item["episode_id"]),
            "instruction": item["instruction"],
            # The scene the expert saw when this seed was recorded, carried
            # through for the record; evaluation cannot re-derive it.
            "manifest_scene_info": item["scene_info"],
            # Which candidate-noise pool this episode ran against. Recorded
            # because it is the one input that two otherwise-identical runs can
            # differ in — without it the products are indistinguishable.
            "noise_seed": args.noise_seed,
            "success": success,
            "episode_length": length,
            "elapsed_s": time.time() - started,
            "error": error,
        }
        rows.append(row)
        with (args.output_dir / "episodes.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{index + 1}/{len(episodes)}] seed={item['episode_id']} success={success} error={error}")

    # The seed set is fixed, so a failed or errored episode is an outcome to
    # record, not a reason to abandon the remaining seeds. Only the successes
    # count towards the rate; episodes that never ran are reported separately
    # so a low rate cannot hide a broken run.
    successes = sum(row["success"] for row in rows)
    errors = sum(1 for row in rows if row["error"])
    summary = {
        "method": args.mode,
        "task": args.task,
        "noise_seed": args.noise_seed,
        "n_episodes": len(rows),
        "n_success": successes,
        "n_errors": errors,
        "success_rate": successes / len(rows),
        "success_rate_ci95": _wilson(successes, len(rows)),
        "robotwin_revision": os.popen(f"git -C {robotwin} rev-parse HEAD").read().strip(),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.tsv").write_text(
        "method\ttask\tn_episodes\tn_success\tn_errors\tsuccess_rate\n"
        f"{args.mode}\t{args.task}\t{len(rows)}\t{successes}\t{errors}\t{summary['success_rate']:.8f}\n",
        encoding="utf-8",
    )
    (args.output_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--mode", choices=("baseline", "taco"), required=True)
    parser.add_argument("--episodes-manifest", type=Path, required=True)
    parser.add_argument("--policy-server", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=100)
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--require-cfn-loaded", action="store_true")
    args = parser.parse_args()
    os.chdir(args.robotwin_root.resolve())
    print(json.dumps(evaluate(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
