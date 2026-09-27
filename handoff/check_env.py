#!/usr/bin/env python3
"""Validate a TACO feature-extraction handoff package before spending hours on it.

Checks, in order of how expensive they are to discover later:
  1. every required file is present, and the 12 datasets/manifests are complete;
  2. every LeRobot patch that aligns the input with the training configuration
     is present (an unpatched copy silently degrades every result, so each is a
     hard failure);
  3. the processor pipeline really builds a 14-number state prompt.

Run from the package root:  python check_env.py
"""

from __future__ import annotations

import sys
from pathlib import Path


def _package_root(start: Path) -> Path:
    """The directory holding third_party/lerobot.

    This file ships inside handoff/ in the repository and at the root of a
    delivered bundle, so the root is located rather than assumed -- otherwise
    running it in place silently checks the wrong paths.
    """
    for candidate in (start, *start.parents):
        if (candidate / "third_party" / "lerobot" / "src" / "lerobot").is_dir():
            return candidate
    return start


def _read(path: Path) -> str:
    # A missing file is a failed check, not a crash: reporting what is wrong
    # with a package is the entire job of this script.
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


PKG = _package_root(Path(__file__).resolve().parent)
TASKS = (
    "click_bell", "move_pillbottle_pad", "move_playingcard_away", "pick_dual_bottles",
    "place_bread_skillet", "place_fan", "place_mouse_pad", "press_stapler",
    "scan_object", "stack_blocks_two", "stamp_seal", "turn_switch",
)
LEROBOT_SRC = PKG / "third_party" / "lerobot" / "src" / "lerobot"
LEROBOT = LEROBOT_SRC / "policies" / "pi05"
# `factory.py` sits one level above the policy packages, not beside them.
LEROBOT_FACTORY = LEROBOT_SRC / "policies" / "factory.py"

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    # Only surface the detail on failure: a passing check should not print the
    # message that describes what went wrong.
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{'  -- ' + detail if detail and not ok else ''}")
    if not ok:
        failures.append(label)


print("1. package layout")
for rel in (
    "pi05_robotwin_lerobot/model.safetensors",
    "pi05_robotwin_lerobot/config.json",
    "pi05_robotwin_lerobot/policy_preprocessor.json",
    "tokenizer/paligemma-3b-pt-224/tokenizer.json",
    "scripts/robotwin_multitask.py",
    "scripts/pi05_policy_server.py",
    "scripts/collect_inernal_representation/pi05_robotwin2/collect.py",
    "cfn/cfn/pi05_rpc.py",
    "third_party/lerobot/src/lerobot/policies/pi05/modeling_pi05.py",
):
    check(rel, (PKG / rel).exists())

datasets = PKG / "datasets" / "robotwin12_official_v1"
manifests = PKG / "robotwin_train_episodes_12tasks"
missing_ds = [t for t in TASKS if not (datasets / t / "meta" / "info.json").is_file()]
missing_mf = [t for t in TASKS if not (manifests / t / "episodes.json").is_file()]
check("12 LeRobot datasets", not missing_ds, f"missing: {missing_ds}")
check("12 train manifests", not missing_mf, f"missing: {missing_mf}")

print("\n2. LeRobot patches (input must match the training configuration)")
processor = _read(LEROBOT / "processor_pi05.py")
modeling = _read(LEROBOT / "modeling_pi05.py")
factory = _read(LEROBOT_FACTORY)
config = _read(LEROBOT / "configuration_pi05.py")
relative_actions = LEROBOT_SRC / "processor" / "relative_action_processor.py"

# openpi tokenizes the state before padding it, so the prompt carries the real
# dims only; padding to max_state_dim here would append 18 "128" bins.
pads_before_digitizing = "pad_vector(state, self.max_state_dim)" in processor
check("state is NOT padded before digitizing", not pads_before_digitizing,
      "processor_pi05.py still pads to max_state_dim before building the prompt")

# The caller maps the resized images with `* 2 - 1`; a pad value of -1.0 would
# put the letterbox band at -3.0.
check("image pad constant is 0.0", "else 0.0" in modeling and "else -1.0" not in modeling,
      "modeling_pi05.py still pads float images with -1.0")

# The postprocessor's relative-to-absolute step reuses the observation state the
# last preprocessor call cached, so it has to be handed the preprocessor's step
# explicitly. Unwired, the step has nothing to undo.
wired = "relative_step = next(" in factory and "step.relative_step = relative_step" in factory
check("postprocessor is wired to the preprocessor's relative step", wired,
      "factory.py does not hand AbsoluteActionsProcessorStep the "
      "preprocessor's RelativeActionsProcessorStep")
check("relative action processor exists", relative_actions.is_file(),
      f"missing {relative_actions}")

# Fields written into config.json by the LeRobot version that converted this
# checkpoint. Absent them the checkpoint fails to load outright -- loud, but it
# costs a policy-server start to discover.
compat = "use_peft" in config and "pretrained_revision" in config
check("checkpoint compatibility fields are declared", compat,
      "configuration_pi05.py is missing use_peft / pretrained_revision, so the "
      "converted checkpoint's config.json will not load")

print("\n3. the pipeline actually builds a 14-number state prompt")
try:
    sys.path.insert(0, str(PKG / "third_party" / "lerobot" / "src"))
    import torch  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors

    policy_path = PKG / "pi05_robotwin_lerobot"
    config = PreTrainedConfig.from_pretrained(policy_path, local_files_only=True)
    pre, _ = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=policy_path,
        preprocessor_overrides={
            "tokenizer_processor": {
                "tokenizer_name": str(PKG / "tokenizer" / "paligemma-3b-pt-224")
            }
        },
    )
    batch = {
        "observation.state": torch.zeros(1, 14),
        "observation.images.cam_high": torch.zeros(1, 3, 240, 320),
        "observation.images.cam_left_wrist": torch.zeros(1, 3, 240, 320),
        "observation.images.cam_right_wrist": torch.zeros(1, 3, 240, 320),
        "task": ["press the bell"],
    }
    prompt = pre(batch)["task"][0]
    state_field = prompt.split("State: ")[1].split(";")[0].split()
    check("prompt carries exactly 14 state numbers", len(state_field) == 14,
          f"got {len(state_field)}: {prompt!r}")
except Exception as exc:  # noqa: BLE001 - report, do not crash the whole check
    check("prompt check", False, f"{type(exc).__name__}: {exc}")

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {failures}")
    print("Do not start a long extraction run until these pass.")
    raise SystemExit(1)
print("All checks passed. Start with:  GPUS=\"0 1 2 3\" ./run_extraction.sh")
