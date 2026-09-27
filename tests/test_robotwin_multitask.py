import importlib.util
import json
from pathlib import Path

import torch

from cfn.feature_dataset import cfn_feature_dataset


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "robotwin_multitask", ROOT / "scripts" / "robotwin_multitask.py"
)
PIPELINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PIPELINE)


def test_checked_manifests_have_expected_counts_and_no_overlap():
    for task in PIPELINE.TASKS:
        train = PIPELINE.load_manifest(
            ROOT / "robotwin_train_episodes_12tasks" / task / "episodes.json", task, 30
        )
        evaluation = PIPELINE.load_manifest(
            ROOT / "robotwin_eval_episodes_12tasks" / task / "episodes.json", task, 100
        )
        train_seeds = {item["episode_id"] for item in train["episodes"]}
        eval_seeds = {item["episode_id"] for item in evaluation["episodes"]}
        assert train_seeds.isdisjoint(eval_seeds)


def test_multitask_features_are_stably_concatenated(tmp_path):
    first = tmp_path / "a" / "feature.pt"
    second = tmp_path / "b" / "feature.pt"
    first.parent.mkdir()
    second.parent.mkdir()
    torch.save(torch.arange(8, dtype=torch.float32).reshape(4, 2), first)
    torch.save(torch.arange(4, dtype=torch.float32).reshape(2, 2), second)

    dataset = cfn_feature_dataset(tmp_path, multi_feature_file=True)
    assert dataset.task_counts == {"a": 4, "b": 2}
    assert torch.equal(dataset.features[:4], torch.arange(8, dtype=torch.float32).reshape(4, 2))
    assert torch.equal(dataset.features[4:], torch.arange(4, dtype=torch.float32).reshape(2, 2))
    assert dataset.task_names == ["a", "a", "a", "a", "b", "b"]
    assert (dataset[0]["CoinFlip_target"] == dataset[0]["CoinFlip_target"]).all()


def test_validate_writes_hash_audit(tmp_path):
    args = type("Args", (), {
        "train_manifests": ROOT / "robotwin_train_episodes_12tasks",
        "eval_manifests": ROOT / "robotwin_eval_episodes_12tasks",
        "output": tmp_path / "audit.json",
    })()
    assert PIPELINE.validate(args) == 0
    audit = json.loads(args.output.read_text())
    assert set(audit["tasks"]) == set(PIPELINE.TASKS)
    assert all(item["overlap_count"] == 0 for item in audit["tasks"].values())
