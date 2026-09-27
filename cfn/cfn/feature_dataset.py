
import hashlib
from pathlib import Path

import numpy as np
import torch

class CoinFlipMaker:
    def __init__(self, output_dimensions=20, only_zero_flips=False):
        self.output_dimensions = output_dimensions
        self.only_zero_flips = only_zero_flips

    def __call__(self, seed):
        if self.only_zero_flips:
            return np.zeros(self.output_dimensions, dtype=np.float32)
        rng = np.random.RandomState(seed)
        return (2 * rng.binomial(1, 0.5, size=self.output_dimensions) - 1).astype(np.float32)


def _stable_seed(task: str, sample_index: int) -> int:
    digest = hashlib.sha256(f"{task}:{sample_index}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _load_feature_file(path: Path) -> tuple[torch.Tensor, str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, torch.Tensor):
        features = payload
        task = path.parent.name if path.name == "feature.pt" else path.stem
    elif isinstance(payload, dict) and isinstance(payload.get("features"), torch.Tensor):
        features = payload["features"]
        task = str(payload.get("task") or path.parent.name)
    else:
        raise TypeError(f"Unsupported feature payload in {path}")
    if features.ndim != 2:
        raise ValueError(f"Expected [samples, dim] features in {path}, got {tuple(features.shape)}")
    return features, task


class cfn_feature_dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        feature_dir,
        multi_feature_file=False,
        output_dimensions=20,
    ):
        source = Path(feature_dir)
        paths = sorted(source.rglob("*.pt")) if multi_feature_file else [source]
        if not paths:
            raise FileNotFoundError(f"No .pt feature files under {source}")

        feature_parts = []
        task_names = []
        local_indices = []
        for path in paths:
            features, task = _load_feature_file(path)
            feature_parts.append(features)
            task_names.extend([task] * len(features))
            local_indices.extend(range(len(features)))

        self.features = torch.cat(feature_parts, dim=0)
        self.task_names = task_names
        self.local_indices = local_indices
        self.CoinFlipMaker = CoinFlipMaker(output_dimensions=output_dimensions)
        self.task_counts = {task: task_names.count(task) for task in sorted(set(task_names))}
        self.sample_weights = torch.tensor(
            [1.0 / self.task_counts[task] for task in task_names], dtype=torch.double
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx) -> dict:
        task = self.task_names[idx]
        return {
            "feature": self.features[idx],
            "CoinFlip_target": self.CoinFlipMaker(_stable_seed(task, self.local_indices[idx])),
        }
