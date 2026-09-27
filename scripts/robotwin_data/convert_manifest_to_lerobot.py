#!/usr/bin/env python3
"""Convert exact-manifest RoboTwin HDF5 episodes directly to LeRobot v3."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


LEGACY_CAMERA_PATHS = {
    "cam_high": "/observation/head_camera/rgb",
    "cam_left_wrist": "/observation/left_camera/rgb",
    "cam_right_wrist": "/observation/right_camera/rgb",
}
OFFICIAL_CAMERA_PATHS = {
    "cam_high": "/vision/cam_head/colors",
    "cam_left_wrist": "/vision/cam_left_wrist/colors",
    "cam_right_wrist": "/vision/cam_right_wrist/colors",
}
MOTOR_NAMES = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
]


def load_image_decoder(robotwin_root: Path):
    """Load RoboTwin's `decode_image_bit`.

    Camera bits come in two byte formats: the legacy channel-reversed streams
    written by a bare `cv2.imencode`, and the conforming JPEGs that
    `encode_image_bit` stamps with an `XPL-RGB1` COM marker. `cv2.imdecode`
    rounds the legacy format back to RGB but returns BGR for the marked one, so
    hand-rolled decoding is wrong on exactly the episodes we collect now.
    `decode_image_bit` reads the marker and returns RGB for both.
    """
    root = str(robotwin_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from data.decode_image_bit import decode_image_bit
    except ImportError as exc:
        raise RuntimeError(
            f"--robotwin-root must point at a RoboTwin checkout providing "
            f"data/decode_image_bit.py: {robotwin_root}"
        ) from exc
    return decode_image_bit


def _decode_images(
    values: np.ndarray, *, decode_image_bit, path: Path, camera: str, size=(640, 480)
) -> np.ndarray:
    frames = np.asarray(decode_image_bit(values))
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB frames for {camera} in {path}, got {frames.shape}")
    return np.stack([cv2.resize(frame, size, interpolation=cv2.INTER_AREA) for frame in frames])


def _joint_vector(group: h5py.Group) -> np.ndarray:
    return np.concatenate(
        [
            group["left_arm_joint_states"][:],
            group["left_ee_joint_states"][:],
            group["right_arm_joint_states"][:],
            group["right_ee_joint_states"][:],
        ],
        axis=1,
    ).astype(np.float32)


def _load_episode(path: Path, decode_image_bit):
    with h5py.File(path, "r") as episode:
        if "/state/left_arm_joint_states" in episode:
            # Current official RoboTwin stores already aligned state/action pairs.
            state = _joint_vector(episode["/state"])
            action = _joint_vector(episode["/action"])
            images = {
                name: _decode_images(
                    episode[source][:], decode_image_bit=decode_image_bit, path=path, camera=name
                )
                for name, source in OFFICIAL_CAMERA_PATHS.items()
            }
        else:
            joints = episode["/joint_action/vector"][:].astype(np.float32)
            images = {
                name: _decode_images(
                    episode[source][:], decode_image_bit=decode_image_bit, path=path, camera=name
                )[:-1]
                for name, source in LEGACY_CAMERA_PATHS.items()
            }
            # Legacy RoboTwin records the state reached after every command.
            state = joints[:-1]
            action = joints[1:]
    if state.shape[1:] != (14,) or action.shape[1:] != (14,):
        raise ValueError(f"Expected 14-D state/action in {path}: {state.shape}, {action.shape}")
    if not len(state):
        raise ValueError(f"Episode must contain at least two frames: {path}")
    if any(len(value) != len(state) for value in images.values()) or len(action) != len(state):
        raise ValueError(f"Frame count mismatch in {path}")
    return state, action, images


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--episodes-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()
    decode_image_bit = load_image_decoder(args.robotwin_root)

    manifest = json.loads(args.episodes_manifest.read_text(encoding="utf-8"))
    episodes = manifest["episodes"]
    # The expert does not succeed on every seed, so the collection manifest —
    # not the seed manifest — says which episodes exist. Falling back to every
    # manifest entry keeps older collections readable.
    collection_path = Path(args.raw_dir) / "collection_manifest.json"
    if collection_path.is_file():
        collected = json.loads(collection_path.read_text(encoding="utf-8"))["collected_episodes"]
    else:
        collected = list(range(len(episodes)))
    if not collected:
        raise RuntimeError(f"Collection produced no episodes: {args.raw_dir}")

    def episode_path(index: int) -> Path:
        official = Path(args.raw_dir) / "data" / f"episode_{index:07d}.hdf5"
        legacy = Path(args.raw_dir) / "data" / f"episode{index}.hdf5"
        return official if official.is_file() else legacy

    paths = [episode_path(index) for index in collected]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing HDF5 episodes: {missing[:3]}")

    output = args.output_root / args.repo_id
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing dataset: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    features = {
        "observation.state": {"dtype": "float32", "shape": (14,), "names": [MOTOR_NAMES]},
        "action": {"dtype": "float32", "shape": (14,), "names": [MOTOR_NAMES]},
    }
    for camera in OFFICIAL_CAMERA_PATHS:
        features[f"observation.images.{camera}"] = {
            "dtype": "image",
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output,
        fps=args.fps,
        robot_type="aloha-agilex",
        features=features,
        use_videos=False,
        image_writer_processes=4,
        image_writer_threads=4,
    )
    try:
        for item, path in zip((episodes[index] for index in collected), paths):
            state, action, images = _load_episode(path, decode_image_bit)
            instruction = item["instruction"]
            for frame_index in range(len(state)):
                frame = {
                    "observation.state": torch.from_numpy(state[frame_index]).float(),
                    "action": torch.from_numpy(action[frame_index]).float(),
                    "task": instruction,
                }
                for camera in OFFICIAL_CAMERA_PATHS:
                    image = images[camera][frame_index]
                    if image.ndim == 3 and image.shape[-1] == 3:
                        image = np.transpose(image, (2, 0, 1))
                    frame[f"observation.images.{camera}"] = image
                dataset.add_frame(frame)
            dataset.save_episode()
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise

    source_manifest = {
        **manifest,
        "lerobot_media": "images",
        "videos_generated": False,
        "collected_episodes": collected,
        "episode_count": len(collected),
        "manifest_episode_count": len(episodes),
    }
    (output / "source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"dataset": str(output), "episodes": len(collected)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
