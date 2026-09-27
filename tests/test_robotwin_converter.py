import importlib.util
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ROBOTWIN = ROOT / "third_party" / "RoboTwin-official"
SPEC = importlib.util.spec_from_file_location(
    "convert_manifest_to_lerobot",
    ROOT / "scripts" / "robotwin_data" / "convert_manifest_to_lerobot.py",
)
CONVERTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONVERTER)

pytestmark = pytest.mark.skipif(
    not (OFFICIAL_ROBOTWIN / "data" / "decode_image_bit.py").is_file(),
    reason="an official RoboTwin checkout providing data/decode_image_bit.py is required",
)


@pytest.fixture(scope="module")
def decode_image_bit():
    return CONVERTER.load_image_decoder(OFFICIAL_ROBOTWIN)


def _legacy_jpeg(image: np.ndarray) -> bytes:
    """Mimic the pre-marker producers: an RGB array handed straight to imencode."""
    success, encoded = cv2.imencode(".jpg", image)
    assert success
    return encoded.tobytes()


def _marked_jpeg(image: np.ndarray) -> bytes:
    """Encode exactly the way the official collector does."""
    from data.decode_image_bit import images_encoding

    encoded, _ = images_encoding([image])
    return encoded[0]


@pytest.mark.parametrize("encode", [_legacy_jpeg, _marked_jpeg])
def test_load_episode_matches_robotwin_pi0_time_shift(tmp_path, decode_image_bit, encode):
    path = tmp_path / "episode0.hdf5"
    joints = np.arange(42, dtype=np.float64).reshape(3, 14)
    images = [np.full((8, 10, 3), value, dtype=np.uint8) for value in (10, 20, 30)]
    encoded = [encode(image) for image in images]
    max_length = max(map(len, encoded))

    with h5py.File(path, "w") as episode:
        episode.create_dataset("joint_action/vector", data=joints)
        for source in CONVERTER.LEGACY_CAMERA_PATHS.values():
            episode.create_dataset(source, data=encoded, dtype=f"S{max_length}")

    state, action, loaded_images = CONVERTER._load_episode(path, decode_image_bit)

    np.testing.assert_array_equal(state, joints[:-1].astype(np.float32))
    np.testing.assert_array_equal(action, joints[1:].astype(np.float32))
    assert state.dtype == np.float32
    assert all(value.shape == (2, 480, 640, 3) for value in loaded_images.values())
    assert all(np.allclose(value[:, 0, 0, 0], [10, 20], atol=2) for value in loaded_images.values())


@pytest.mark.parametrize("encode", [_legacy_jpeg, _marked_jpeg])
def test_load_episode_reads_official_xpolicylab_schema(tmp_path, decode_image_bit, encode):
    path = tmp_path / "episode_0000000.hdf5"
    left = np.arange(12, dtype=np.float32).reshape(2, 6)
    right = left + 20
    left_gripper = np.array([[0.1], [0.2]], dtype=np.float32)
    right_gripper = np.array([[0.3], [0.4]], dtype=np.float32)
    encoded = [encode(np.full((8, 10, 3), value, dtype=np.uint8)) for value in (10, 20)]
    max_length = max(map(len, encoded))

    with h5py.File(path, "w") as episode:
        for group_name, offset in (("state", 0), ("action", 100)):
            group = episode.create_group(group_name)
            group.create_dataset("left_arm_joint_states", data=left + offset)
            group.create_dataset("left_ee_joint_states", data=left_gripper + offset)
            group.create_dataset("right_arm_joint_states", data=right + offset)
            group.create_dataset("right_ee_joint_states", data=right_gripper + offset)
        for source in CONVERTER.OFFICIAL_CAMERA_PATHS.values():
            episode.create_dataset(source, data=encoded, dtype=f"S{max_length}")

    state, action, loaded_images = CONVERTER._load_episode(path, decode_image_bit)

    expected = np.concatenate([left, left_gripper, right, right_gripper], axis=1)
    np.testing.assert_array_equal(state, expected)
    np.testing.assert_array_equal(action, expected + 100)
    assert all(value.shape == (2, 480, 640, 3) for value in loaded_images.values())


@pytest.mark.parametrize("encode", [_legacy_jpeg, _marked_jpeg])
def test_decode_preserves_rgb_channel_order(tmp_path, decode_image_bit, encode):
    """Both stored byte formats must come back RGB, not channel-swapped."""
    path = tmp_path / "episode0.hdf5"
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    image[..., 0], image[..., 1], image[..., 2] = 240, 40, 10
    # The legacy layout stores one more frame than the state/action pairs.
    encoded = [encode(image)] * 3
    max_length = max(map(len, encoded))

    with h5py.File(path, "w") as episode:
        episode.create_dataset("joint_action/vector", data=np.zeros((3, 14), dtype=np.float32))
        for source in CONVERTER.LEGACY_CAMERA_PATHS.values():
            episode.create_dataset(source, data=encoded, dtype=f"S{max_length}")

    _, _, loaded_images = CONVERTER._load_episode(path, decode_image_bit)

    for frames in loaded_images.values():
        red, green, blue = (frames[0, 0, 0, channel].astype(int) for channel in range(3))
        assert red > 180 and green < 100 and blue < 60, (red, green, blue)
