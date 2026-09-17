"""Pure NumPy conversions between this SimplerEnv fork and GR00T N1.7."""

from __future__ import annotations

import math
from typing import Dict

import numpy as np


LANGUAGE_KEY = "annotation.human.action.task_description"
ACTION_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
BRIDGE_DEFAULT_ROTATION = np.asarray(
    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
    dtype=np.float64,
)


def quat_wxyz_to_euler(quaternion: np.ndarray) -> np.ndarray:
    """Convert a wxyz quaternion to intrinsic xyz roll/pitch/yaw."""
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        raise ValueError("The end-effector quaternion has zero norm")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sin_pitch = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    pitch = math.asin(sin_pitch)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.asarray([roll, pitch, yaw], dtype=np.float32)


def quat_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert one wxyz quaternion to a 3x3 rotation matrix."""
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        raise ValueError("The end-effector quaternion has zero norm")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_euler_xyz(matrix: np.ndarray) -> np.ndarray:
    """Match transforms3d.mat2euler's default static xyz convention."""
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 rotation matrix, got {value.shape}")
    cy = math.sqrt(value[0, 0] * value[0, 0] + value[1, 0] * value[1, 0])
    if cy > 4.0 * np.finfo(float).eps:
        roll = math.atan2(value[2, 1], value[2, 2])
        pitch = math.atan2(-value[2, 0], cy)
        yaw = math.atan2(value[1, 0], value[0, 0])
    else:
        roll = math.atan2(-value[1, 2], value[1, 1])
        pitch = math.atan2(-value[2, 0], cy)
        yaw = 0.0
    return np.asarray([roll, pitch, yaw], dtype=np.float32)


class GripperActionAdapter:
    """Convert GR00T's absolute [0,1] gripper prediction for SimplerEnv."""

    def __init__(self, policy_setup: str, sticky_steps: int = 15):
        if policy_setup not in {"google_robot", "widowx_bridge"}:
            raise ValueError(f"Unsupported GR00T policy_setup: {policy_setup}")
        self.policy_setup = policy_setup
        self.sticky_steps = int(sticky_steps)
        self.reset()

    def reset(self) -> None:
        self.sticky_action_is_on = False
        self.sticky_gripper_action = 0.0
        self.gripper_action_repeat = 0

    def __call__(self, action: float | np.ndarray) -> np.ndarray:
        value = float(np.asarray(action).reshape(-1)[0])
        if self.policy_setup == "widowx_bridge":
            return np.asarray([2.0 * (value > 0.5) - 1.0], dtype=np.float32)

        # NVIDIA's GoogleFractalEnv: [0,1] absolute open amount becomes a
        # relative command, held for 15 simulator steps after a threshold event.
        relative = 1.0 - 2.0 * value
        if abs(relative) > 0.5 and not self.sticky_action_is_on:
            self.sticky_action_is_on = True
            self.sticky_gripper_action = relative
        if self.sticky_action_is_on:
            self.gripper_action_repeat += 1
            relative = self.sticky_gripper_action
        if self.gripper_action_repeat == self.sticky_steps:
            self.sticky_action_is_on = False
            self.gripper_action_repeat = 0
            self.sticky_gripper_action = 0.0
        return np.asarray([relative], dtype=np.float32)


def collapse_segmentation_mask(mask: np.ndarray | None, image_shape: tuple[int, int]) -> np.ndarray | None:
    """Return one HxW boolean target mask from the evaluator's possible mask layouts."""
    if mask is None:
        return None
    value = np.asarray(mask)
    if value.ndim == 2:
        collapsed = value
    elif value.ndim == 3 and value.shape[:2] == image_shape:
        collapsed = np.any(value, axis=-1)
    elif value.ndim == 3 and value.shape[-2:] == image_shape:
        collapsed = np.any(value, axis=0)
    else:
        raise ValueError(
            f"Unsupported segmentation mask shape {value.shape}; expected HxW, HxWxN, or NxHxW"
        )
    if collapsed.shape != image_shape:
        raise ValueError(f"Mask/image shape mismatch: {collapsed.shape} != {image_shape}")
    return collapsed.astype(bool, copy=False)


def build_flat_observation(
    image: np.ndarray,
    instruction: str,
    proprio: np.ndarray,
    policy_setup: str,
) -> Dict[str, object]:
    """Build the flat observation accepted by ``Gr00tSimPolicyWrapper``.

    The local evaluator exposes proprio as xyz + quaternion(wxyz) + gripper.
    GR00T's Google configuration expects xyzw, while WidowX expects Euler xyz
    and a compatibility padding state.
    """
    image = np.asarray(image)
    proprio = np.asarray(proprio, dtype=np.float32).reshape(-1)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 HxWx3 RGB image, got {image.dtype} {image.shape}")
    if proprio.size != 8:
        raise ValueError(f"Expected 8-D xyz+wxyz+gripper proprio, got {proprio.size} values")

    xyz = proprio[:3]
    quat_wxyz = proprio[3:7]
    gripper = proprio[7]
    scalar = lambda value: np.asarray(value, dtype=np.float32).reshape(1, 1, 1)

    if policy_setup == "google_robot":
        video_key = "image"
        state_values = dict(
            zip(("x", "y", "z", "rx", "ry", "rz", "rw", "gripper"),
                (*xyz, *quat_wxyz[[1, 2, 3, 0]], 1.0 - gripper))
        )
    elif policy_setup == "widowx_bridge":
        video_key = "image_0"
        rpy = matrix_to_euler_xyz(
            quat_wxyz_to_matrix(quat_wxyz) @ BRIDGE_DEFAULT_ROTATION.T
        )
        state_values = dict(
            zip(("x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"),
                (*xyz, *rpy, 0.0, gripper))
        )
    else:
        raise ValueError(f"Unsupported GR00T policy_setup: {policy_setup}")

    observation: Dict[str, object] = {
        f"video.{video_key}": image[None, None],
        LANGUAGE_KEY: [str(instruction)],
    }
    observation.update({f"state.{key}": scalar(value) for key, value in state_values.items()})
    return observation


def stack_action_dict(action: Dict[str, np.ndarray]) -> np.ndarray:
    """Convert GR00T's seven scalar action streams to a [T, 7] array."""
    columns = []
    for key in ACTION_KEYS:
        value = action.get(f"action.{key}", action.get(key))
        if value is None:
            raise KeyError(f"GR00T response is missing action key: {key}")
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[0] != 1 or arr.shape[2] != 1:
            raise ValueError(f"Unexpected action.{key} shape: {arr.shape}; expected [1, T, 1]")
        columns.append(arr[0, :, 0])
    return np.stack(columns, axis=-1)
