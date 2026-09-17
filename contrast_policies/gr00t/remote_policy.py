"""Thin SimplerEnv client for the isolated GR00T N1.7 inference service."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import zmq


PCD_ROOT = Path(__file__).resolve().parents[2]
if str(PCD_ROOT) not in sys.path:
    sys.path.insert(0, str(PCD_ROOT))

from .observation import (
    GripperActionAdapter,
    collapse_segmentation_mask,
    stack_action_dict,
)


class Gr00tRemotePolicy:
    def __init__(
        self,
        server_address="tcp://127.0.0.1:5555",
        timeout_ms=120000,
        policy_setup="google_robot",
        method="baseline",
        exec_horizon=1,
        alpha=0.2,
        num_repeats=24,
        bandwidth_factor=1.0,
        keep_threshold=0.5,
        reset_frequency=4,
        early_exit_layer=-1,
        **_unused,
    ):
        self.server_address = server_address
        self.policy_setup = policy_setup
        if method not in {"baseline", "pcd_mask", "pcd_fast"}:
            raise ValueError(f"Unsupported GR00T method: {method}")
        self.method = method
        self.exec_horizon = int(exec_horizon)
        self.gripper_adapter = GripperActionAdapter(policy_setup)
        self.pcd_fast_config = {
            "alpha": float(alpha),
            "reset_frequency": int(reset_frequency),
            "early_exit_layer": int(early_exit_layer),
        }
        self.pcd_mask_config = {
            "alpha": float(alpha),
            "num_repeats": int(num_repeats),
            "bandwidth_factor": float(bandwidth_factor),
            "keep_threshold": float(keep_threshold),
            "exec_horizon": self.exec_horizon,
        }
        self.seed = 0
        context = zmq.Context.instance()
        self.socket = context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
        self.socket.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(server_address)
        reply = self._request({"command": "ping"})
        if not reply.get("ok"):
            raise RuntimeError(f"GR00T server ping failed: {reply}")

    def _request(self, payload):
        try:
            self.socket.send_pyobj(payload)
            response = self.socket.recv_pyobj()
        except zmq.Again as error:
            raise TimeoutError(
                f"GR00T service at {self.server_address} did not respond before the timeout"
            ) from error
        if not response.get("ok", False):
            detail = response.get("traceback", response.get("error", response))
            raise RuntimeError(f"GR00T service failed:\n{detail}")
        return response

    def reset(self, instruction, seed=None):
        del instruction
        self.seed = int(seed or 0)
        self.gripper_adapter.reset()
        self._request({"command": "reset", "seed": self.seed})

    def step(
        self,
        image,
        instruction,
        timestep,
        segmentation_mask,
        proprio,
        contrast_image=None,
        logger=None,
        **_unused,
    ):
        image = np.ascontiguousarray(image, dtype=np.uint8)
        source_shape = image.shape[:2]
        mask = (
            collapse_segmentation_mask(segmentation_mask, source_shape)
            if self.method == "pcd_fast"
            else None
        )
        if self.method == "pcd_mask":
            if contrast_image is None:
                raise ValueError("GR00T PCD-Mask requires a contrast_image")
            contrast_image = np.ascontiguousarray(contrast_image, dtype=np.uint8)
            if contrast_image.shape != image.shape:
                raise ValueError(
                    f"Contrast/original image shape mismatch: "
                    f"{contrast_image.shape} != {image.shape}"
                )
        target_width, target_height = (
            (320, 256) if self.policy_setup == "google_robot" else (256, 256)
        )
        if image.shape[:2] != (target_height, target_width):
            image = cv2.resize(image, (target_width, target_height))
            if contrast_image is not None:
                contrast_image = cv2.resize(
                    contrast_image, (target_width, target_height)
                )
            if mask is not None:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (target_width, target_height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
        response = self._request({
            "command": "step",
            "image": image,
            "instruction": instruction,
            "proprio": np.asarray(proprio, dtype=np.float32),
            "segmentation_mask": mask,
            "contrast_image": contrast_image,
            "timestep": int(timestep),
            "seed": self.seed,
            "method": self.method,
            "pcd_fast": self.method == "pcd_fast",
            "pcd_fast_config": self.pcd_fast_config,
            "pcd_mask_config": self.pcd_mask_config,
        })
        if "action_array" in response:
            raw_actions = np.asarray(response["action_array"], dtype=np.float32)
        else:
            raw_actions = stack_action_dict(response["action"])
        horizon = min(self.exec_horizon, raw_actions.shape[0])
        formatted = []
        for action in raw_actions[:horizon]:
            env_gripper = self.gripper_adapter(action[6])
            formatted.append({
                "world_vector": action[:3],
                "rot_axangle": action[3:6],
                "gripper": env_gripper,
                "terminate_episode": np.asarray([0.0], dtype=np.float32),
            })
        return raw_actions, formatted
