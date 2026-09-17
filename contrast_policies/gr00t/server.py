"""GR00T N1.7 inference server for baseline, PCD-Mask, and PCD-Fast."""

from __future__ import annotations

import argparse
import logging
import traceback
import numpy as np
import torch
import zmq
from gr00t.policy.gr00t_policy import (
    Gr00tPolicy,
    Gr00tSimPolicyWrapper,
)

from contrast_policies.gr00t.observation import build_flat_observation
from contrast_policies.gr00t.pcd_mask_kde import select_policy_output
from contrast_policies.gr00t.pcd_mask_policy_output import sample_policy_outputs
from contrast_policies.gr00t.pcd_fast_policy_output import (
    CUTOFF_BATCH_SIZE,
    CUTOFF_WEIGHT,
    GUIDE_FRACTION,
    OBJECT_PATCH_THRESHOLD,
    PCDFastConfig,
    install_pcd_fast,
    reset_pcd_fast,
    set_pcd_fast_context,
    transform_target_mask,
)


LOGGER = logging.getLogger("contrast_policies.gr00t.server")


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _action_dict_to_array(action: dict[str, np.ndarray]) -> np.ndarray:
    """Convert decoded scalar action streams to [B,H,7]."""
    columns = []
    for name in ("x", "y", "z", "roll", "pitch", "yaw", "gripper"):
        value = action.get(f"action.{name}", action.get(name))
        if value is None:
            raise KeyError(f"GR00T response is missing action key: {name}")
        array = np.asarray(value, dtype=np.float32)
        if array.ndim != 3 or array.shape[-1] != 1:
            raise ValueError(
                f"Unexpected action.{name} shape {array.shape}; expected [B,H,1]"
            )
        columns.append(array[..., 0])
    return np.stack(columns, axis=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--embodiment-tag", required=True)
    parser.add_argument("--policy-setup", choices=("google_robot", "widowx_bridge"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    base_policy = Gr00tPolicy(
        embodiment_tag=args.embodiment_tag,
        model_path=args.model_path,
        device=args.device,
        strict=True,
    )
    install_pcd_fast(base_policy.model.action_head, PCDFastConfig())
    policy = Gr00tSimPolicyWrapper(base_policy, strict=True)

    context = zmq.Context.instance()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://{args.host}:{args.port}")
    LOGGER.info("GR00T-PCD-Fast ready on tcp://%s:%d", args.host, args.port)
    LOGGER.info(
        "model=%s embodiment=%s guide_fraction=%s cutoff_weight=%s "
        "object_patch_threshold=%s cutoff_batch_size=%s",
        args.model_path,
        args.embodiment_tag,
        GUIDE_FRACTION,
        CUTOFF_WEIGHT,
        OBJECT_PATCH_THRESHOLD,
        CUTOFF_BATCH_SIZE,
    )

    while True:
        request = socket.recv_pyobj()
        try:
            command = request.get("command", "step")
            if command == "ping":
                response = {"ok": True, "model_path": args.model_path}
            elif command == "reset":
                seed = int(request.get("seed", 0))
                _set_seed(seed)
                reset_pcd_fast(base_policy.model.action_head)
                response = {"ok": True}
            elif command == "step":
                timestep = int(request.get("timestep", 0))
                observation = build_flat_observation(
                    request["image"], request["instruction"], request["proprio"], args.policy_setup
                )
                for key, value in request.get("pcd_fast_config", {}).items():
                    if hasattr(base_policy.model.action_head._pcd_fast_config, key):
                        setattr(base_policy.model.action_head._pcd_fast_config, key, value)
                method = request.get("method", "pcd_fast" if request.get("pcd_fast", True) else "baseline")
                if method == "pcd_mask":
                    pcd_mask_config = request.get("pcd_mask_config", {})
                    repeats = int(pcd_mask_config.get("num_repeats", 24))
                    default_exec_horizon = 1 if args.policy_setup == "google_robot" else 4
                    exec_horizon = int(
                        pcd_mask_config.get("exec_horizon", default_exec_horizon)
                    )
                    decode_horizon = len(
                        base_policy.modality_configs["action"].delta_indices
                    )
                    if not 1 <= exec_horizon <= decode_horizon:
                        raise ValueError(
                            f"PCD-Mask exec_horizon must be in [1, {decode_horizon}], "
                            f"got {exec_horizon}"
                        )
                    embodiment_key = base_policy.embodiment_tag.value
                    action_dim = int(
                        base_policy.processor.state_action_processor.get_action_dim(
                            embodiment_key
                        )
                    )
                    contrast_image = request.get("contrast_image")
                    if contrast_image is None:
                        raise ValueError("PCD-Mask request is missing contrast_image")
                    contrast_observation = build_flat_observation(
                        contrast_image,
                        request["instruction"],
                        request["proprio"],
                        args.policy_setup,
                    )
                    samples, contrast_samples, original_states = (
                        sample_policy_outputs(
                            base_policy,
                            observation,
                            contrast_observation,
                            repeats=repeats,
                            output_horizon=exec_horizon,
                            output_action_dim=action_dim,
                        )
                    )
                    decoded_normalized = select_policy_output(
                        samples,
                        contrast_samples,
                        alpha=float(pcd_mask_config.get("alpha", 0.2)),
                        bandwidth_factor=float(
                            pcd_mask_config.get("bandwidth_factor", 1.0)
                        ),
                        keep_threshold=float(
                            pcd_mask_config.get("keep_threshold", 0.5)
                        ),
                    )
                    decoded_action = base_policy.processor.decode_action(
                        decoded_normalized[None],
                        base_policy.embodiment_tag,
                        original_states,
                    )
                    decoded_action = {
                        key: np.asarray(value, dtype=np.float32)
                        for key, value in decoded_action.items()
                    }
                    decoded = _action_dict_to_array(decoded_action)[0]
                    response = {
                        "ok": True,
                        "action_array": decoded,
                        "info": {},
                        "pcd_mask": {
                            "enabled": True,
                            "num_repeats": repeats,
                            "batch_size": 2 * repeats,
                            "exec_horizon": exec_horizon,
                            "action_dim": action_dim,
                            "sampling": "pi0_batched",
                            "kde_space": "normalized_action",
                        },
                        "pcd_fast": {"enabled": False, "reason": "pcd_mask"},
                    }
                elif method in {"baseline", "pcd_fast"}:
                    segmentation_mask = request.get("segmentation_mask")
                    if method == "pcd_fast":
                        if segmentation_mask is None:
                            raise ValueError("GR00T PCD-Fast request is missing segmentation_mask")
                        segmentation_mask = transform_target_mask(
                            segmentation_mask, base_policy.processor
                        )
                    set_pcd_fast_context(
                        base_policy.model.action_head,
                        enabled=method == "pcd_fast",
                        segmentation_mask=segmentation_mask,
                        timestep=timestep,
                    )
                    action, info = policy.get_action(observation)
                    response = {
                        "ok": True,
                        "action": action,
                        "info": info,
                        "pcd_fast": base_policy.model.action_head._pcd_fast_last_diagnostics,
                    }
                else:
                    raise ValueError(f"Unknown inference method: {method}")
            elif command == "shutdown":
                response = {"ok": True}
                socket.send_pyobj(response)
                break
            else:
                raise ValueError(f"Unknown command: {command}")
        except Exception as error:
            LOGGER.exception("Request failed")
            response = {"ok": False, "error": str(error), "traceback": traceback.format_exc()}
        socket.send_pyobj(response)


if __name__ == "__main__":
    main()
