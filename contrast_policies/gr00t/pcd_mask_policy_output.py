"""Batched policy-output generation for GR00T PCD-Mask."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature

from gr00t.data.types import MessageType
from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype

from .pcd_fast_policy_output import set_pcd_fast_context


def _merge_flat_observations(
    observation: dict[str, object],
    contrast_observation: dict[str, object],
) -> dict[str, object]:
    """Build the Pi0-style two-condition batch: original then contrast."""
    if observation.keys() != contrast_observation.keys():
        raise ValueError(
            "Original and contrast GR00T observations have different keys"
        )
    merged: dict[str, object] = {}
    for key in observation:
        first, second = observation[key], contrast_observation[key]
        if isinstance(first, np.ndarray) and isinstance(second, np.ndarray):
            merged[key] = np.concatenate((first, second), axis=0)
        elif isinstance(first, (list, tuple)) and isinstance(
            second, (list, tuple)
        ):
            merged[key] = [*first, *second]
        else:
            raise TypeError(
                f"Cannot merge GR00T observation key {key!r}: "
                f"{type(first).__name__} and {type(second).__name__}"
            )
    return merged


def _flat_to_nested(
    policy: Gr00tPolicy,
    observation: dict[str, object],
) -> dict[str, dict[str, Any]]:
    """Apply the observation-only part of Gr00tSimPolicyWrapper."""
    nested: dict[str, dict[str, Any]] = {
        "video": {},
        "state": {},
        "language": {},
    }
    for modality in nested:
        for key in policy.modality_configs[modality].modality_keys:
            if modality == "language":
                parsed_key = (
                    "annotation.human.coarse_action"
                    if key == "task"
                    and "annotation.human.coarse_action" in observation
                    else key
                )
                values = observation[parsed_key]
                if isinstance(values, np.ndarray):
                    values = values.reshape(-1).tolist()
                elif isinstance(values, str):
                    values = [values]
                nested[modality][key] = [[str(item)] for item in values]
            else:
                nested[modality][key] = observation[f"{modality}.{key}"]
    return nested


def _split_repeat_concat(tensor: torch.Tensor, repeats: int) -> torch.Tensor:
    """Expand conditions in the order original*R, contrast*R."""
    if tensor.ndim == 0 or tensor.shape[0] != 2:
        raise ValueError(
            "Pi0-style GR00T PCD-Mask expects every condition tensor to have "
            f"batch size 2; got {tuple(tensor.shape)}"
        )
    repeat_shape = (repeats, *([1] * (tensor.ndim - 1)))
    return torch.cat(
        (
            tensor[0:1].repeat(*repeat_shape),
            tensor[1:2].repeat(*repeat_shape),
        ),
        dim=0,
    )


def _repeat_batch_feature(feature: BatchFeature, repeats: int) -> BatchFeature:
    """Repeat batched tensor fields while retaining non-batched metadata."""
    data: dict[str, Any] = {}
    for key, value in feature.items():
        if (
            isinstance(value, torch.Tensor)
            and value.ndim > 0
            and value.shape[0] == 2
        ):
            data[key] = _split_repeat_concat(value, repeats)
        else:
            data[key] = value
    return BatchFeature(data=data)


def _prepare_model_inputs(
    model: Any,
    inputs: dict[str, torch.Tensor],
) -> tuple[BatchFeature, BatchFeature]:
    """Move only tensors consumed by the backbone and action head."""
    backbone_keys = (
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
    )
    action_keys = ("state", "embodiment_id")
    missing = [key for key in (*backbone_keys, *action_keys) if key not in inputs]
    if missing:
        raise KeyError(f"GR00T PCD-Mask input is missing required keys: {missing}")

    def to_model_device(value: torch.Tensor) -> torch.Tensor:
        if torch.is_floating_point(value):
            return value.to(model.device, dtype=model.dtype)
        return value.to(model.device)

    backbone_inputs = BatchFeature(
        data={key: to_model_device(inputs[key]) for key in backbone_keys}
    )
    action_inputs = BatchFeature(
        data={key: to_model_device(inputs[key]) for key in action_keys}
    )
    return backbone_inputs, action_inputs


def sample_policy_outputs(
    policy: Gr00tPolicy,
    observation: dict[str, object],
    contrast_observation: dict[str, object],
    *,
    repeats: int,
    output_horizon: int,
    output_action_dim: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Generate original and contrast action samples in one batched call."""
    if repeats < 2:
        raise ValueError("PCD-Mask num_repeats must be at least 2")
    if output_horizon < 1 or output_action_dim < 1:
        raise ValueError(
            "PCD-Mask output horizon and action dimension must be positive; "
            f"got horizon={output_horizon}, action_dim={output_action_dim}"
        )

    merged = _merge_flat_observations(observation, contrast_observation)
    nested = _flat_to_nested(policy, merged)
    unbatched = policy._unbatch_observation(nested)
    processed_inputs = []
    states: list[dict[str, np.ndarray]] = []
    for item in unbatched:
        step_data = policy._to_vla_step_data(item)
        states.append(step_data.states)
        messages = [
            {"type": MessageType.EPISODE_STEP.value, "content": step_data}
        ]
        processed_inputs.append(policy.processor(messages))
    collated = _rec_to_dtype(
        policy.collate_fn(processed_inputs), dtype=torch.bfloat16
    )

    model = policy.model
    action_head = model.action_head
    set_pcd_fast_context(
        action_head, enabled=False, segmentation_mask=None, timestep=0
    )
    with torch.inference_mode():
        backbone_inputs, action_inputs = _prepare_model_inputs(
            model, collated["inputs"]
        )
        backbone_output = model.backbone(backbone_inputs)
        features = action_head._encode_features(backbone_output, action_inputs)

        repeated_backbone = _repeat_batch_feature(backbone_output, repeats)
        repeated_backbone_features = repeated_backbone.backbone_features
        repeated_state_features = _split_repeat_concat(
            features.state_features, repeats
        )
        repeated_embodiment = _split_repeat_concat(
            action_inputs.embodiment_id, repeats
        )
        prediction = action_head.get_action_with_features(
            backbone_features=repeated_backbone_features,
            state_features=repeated_state_features,
            embodiment_id=repeated_embodiment,
            backbone_output=repeated_backbone,
            action_input=action_inputs,
            options=None,
        )
        action_pred = prediction["action_pred"]
        if (
            output_horizon > action_pred.shape[1]
            or output_action_dim > action_pred.shape[2]
        ):
            raise ValueError(
                "Requested PCD-Mask output slice exceeds the model prediction "
                f"shape: horizon={output_horizon}, action_dim={output_action_dim}, "
                f"prediction={tuple(action_pred.shape)}"
            )
        normalized = (
            action_pred[:, :output_horizon, :output_action_dim]
            .float()
            .cpu()
            .numpy()
        )

    if normalized.shape[0] != 2 * repeats:
        raise RuntimeError(
            f"GR00T PCD-Mask produced {normalized.shape[0]} samples; "
            f"expected {2 * repeats}"
        )
    original_states = {
        key: np.stack([states[0][key]], axis=0)
        for key in policy.modality_configs["state"].modality_keys
    }
    return normalized[:repeats], normalized[repeats:], original_states
