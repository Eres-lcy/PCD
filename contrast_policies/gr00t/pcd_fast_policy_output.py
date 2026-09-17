"""Runtime PCD-Fast patch for the GR00T N1.7 flow-matching action head.

The patch is deliberately external to NVIDIA's source tree.  It reuses the
official intermediate hidden states and final projection weights, selects an
image cross-attention layer using target attention plus a cutoff distance, and
guides only the first action's six spatial dimensions during early denoising.
"""

from __future__ import annotations

import math
import types
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


GUIDE_FRACTION = 0.4
CUTOFF_WEIGHT = 0.2
OBJECT_PATCH_THRESHOLD = 0.5
CUTOFF_BATCH_SIZE = 9


@dataclass
class PCDFastConfig:
    alpha: float = 0.2
    reset_frequency: int = 4
    early_exit_layer: int = -1


def transform_target_mask(mask: np.ndarray, processor: Any) -> np.ndarray:
    """Transform a SimplerEnv mask into the checkpoint's visual frame."""
    source = np.asarray(mask, dtype=np.uint8)
    if source.ndim != 2:
        raise ValueError(
            "PCD-Fast expects an HxW mask before image transforms, "
            f"got {source.shape}"
        )
    if not getattr(processor, "use_albumentations", False):
        raise ValueError(
            "GR00T PCD-Fast mask alignment requires the checkpoint's "
            "Albumentations evaluation transform"
        )
    dummy_image = np.zeros((*source.shape, 3), dtype=np.uint8)
    transformed = processor.eval_image_transform(
        image=dummy_image,
        mask=source,
    )
    if "mask" not in transformed:
        raise RuntimeError(
            "GR00T evaluation transform did not return the PCD-Fast mask"
        )
    return np.asarray(transformed["mask"], dtype=bool)


def _guide_step_count(num_inference_timesteps: int) -> int:
    """Return Pi0's floor-rounded early-denoising step count."""
    return int(num_inference_timesteps * GUIDE_FRACTION)


def _guided_first_spatial_velocity(
    final_velocity: torch.Tensor,
    shallow_velocity: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Apply Pi0's fp32 contrast and norm recovery to action 0's xyz+rpy."""
    baseline = final_velocity[:, 0, :6].float()
    shallow = shallow_velocity[:, 0, :6].float()
    guided = baseline + alpha * (baseline - shallow)
    baseline_norm = baseline.norm(dim=-1, keepdim=True)
    guided_norm = guided.norm(dim=-1, keepdim=True)
    guided = guided * (baseline_norm / (guided_norm + 1e-8))
    return guided.to(final_velocity.dtype)


def _factor_grid(token_count: int, aspect_ratio: float) -> tuple[int, int]:
    """Find the HxW factorization closest to the source image aspect ratio."""
    if token_count <= 0:
        raise ValueError("Image token count must be positive")
    best = (1, token_count)
    best_error = float("inf")
    for height in range(1, int(math.sqrt(token_count)) + 1):
        if token_count % height:
            continue
        width = token_count // height
        for candidate in ((height, width), (width, height)):
            error = abs((candidate[1] / candidate[0]) - aspect_ratio)
            if error < best_error:
                best, best_error = candidate, error
    return best


def _object_token_mask(
    segmentation_mask: np.ndarray,
    image_mask: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Project an HxW segmentation mask onto each sample's visual token slots."""
    source = torch.as_tensor(segmentation_mask, dtype=torch.float32, device=image_mask.device)
    if source.ndim != 2:
        raise ValueError(f"PCD-Fast expects a collapsed HxW mask, got {tuple(source.shape)}")
    result = torch.zeros_like(image_mask, dtype=torch.bool)
    aspect = source.shape[1] / source.shape[0]
    for batch_index in range(image_mask.shape[0]):
        token_count = int(image_mask[batch_index].sum().item())
        if token_count == 0:
            continue
        grid_h, grid_w = _factor_grid(token_count, aspect)
        resized = F.interpolate(
            source[None, None], size=(grid_h, grid_w), mode="area"
        )[0, 0].reshape(-1)
        # Treat a half-covered visual token as belonging to the target too.
        result[batch_index, image_mask[batch_index]] = resized >= threshold
    return result


def _euler_to_quaternion(eulers: torch.Tensor) -> torch.Tensor:
    """Convert xyz Euler angles to normalized wxyz quaternions."""
    roll, pitch, yaw = eulers.unbind(dim=-1)
    half_roll, half_pitch, half_yaw = roll * 0.5, pitch * 0.5, yaw * 0.5
    cr, sr = torch.cos(half_roll), torch.sin(half_roll)
    cp, sp = torch.cos(half_pitch), torch.sin(half_pitch)
    cy, sy = torch.cos(half_yaw), torch.sin(half_yaw)
    quaternion = torch.stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        dim=-1,
    )
    return quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _trajectory_distance(
    candidate_actions: torch.Tensor,
    final_actions: torch.Tensor,
) -> torch.Tensor:
    """Pi0 cutoff: mean translation plus quaternion distance over batch/horizon."""
    if candidate_actions.ndim != 4 or final_actions.ndim != 3:
        raise ValueError("Expected candidate [L,B,H,A] and final [B,H,A] actions")
    if candidate_actions.shape[1:3] != final_actions.shape[:2]:
        raise ValueError("Candidate and final trajectory batch/horizon shapes must match")
    if candidate_actions.shape[-1] < 6 or final_actions.shape[-1] < 6:
        raise ValueError("PCD-Fast trajectory distance requires six spatial action dimensions")

    candidates = candidate_actions.float()
    final = final_actions.float().unsqueeze(0)
    translation = (candidates[..., :3] - final[..., :3]).norm(dim=-1)
    candidate_quaternion = _euler_to_quaternion(candidates[..., 3:6])
    final_quaternion = _euler_to_quaternion(final[..., 3:6])
    # Match Pi0's acos guard exactly.  The lower bound is inert after abs();
    # the 1-1e-7 upper bound avoids an undefined acos gradient/rounding edge.
    quaternion_dot = (
        (candidate_quaternion * final_quaternion)
        .sum(dim=-1)
        .abs()
        .clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    )
    rotation = 2.0 * torch.acos(quaternion_dot)
    return (translation + rotation).mean(dim=(-1, -2))


def _repeat_batch(tensor: torch.Tensor, repeats: int) -> torch.Tensor:
    """Repeat one batch as a view where possible, materializing only when required."""
    return tensor.unsqueeze(0).expand(repeats, *tensor.shape).reshape(
        repeats * tensor.shape[0], *tensor.shape[1:]
    )


def _project_hidden(dit: Any, hidden: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
    temb = dit.timestep_encoder(timestep)
    shift, scale = dit.proj_out_1(F.silu(temb)).chunk(2, dim=1)
    hidden = dit.norm_out(hidden) * (1 + scale[:, None]) + shift[:, None]
    return dit.proj_out_2(hidden)


def _attention_ratio(
    block: Any,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    object_tokens: torch.Tensor,
    temb: torch.Tensor,
    action_horizon: int,
) -> torch.Tensor:
    """Reconstruct GR00T's custom block ``attn1`` cross-attention probabilities."""
    normalized = (
        block.norm1(hidden_states, temb)
        if getattr(block, "norm_type", None) == "ada_norm"
        else block.norm1(hidden_states)
    )
    if getattr(block, "pos_embed", None) is not None:
        normalized = block.pos_embed(normalized)
    attention = block.attn1
    query = attention.to_q(normalized)
    key = attention.to_k(encoder_hidden_states)
    heads = attention.heads
    query = query.view(query.shape[0], query.shape[1], heads, -1).transpose(1, 2)
    key = key.view(key.shape[0], key.shape[1], heads, -1).transpose(1, 2)
    if getattr(attention, "norm_q", None) is not None:
        query = attention.norm_q(query)
    if getattr(attention, "norm_k", None) is not None:
        key = attention.norm_k(key)
    scores = torch.matmul(query, key.transpose(-1, -2)) * attention.scale
    valid = encoder_attention_mask[:, None, None, :].bool()
    scores = scores.float().masked_fill(~valid, torch.finfo(torch.float32).min)
    probs = scores.softmax(dim=-1)
    action_probs = probs[:, :, -action_horizon:, :]
    numerator = (action_probs * object_tokens[:, None, None, :]).sum(dim=-1)
    denominator = action_probs.sum(dim=-1).clamp_min(1e-8)
    return (numerator / denominator).mean()


def _image_cross_layers(dit: Any) -> list[int]:
    every = int(dit.attend_text_every_n_blocks)
    return [
        index
        for index in range(len(dit.transformer_blocks))
        if index % 2 == 0 and index % (2 * every) != 0
    ]


def _rollout_layer_trajectories(
    action_head: Any,
    *,
    initial_actions: torch.Tensor,
    backbone_features: torch.Tensor,
    state_features: torch.Tensor,
    embodiment_id: torch.Tensor,
    backbone_output: Any,
    candidate_layers: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run Pi0-equivalent full trajectories in memory-bounded candidate chunks.

    Candidate layer numbers are 1-based hidden-state indices.  The final DiT
    layer is rolled out as the reference trajectory.  Every trajectory starts
    from the same noise, while chunking changes only execution order.  Since
    every trajectory also has identical input at denoising step zero, that
    first DiT forward is shared before the trajectories diverge.
    """
    model = action_head.model
    final_layer = len(model.transformer_blocks)
    rollout_layers = [*candidate_layers, final_layer]
    batch_size = initial_actions.shape[0]
    dt = 1.0 / action_head.num_inference_timesteps
    first_timesteps = torch.zeros(
        (batch_size,), dtype=torch.long, device=initial_actions.device
    )
    first_action_features = action_head.action_encoder(
        initial_actions, first_timesteps, embodiment_id
    )
    if action_head.config.add_pos_embed:
        pos_ids = torch.arange(
            first_action_features.shape[1],
            dtype=torch.long,
            device=initial_actions.device,
        )
        first_action_features = first_action_features + action_head.position_embedding(
            pos_ids
        ).unsqueeze(0)
    first_sa_embs = torch.cat((state_features, first_action_features), dim=1)
    first_model_output, first_hidden_states = model(
        hidden_states=first_sa_embs,
        encoder_hidden_states=backbone_features,
        timestep=first_timesteps,
        image_mask=backbone_output.image_mask,
        backbone_attention_mask=backbone_output.backbone_attention_mask,
        return_all_hidden_states=True,
    )

    first_velocities = []
    for layer in rollout_layers:
        if layer == final_layer:
            projected = first_model_output
        else:
            projected = _project_hidden(
                model, first_hidden_states[layer], first_timesteps
            )
        first_velocities.append(
            action_head.action_decoder(projected, embodiment_id)[
                :, -action_head.action_horizon :
            ]
        )
    trajectory_actions = initial_actions.unsqueeze(0) + dt * torch.stack(first_velocities)

    if action_head.num_inference_timesteps == 1:
        return trajectory_actions[:-1], trajectory_actions[-1]

    completed: dict[int, torch.Tensor] = {}

    for start in range(0, len(rollout_layers), CUTOFF_BATCH_SIZE):
        chunk_layers = rollout_layers[start : start + CUTOFF_BATCH_SIZE]
        chunk_size = len(chunk_layers)
        actions = trajectory_actions[start : start + chunk_size].reshape(
            chunk_size * batch_size,
            action_head.action_horizon,
            action_head.action_dim,
        )
        chunk_backbone = _repeat_batch(backbone_features, chunk_size)
        chunk_state = _repeat_batch(state_features, chunk_size)
        chunk_embodiment = _repeat_batch(embodiment_id, chunk_size)
        chunk_image_mask = _repeat_batch(backbone_output.image_mask, chunk_size)
        chunk_backbone_mask = _repeat_batch(
            backbone_output.backbone_attention_mask, chunk_size
        )

        for denoise_step in range(1, action_head.num_inference_timesteps):
            t_cont = denoise_step / float(action_head.num_inference_timesteps)
            t_discretized = int(t_cont * action_head.num_timestep_buckets)
            timesteps = torch.full(
                (chunk_size * batch_size,), t_discretized, device=actions.device
            )
            action_features = action_head.action_encoder(
                actions, timesteps, chunk_embodiment
            )
            if action_head.config.add_pos_embed:
                pos_ids = torch.arange(
                    action_features.shape[1], dtype=torch.long, device=actions.device
                )
                action_features = action_features + action_head.position_embedding(
                    pos_ids
                ).unsqueeze(0)
            sa_embs = torch.cat((chunk_state, action_features), dim=1)
            model_output, hidden_states = model(
                hidden_states=sa_embs,
                encoder_hidden_states=chunk_backbone,
                timestep=timesteps,
                image_mask=chunk_image_mask,
                backbone_attention_mask=chunk_backbone_mask,
                return_all_hidden_states=True,
            )

            velocities = []
            for slot, layer in enumerate(chunk_layers):
                batch_slice = slice(slot * batch_size, (slot + 1) * batch_size)
                if layer == final_layer:
                    projected = model_output[batch_slice]
                else:
                    projected = _project_hidden(
                        model,
                        hidden_states[layer][batch_slice],
                        timesteps[batch_slice],
                    )
                velocity = action_head.action_decoder(
                    projected, chunk_embodiment[batch_slice]
                )[:, -action_head.action_horizon :]
                velocities.append(velocity)
            actions = actions + dt * torch.stack(velocities, dim=0).reshape_as(actions)

        chunk_actions = actions.reshape(
            chunk_size, batch_size, action_head.action_horizon, action_head.action_dim
        )
        for slot, layer in enumerate(chunk_layers):
            completed[layer] = chunk_actions[slot]

    candidate_actions = torch.stack([completed[layer] for layer in candidate_layers])
    return candidate_actions, completed[final_layer]


def install_pcd_fast(action_head: Any, config: PCDFastConfig) -> None:
    """Monkey-patch one loaded N1.7 action head; alpha=0 remains upstream-exact."""
    if getattr(action_head, "_pcd_fast_installed", False):
        action_head._pcd_fast_config = config
        return
    if not getattr(action_head.config, "use_alternate_vl_dit", False):
        raise ValueError("PCD-Fast currently requires GR00T N1.7 AlternateVLDiT")

    original = action_head.get_action_with_features
    action_head._pcd_fast_installed = True
    action_head._pcd_fast_config = config
    action_head._pcd_fast_context = None
    action_head._pcd_fast_selected_layer = None
    action_head._pcd_fast_cutoff_metric = None
    action_head._pcd_fast_last_diagnostics = {}

    @torch.no_grad()
    def patched(
        self: Any,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: Any,
        action_input: Any,
        options: dict[str, Any] | None = None,
    ) -> Any:
        cfg: PCDFastConfig = self._pcd_fast_config
        context = self._pcd_fast_context or {}
        # This guarantees paired alpha=0 baseline runs execute NVIDIA's method byte-for-byte.
        if cfg.alpha == 0 or not context.get("enabled", False) or "action" in action_input:
            self._pcd_fast_last_diagnostics = {"enabled": False, "reason": "upstream_baseline"}
            return original(
                backbone_features, state_features, embodiment_id,
                backbone_output, action_input, options,
            )

        from transformers.feature_extraction_utils import BatchFeature

        vl_embeds = backbone_features
        batch_size, device = vl_embeds.shape[0], vl_embeds.device
        dt = 1.0 / self.num_inference_timesteps
        # Match Pi0 exactly: stop_step = int(num_inference_steps * 0.4).
        # This matters for N1.7, which has only four denoising steps (floor=1,
        # whereas the previous ceil implementation incorrectly guided two).
        guide_steps = _guide_step_count(self.num_inference_timesteps)
        candidate_blocks = _image_cross_layers(self.model)
        if not candidate_blocks:
            raise RuntimeError("No image cross-attention layers found in AlternateVLDiT")

        fixed_layer = cfg.early_exit_layer
        selected_hidden_index = self._pcd_fast_selected_layer
        refresh = (
            selected_hidden_index is None
            or int(context.get("timestep", 0)) % max(1, cfg.reset_frequency) == 0
        )
        candidate_layers = [index + 1 for index in candidate_blocks]
        if fixed_layer < 0 and self._pcd_fast_cutoff_metric is None:
            # Pi0's infer_action_batch samples a dedicated cutoff noise first;
            # the subsequent real infer_action samples a new, independent noise.
            cutoff_initial_actions = torch.randn(
                (batch_size, self.config.action_horizon, self.action_dim),
                dtype=vl_embeds.dtype,
                device=device,
            )
            candidate_actions, final_actions = _rollout_layer_trajectories(
                self,
                initial_actions=cutoff_initial_actions,
                backbone_features=vl_embeds,
                state_features=state_features,
                embodiment_id=embodiment_id,
                backbone_output=backbone_output,
                candidate_layers=candidate_layers,
            )
            self._pcd_fast_cutoff_metric = _trajectory_distance(
                candidate_actions, final_actions
            ).detach()
        # This is the real policy noise.  On the first control step it is
        # intentionally sampled after (and independently of) cutoff noise,
        # matching Pi0 PCD-Fast's infer_action_batch -> infer_action call order.
        actions = torch.randn(
            (batch_size, self.config.action_horizon, self.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )

        for denoise_step in range(self.num_inference_timesteps):
            t_cont = denoise_step / float(self.num_inference_timesteps)
            t_discretized = int(t_cont * self.num_timestep_buckets)
            timesteps = torch.full((batch_size,), t_discretized, device=device)
            action_features = self.action_encoder(actions, timesteps, embodiment_id)
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)
            sa_embs = torch.cat((state_features, action_features), dim=1)

            attention_values: dict[int, torch.Tensor] = {}
            handles = []
            if refresh and denoise_step == 0 and fixed_layer < 0:
                mask = context.get("segmentation_mask")
                if mask is None:
                    raise ValueError(
                        "Dynamic GR00T PCD-Fast layer selection requires a segmentation mask"
                    )
                object_tokens = _object_token_mask(
                    mask, backbone_output.image_mask, OBJECT_PATCH_THRESHOLD
                )
                image_attention_mask = (
                    backbone_output.image_mask & backbone_output.backbone_attention_mask
                )
                for block_index in candidate_blocks:
                    block = self.model.transformer_blocks[block_index]

                    def capture(
                        module: Any,
                        args: tuple[Any, ...],
                        kwargs: dict[str, Any],
                        *,
                        index=block_index,
                    ):
                        hidden = kwargs.get("hidden_states", args[0] if args else None)
                        encoder = kwargs.get("encoder_hidden_states")
                        block_temb = kwargs.get("temb")
                        attention_values[index + 1] = _attention_ratio(
                            module,
                            hidden,
                            encoder,
                            image_attention_mask,
                            object_tokens,
                            block_temb,
                            self.action_horizon,
                        )

                    handles.append(block.register_forward_pre_hook(capture, with_kwargs=True))

            try:
                need_hidden_states = denoise_step < guide_steps
                model_result = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                    return_all_hidden_states=need_hidden_states,
                )
                if need_hidden_states:
                    model_output, hidden_states = model_result
                else:
                    model_output, hidden_states = model_result, None
            finally:
                for handle in handles:
                    handle.remove()

            final_velocity = self.action_decoder(model_output, embodiment_id)[
                :, -self.action_horizon :
            ]

            if refresh and denoise_step == 0:
                if fixed_layer >= 0:
                    if fixed_layer not in candidate_layers:
                        raise ValueError(
                            f"early_exit_layer={fixed_layer} is not an image "
                            "cross-attention layer; "
                            f"choose one of {candidate_layers}"
                        )
                    selected_hidden_index = fixed_layer
                else:
                    distances = self._pcd_fast_cutoff_metric.to(device=device, dtype=torch.float32)
                    attentions = torch.stack([
                        attention_values.get(layer, torch.tensor(float("nan"), device=device))
                        for layer in candidate_layers
                    ]).float()
                    if torch.isfinite(attentions).any():
                        score = torch.where(
                            torch.isfinite(attentions),
                            attentions,
                            torch.full_like(attentions, float("inf")),
                        ) + CUTOFF_WEIGHT * distances
                    else:
                        score = distances
                    selected_hidden_index = candidate_layers[int(score.argmin().item())]
                self._pcd_fast_selected_layer = selected_hidden_index
                refresh = False

            if denoise_step < guide_steps:
                projected = _project_hidden(
                    self.model, hidden_states[selected_hidden_index], timesteps
                )
                shallow_velocity = self.action_decoder(projected, embodiment_id)[
                    :, -self.action_horizon :
                ]
                step_alpha = cfg.alpha * (1.0 - denoise_step / guide_steps)
                guided = final_velocity.clone()
                guided[:, 0, :6] = _guided_first_spatial_velocity(
                    final_velocity, shallow_velocity, step_alpha
                )
                velocity = guided
            else:
                velocity = final_velocity
            actions = actions + dt * velocity

        # Keep the response schema without synchronizing diagnostic tensors back
        # to CPU on every policy request.
        self._pcd_fast_last_diagnostics = {"enabled": True}
        return BatchFeature(data={
            "action_pred": actions,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        })

    action_head.get_action_with_features = types.MethodType(patched, action_head)


def set_pcd_fast_context(
    action_head: Any, *, enabled: bool, segmentation_mask: Any, timestep: int
) -> None:
    action_head._pcd_fast_context = {
        "enabled": bool(enabled),
        "segmentation_mask": segmentation_mask,
        "timestep": int(timestep),
    }


def reset_pcd_fast(action_head: Any) -> None:
    action_head._pcd_fast_selected_layer = None
    action_head._pcd_fast_cutoff_metric = None
    action_head._pcd_fast_context = None
    action_head._pcd_fast_last_diagnostics = {}
