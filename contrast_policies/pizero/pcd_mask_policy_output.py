import numpy as np
import torch

from simpler_env.policies.pizero.pizero_model import PiZeroInference
from .pcd_mask_kde import PCDMaskKDE


def split_repeat_concat(tensor, num_repeats):
    """Repeat original and contrast conditions without interleaving them."""
    assert tensor.size(0) == 2
    original, contrast = torch.split(tensor, 1, dim=0)
    repeat_shape = (num_repeats,) + (1,) * (tensor.ndim - 1)
    return torch.cat(
        [
            original.repeat(*repeat_shape),
            contrast.repeat(*repeat_shape),
        ],
        dim=0,
    )


class PiZeroPCDMaskPolicyOutput(PiZeroInference):
    """Original Pi0 PCD logic adapted to the current Pi0 model API."""

    def __init__(
        self,
        alpha=0.2,
        num_repeats=24,
        bandwidth_factor=1.0,
        keep_threshold=0.5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_repeats = num_repeats
        self.kde = PCDMaskKDE(
            alpha,
            bandwidth_factor,
            keep_threshold,
        )

        # Match the original policy: sample unclipped trajectories, apply KDE
        # contrast decoding, then clip only the selected trajectory.
        self.clip_value = self.model.final_action_clip_value
        self.model.final_action_clip_value = None

    @torch.no_grad()
    def step(self, image, contrast_image, instruction, proprio):
        inputs = self.preprocess_inputs(image, instruction, proprio)
        contrast_inputs = self.preprocess_inputs(
            contrast_image,
            instruction,
            proprio,
        )

        all_inputs = {
            key: torch.cat([inputs[key], contrast_inputs[key]], dim=0)
            for key in inputs
        }
        all_actions = self.forward_actions(all_inputs)
        actions, contrast_actions = torch.chunk(all_actions, 2, dim=0)

        raw_actions = self.kde(actions, contrast_actions)
        if self.clip_value is not None:
            raw_actions = torch.clamp(
                raw_actions,
                -self.clip_value,
                self.clip_value,
            )

        actions = self.env_adapter.postprocess(
            raw_actions[0].float().cpu().numpy()
        )
        formatted_actions = [
            {
                "world_vector": action[:3],
                "rot_axangle": action[3:6],
                "gripper": action[6:7],
                "terminate_episode": np.array([0.0]),
            }
            for action in actions
        ]
        return raw_actions, formatted_actions

    def forward_actions(self, inputs):
        if self.use_naive:
            # The original infer_actions_naive implementation is intentionally
            # unsupported as well.
            raise NotImplementedError(
                "Original Pi0 PCD supports only use_naive=False."
            )

        with torch.inference_mode():
            return self._infer_actions_current(**inputs)

    def _infer_actions_current(
        self,
        input_ids,
        pixel_values,
        image_text_proprio_mask,
        action_mask,
        vlm_position_ids,
        proprio_position_ids,
        action_position_ids,
        proprios,
    ):
        """Current-API equivalent of the original Pi0 infer_actions method."""
        if pixel_values.size(0) != 2:
            raise ValueError(
                "Original Pi0 PCD expects an original/contrast batch of 2, "
                f"got {pixel_values.size(0)}."
            )

        dtype = pixel_values.dtype
        device = pixel_values.device
        model = self.model
        kv_caches = model.joint_model.build_mixture_caches()

        # Encode the original and contrast conditions once before expanding
        # each branch to N stochastic action trajectories.
        inputs_embeds = model._forward_siglip_and_text_embedding(
            input_ids,
            pixel_values,
        )
        proprio_embeds = model.proprio_encoder(proprios)

        inputs_embeds = split_repeat_concat(
            inputs_embeds,
            self.num_repeats,
        )
        proprio_embeds = split_repeat_concat(
            proprio_embeds,
            self.num_repeats,
        )
        image_text_proprio_mask = split_repeat_concat(
            image_text_proprio_mask,
            self.num_repeats,
        )
        action_mask = split_repeat_concat(
            action_mask,
            self.num_repeats,
        )
        vlm_position_ids = split_repeat_concat(
            vlm_position_ids,
            self.num_repeats,
        )
        proprio_position_ids = split_repeat_concat(
            proprio_position_ids,
            self.num_repeats,
        )
        action_position_ids = split_repeat_concat(
            action_position_ids,
            self.num_repeats,
        )

        _, kv_caches = model.joint_model(
            attention_mask=image_text_proprio_mask,
            position_ids_all={
                "vlm": vlm_position_ids,
                "proprio": proprio_position_ids,
            },
            embeds_all={
                "vlm": inputs_embeds,
                "proprio": proprio_embeds,
            },
            # The original JointModel skipped only VLM after the final
            # attention layer. The current default also skips proprio.
            final_layer_post_attn_skip_names=("vlm",),
            kv_caches=kv_caches,
            return_caches=True,
        )

        total_samples = 2 * self.num_repeats
        action = torch.randn(
            (total_samples, model.horizon_steps, model.action_dim),
            device=device,
            dtype=dtype,
        )
        timestep = torch.zeros(
            total_samples,
            device=device,
            dtype=dtype,
        )

        delta_t = 1.0 / model.num_inference_steps
        for _ in range(model.num_inference_steps):
            time_cond = model.time_embedding(timestep)
            if model.action_expert_adaptive_mode:
                action_embeds = model.action_encoder(action)
            else:
                action_embeds = model.action_encoder(action, time_cond)

            action_outputs = model.joint_model(
                attention_mask=action_mask,
                position_ids_all={"action": action_position_ids},
                embeds_all={"action": action_embeds},
                time_cond=time_cond,
                final_layer_post_attn_skip_names=("vlm",),
                kv_caches=kv_caches,
                cache_mode="append_non_active",
            )
            action_hidden = action_outputs["action"]
            action_velocity = model.action_decoder(action_hidden)
            action += delta_t * action_velocity
            timestep += delta_t

        return action
