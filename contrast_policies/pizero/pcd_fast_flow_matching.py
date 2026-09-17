"""
Wrapper around the joint model (mixtures). Siglip from PaliGemma, action-time encoder, proprio encoder, action decoder. Flow matching training

Generates causal masking for the mixtures

Potentially customized to add/remove mixtures, e.g., remove proprio or add another vision module

"""
import cv2
import logging
from typing import Optional, Tuple

import numpy as np
import hydra
import torch
from torch import nn
from src.model.kv_cache import KVCache
from src.model.vla.modules import (
    ActionEncoder,
    SinusoidalPosEmb,
)
from src.utils.decorator import NoSyncBase
from src.utils.monitor import log_execution_time

log = logging.getLogger(__name__)


class PiZeroPCDFast(nn.Module, NoSyncBase):
    @log_execution_time(log)
    def __init__(self, cfg, use_ddp: bool = False, early_exit_layer: Optional[int] = 18, alpha=0.2, reset_frequency=4):
        super().__init__()
        self.cfg = cfg
        self.use_ddp = use_ddp  # used in NoSyncBase
        self.vocab_size = cfg.vocab_size
        self.pad_token_id = cfg.pad_token_id
        self.image_token_index = cfg.image_token_index
        self.use_lm_head = cfg.get("use_lm_head", False)
        self.early_exit_layer = early_exit_layer
        self.alpha = alpha
        self.reset_frequency = reset_frequency
        self.contrastive_layer_index = early_exit_layer
        self.cutoff_metric = None

        self.max_image_text_tokens = cfg.max_image_text_tokens
        self.num_proprio_tokens = cfg.cond_steps
        self.num_action_tokens = cfg.horizon_steps
        self.total_num_tokens = (
            self.max_image_text_tokens
            + self.num_proprio_tokens
            + self.num_action_tokens
        )

        self.image_text_hidden_size = cfg.mixture.vlm.hidden_size
        self.proprio_hidden_size = cfg.mixture.proprio.hidden_size
        self.action_hidden_size = cfg.mixture.action.hidden_size

        # Action parameterization
        self.num_inference_steps = cfg.num_inference_steps
        self.horizon_steps = cfg.horizon_steps
        self.action_dim = cfg.action_dim
        self.proprio_dim = cfg.proprio_dim
        self.final_action_clip_value = cfg.final_action_clip_value
        self.flow_sig_min = cfg.get("flow_sig_min", 0.001)

        # text input only
        self.embed_tokens = nn.Embedding(
            cfg.vocab_size,
            self.image_text_hidden_size,
            self.pad_token_id,
        )  # 0.527B parameters

        # Vision
        self.vision_tower = hydra.utils.instantiate(cfg.vision)
        self.multi_modal_projector = hydra.utils.instantiate(cfg.vision_projector)

        # Mixtures
        
        cfg.joint['_target_'] = (
            'contrast_policies.pizero.pcd_fast_joint_layer_outputs.PCDFastJointModel'
        )
        cfg.joint.config.early_exit_layer = early_exit_layer
        self.joint_model = hydra.utils.instantiate(cfg.joint)

        # Action, proprio, time encoders
        self.action_expert_adaptive_mode = cfg.action_expert_adaptive_mode
        if cfg.action_expert_adaptive_mode:  # adaLN or adaLN-Zero
            self.action_encoder = ActionEncoder(
                self.action_dim,
                self.action_hidden_size,
                time_cond=False,
            )
            self.time_embedding = SinusoidalPosEmb(
                cfg.time_hidden_size, cfg.time_max_period
            )
        else:  # matching pi0
            self.action_encoder = ActionEncoder(
                self.action_dim,
                self.action_hidden_size,
                time_cond=True,
            )
            self.time_embedding = SinusoidalPosEmb(
                self.action_hidden_size, cfg.time_max_period
            )
        self.proprio_encoder = nn.Linear(
            self.proprio_dim,
            self.proprio_hidden_size,
        )

        # Action decoder
        self.action_decoder = nn.Linear(
            self.action_hidden_size,
            self.action_dim,
        )

        # optional text output
        if self.use_lm_head:
            self.lm_head = nn.Linear(
                self.image_text_hidden_size,
                self.vocab_size,
                bias=False,
            )
            self.lm_head.weight = self.embed_tokens.weight  # tie weights

    @property
    def action_expert_parameters(self):
        return (
            list(self.action_encoder.parameters())
            + list(self.action_decoder.parameters())
            + list(self.proprio_encoder.parameters())
            + list(self.joint_model.mixtures["action"].parameters())
        )  # note: action and proprio share weights

    @property
    def trainable_vlm_parameters(self):
        return (
            list(self.vision_tower.parameters())
            + list(self.multi_modal_projector.parameters())
            + self.trainable_gemma_parameters
        )

    @property
    def lora_trainable_vlm_parameters(self):
        params = []
        for name, param in self.vision_tower.named_parameters():
            if "lora_" in name:
                params.append(param)
        for name, param in self.multi_modal_projector.named_parameters():
            if "lora_" in name:
                params.append(param)
        params.extend(self.trainable_lora_gemma_parameters)
        return params

    @property
    def trainable_gemma_parameters(self):
        gemma_parameters = []
        for name, param in self.joint_model.mixtures["vlm"].named_parameters():
            if not self._check_gemma_unused_parameter_by_name(name):
                gemma_parameters.append(param)
        return gemma_parameters

    @property
    def trainable_lora_gemma_parameters(self):
        gemma_parameters = []
        for name, param in self.joint_model.mixtures["vlm"].named_parameters():
            if not self._check_gemma_unused_parameter_by_name(name):
                if "lora_" in name:
                    gemma_parameters.append(param)
        return gemma_parameters

    @log_execution_time(log)
    def load_pretrained_weights(self):
        """vision, projector, lm from paligemma"""
        import glob
        import os

        from safetensors import safe_open

        # load tensors from files
        safetensors_files = glob.glob(
            os.path.join(self.cfg.pretrained_model_path, "*.safetensors")
        )
        tensors = {}
        for safetensors_file in safetensors_files:
            with safe_open(safetensors_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)

        # load embed tokens
        embed_tokens_state_dict = self.embed_tokens.state_dict()
        for k, v in tensors.items():
            if "embed_tokens" in k:
                new_key = k.replace("language_model.model.embed_tokens.", "")
                embed_tokens_state_dict[new_key] = v
        self.embed_tokens.load_state_dict(embed_tokens_state_dict, strict=True)
        log.info("Loaded pre-trained weights for embed tokens")

        # load vision tower --- "vision_tower.vision_model" -> "vision_model"
        vision_tower_state_dict = self.vision_tower.state_dict()
        for k, v in tensors.items():
            if "vision_tower" in k:
                new_key = k.replace("vision_tower.", "")
                vision_tower_state_dict[new_key] = v
        self.vision_tower.load_state_dict(vision_tower_state_dict, strict=True)
        log.info("Loaded pre-trained weights for vision tower")

        # load projector --- "multi_modal_projector.linear" -> "linear"
        multi_modal_projector_state_dict = self.multi_modal_projector.state_dict()
        for k, v in tensors.items():
            if "multi_modal_projector" in k:
                new_key = k.replace("multi_modal_projector.", "")
                multi_modal_projector_state_dict[new_key] = v
        self.multi_modal_projector.load_state_dict(
            multi_modal_projector_state_dict, strict=True
        )
        log.info("Loaded pre-trained weights for projector")

        # load lm --- do not change any lora weights
        joint_model_state_dict = self.joint_model.state_dict()
        lora_keys = []
        for key in (
            joint_model_state_dict.keys()
        ):  # avoid RuntimeError: OrderedDict mutated during iteration
            if "lora_" in key:
                lora_keys.append(key)
        for key in lora_keys:
            del joint_model_state_dict[key]
        for k, v in tensors.items():
            if "language_model.model" in k:
                new_key = k.replace("language_model.model.", "mixtures.vlm.")
                joint_model_state_dict[new_key] = v
        self.joint_model.load_state_dict(joint_model_state_dict, strict=False)
        log.info("Loaded pre-trained weights for lm part of the joint model")

    def _check_gemma_unused_parameter_by_name(self, name: str) -> bool:
        """no need to train vlm parameters after attention of last layer"""
        last_hidden_layer_index = self.joint_model.num_hidden_layers - 1
        if (
            f"{last_hidden_layer_index}.post" in name
            or f"{last_hidden_layer_index}.mlp" in name
            or f"{last_hidden_layer_index}.self_attn.o_proj" in name
            or f"{last_hidden_layer_index}.self_attn.v_proj" in name
        ):  # final norm is not initialized
            return True
        return False

    def freeze_non_lora_weights_in_vlm(self):
        """Keep all bias frozen"""
        for name, param in self.vision_tower.named_parameters():
            param.requires_grad = True if "lora_" in name else False
        log.info("Froze non-lora weights in vision tower")

        for name, param in self.multi_modal_projector.named_parameters():
            param.requires_grad = True if "lora_" in name else False
        log.info("Froze non-lora weights in projector")

        for name, param in self.joint_model.mixtures["vlm"].named_parameters():
            if not self._check_gemma_unused_parameter_by_name(name):
                param.requires_grad = True if "lora_" in name else False
        log.info("Froze non-lora weights in lm part of the joint model")

    def freeze_unused_weights(self):
        """text embedding and part of last layer of vlm, including lora"""
        self.embed_tokens.weight.requires_grad = False
        for name, param in self.joint_model.mixtures["vlm"].named_parameters():
            if self._check_gemma_unused_parameter_by_name(name):
                param.requires_grad = False

    def freeze_all_weights(self):
        for _, param in self.named_parameters():
            param.requires_grad = False

    def tie_action_proprio_weights(self):
        """technically more than just tying weights"""
        self.joint_model.mixtures["proprio"] = self.joint_model.mixtures["action"]

    def build_text_cache(self):
        return KVCache()

    # ---------- Input preparation ----------#

    def build_causal_mask_and_position_ids(
        self, attention_mask: torch.Tensor, dtype: torch.dtype
    ) -> Tuple[torch.FloatTensor]:
        """
        block attention --- padding for unused text tokens

                 img/text img/text img/text (padding) proprio action action
        img/text    x        x        x
        img/text    x        x        x
        img/text    x        x        x
        (padding)
        proprio     x        x        x                 x
        action      x        x        x                 x       x      x
        action      x        x        x                 x       x      x
        """
        bsz = attention_mask.size(0)
        proprio_start = self.max_image_text_tokens
        proprio_end = self.max_image_text_tokens + self.num_proprio_tokens
        action_start = proprio_end
        image_text_token_cnts = torch.sum(attention_mask, dim=1)
        causal_mask = torch.full(
            (bsz, self.total_num_tokens, self.total_num_tokens),
            torch.finfo(dtype).min,
            dtype=dtype,
        )  # smallest value, avoid using inf for softmax nan issues with padding
        for idx, cnt in enumerate(image_text_token_cnts):
            causal_mask[idx, :cnt, :cnt] = 0  # image/text attend to itself
            causal_mask[idx, proprio_start:, :cnt] = (
                0  # proprio/action attend to image/text
            )
        causal_mask[:, proprio_start:proprio_end, proprio_start:proprio_end] = (
            0  # proprio attend to itself
        )
        causal_mask[:, action_start:, proprio_start:] = (
            0  # action attend to itself and proprio
        )

        # add the head dimension
        # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, Num_Heads_Q, Q_Len, KV_Len]
        causal_mask = causal_mask.unsqueeze(1)

        # position ids for each blocks --- start at 1
        vlm_position_ids = torch.arange(1, self.max_image_text_tokens + 1).repeat(
            bsz, 1
        )
        proprio_position_ids = torch.arange(1, self.num_proprio_tokens + 1).repeat(
            bsz, 1
        )
        action_position_ids = torch.arange(
            self.num_proprio_tokens + 1,
            self.num_proprio_tokens + self.num_action_tokens + 1,
        ).repeat(bsz, 1)
        # since proprio and action share the same mixture weights, makes sense to use [1 (proprio), 2 (action), 3 (action), ...] instead of [1 (proprio), 1 (action), 2 (action), ...]
        return causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids

    def split_full_mask_into_submasks(
        self, causal_mask: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """split into ones for paligemma and action"""
        image_text_proprio_mask = causal_mask[
            ...,
            : self.max_image_text_tokens + self.num_proprio_tokens,
            : self.max_image_text_tokens + self.num_proprio_tokens,
        ]
        action_mask = causal_mask[..., -self.num_action_tokens :, :]
        return image_text_proprio_mask, action_mask

    def build_causal_mask_and_position_ids_for_text(
        self,
        q_len: int,
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        dtype, device = attention_mask.dtype, attention_mask.device

        if kv_cache is None or kv_cache.num_items() == 0:
            # do not mask any token, because we're in the prefill phase
            # assume no padding
            causal_mask = torch.full((bsz, q_len, q_len), 0, dtype=dtype, device=device)
        else:
            assert q_len == 1, "Using KV cache so should only use one single token"
            kv_len = kv_cache.num_items() + q_len
            # also in this case we don't need to mask anything, since each query should be able to attend all previous tokens.
            # this only works when we have no padding
            causal_mask = torch.full(
                (bsz, q_len, kv_len), 0, dtype=dtype, device=device
            )

        # add the head dimension
        # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, Num_Heads_Q, Q_Len, KV_Len]
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            # use the last location
            position_ids = attention_mask.cumsum(-1)[:, -1:]
        else:
            # create position_ids based on the size of the attention_mask
            # for padded tokens, use number 1
            position_ids = (attention_mask.cumsum(-1)).masked_fill_(
                (attention_mask == 0), 1
            )
        return causal_mask, position_ids

    # ---------- Inference ----------#

    def _forward_siglip_and_text_embedding(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
    ) -> torch.FloatTensor:
        dtype, device = pixel_values.dtype, pixel_values.device

        # text embedding
        # [Batch_Size, Seq_Len, Hidden_Size]
        inputs_embeds = self.embed_tokens(input_ids)

        # image features from siglip and projector
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Hidden_Size]
        selected_image_feature = self.vision_tower(pixel_values)
        image_features = self.multi_modal_projector(selected_image_feature)

        # normalize the image features
        _, _, embed_dim = image_features.shape
        bsz, seq_len = input_ids.shape
        scaled_image_features = image_features / (self.image_text_hidden_size**0.5)

        # put embedding together - image, text, padding
        final_embedding = torch.full(
            (bsz, seq_len, embed_dim), self.pad_token_id, dtype=dtype, device=device
        )

        # [Batch_Size, Seq_Len]
        text_mask = (input_ids != self.image_token_index) & (
            input_ids != self.pad_token_id
        )
        image_mask = input_ids == self.image_token_index
        final_embedding[text_mask] = inputs_embeds[text_mask]
        for i in range(bsz):
            image_indices = image_mask[i].nonzero(as_tuple=True)[0]
            num_image_tokens = len(image_indices)
            final_embedding[i, image_indices] = scaled_image_features[
                i, :num_image_tokens
            ]
        return final_embedding

    def infer_action(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_text_proprio_mask: torch.FloatTensor,
        action_mask: torch.FloatTensor,
        vlm_position_ids: torch.LongTensor,
        proprio_position_ids: torch.LongTensor,
        action_position_ids: torch.LongTensor,
        proprios: torch.FloatTensor,
        timestep: int,
        segmentation_mask: np.ndarray,
    ) -> torch.FloatTensor:
        dtype, device = pixel_values.dtype, pixel_values.device
        bsz = pixel_values.size(0)

        kv_caches = self.joint_model.build_mixture_caches()

        # merge the text tokens and the image tokens
        inputs_embeds = self._forward_siglip_and_text_embedding(input_ids, pixel_values)

        # proprio
        proprio_embeds = self.proprio_encoder(proprios)

        # forward pass through the vlm and proprio, cache the kv
        kv_caches = self.joint_model(
            attention_mask=image_text_proprio_mask,
            position_ids_all={
                "vlm": vlm_position_ids,
                "proprio": proprio_position_ids,
            },
            embeds_all={
                "vlm": inputs_embeds,
                "proprio": proprio_embeds,
            },
            kv_caches=kv_caches,
            return_caches=True,
        )['kv_caches']

        # sample pure action noise
        action = torch.randn(
            (bsz, self.horizon_steps, self.action_dim), device=device, dtype=dtype
        )

        # forward euler integration --- using kv caches of vlm and proprio
        delta_t = 1.0 / self.num_inference_steps
        t = torch.zeros(bsz, device=device, dtype=dtype)
        attn_weights = []
        should_select_layer = timestep % self.reset_frequency == 0
        stop_step = int(self.num_inference_steps * 0.4)
        for denoising_step in range(self.num_inference_steps):
            # encode action and time into embedding
            time_cond = self.time_embedding(t)
            # [Batch_Size, Horizon_Steps, Embed_Dim]
            if self.action_expert_adaptive_mode:
                action_embeds = self.action_encoder(action)
            else:
                action_embeds = self.action_encoder(action, time_cond)
            # [Batch_Size, Horizon_Steps, Embed_Dim]
            should_return_attention = should_select_layer and denoising_step == 0
            outputs = self.joint_model(
                attention_mask=action_mask,
                position_ids_all={"action": action_position_ids},
                embeds_all={"action": action_embeds},
                time_cond=time_cond,
                kv_caches=kv_caches,
                cache_mode="append_non_active",  # use caches from other mixtures, i.e., vlm and proprio
                return_attn_weights=should_return_attention,
                return_all_hidden_states=True,
            )
            action_embeds = outputs['embeds']["action"]
            if should_return_attention:
                attn_weights.append(outputs["attn_weights"].detach().cpu())
            # decode action: [Batch_Size, Horizon_Steps, Action_Dim]
            action_all_hhidden_states = (outputs.get('all_hidden_states', None)).get('action', None)
            L, B, S, D = action_all_hhidden_states.shape
            flat_embeds = action_all_hhidden_states.view(L * B, S, D)
            flat_v_all_layers = self.action_decoder(flat_embeds)
            v_all_layers = flat_v_all_layers.view(L, B, S, -1).float()
            action_vel = v_all_layers[-1] # last layer's output

            if denoising_step == 0 and should_select_layer:
                self.contrastive_layer_index = self.get_contrastive_layer(
                    segmentation_mask,
                    attn_weights,
                )
            if denoising_step < stop_step:

                selected_action_vel = v_all_layers[self.contrastive_layer_index]

                action_vel_spatial_step0 = action_vel[..., 0:1, :-1]
                action_vel_gripper_step0 = action_vel[..., 0:1, -1:]

                selected_action_vel_spatial_step0 = selected_action_vel[..., 0:1, :-1]

                alpha = self.alpha * (1 - denoising_step / stop_step) # linearly decay
                vel_guided_spatial_step0 = action_vel_spatial_step0 + alpha * (action_vel_spatial_step0 - selected_action_vel_spatial_step0)
                norm_vel_spatial_step0 = torch.linalg.norm(action_vel_spatial_step0, dim=-1, keepdim=True)
                norm_vel_guided_spatial_step0 = torch.linalg.norm(vel_guided_spatial_step0, dim=-1, keepdim=True)

                normalized_vel_guided_spatial_step0 = vel_guided_spatial_step0 * (norm_vel_spatial_step0 / (norm_vel_guided_spatial_step0 + 1e-8))
                normalized_vel_guided_step0 = torch.cat([normalized_vel_guided_spatial_step0, action_vel_gripper_step0], dim=-1)
                normalized_vel_guided = torch.cat([normalized_vel_guided_step0, action_vel[..., 1:, :]], dim=1)

                action += delta_t * normalized_vel_guided
                action = action.bfloat16()
            else :
                action += delta_t * action_vel
            t += delta_t

        # clamp final output if specified
        if self.final_action_clip_value is not None:
            action = torch.clamp(
                action,
                -self.final_action_clip_value,
                self.final_action_clip_value,
            )
        return action, attn_weights

    
    def get_contrastive_layer(
        self,
        segmentation_mask: list[np.ndarray],
        attention_map,
    ) -> int:
        # get attention on object tokens
        attn_weights = torch.stack(attention_map, dim=0)
        attn_weights = attn_weights.mean(dim=3) # average over heads
        attn_weights = attn_weights.mean(dim=3) # average over horizons
        attn_weights = attn_weights[:, :, :, 0:256].squeeze()
        attn_weights = attn_weights.float().numpy() # [layer_len, image_token_len]

        segmentation_mask = np.any(segmentation_mask, axis=0).astype(np.uint8)
        resized_mask = cv2.resize(segmentation_mask, (224, 224), interpolation=cv2.INTER_NEAREST)
        patch_means = resized_mask.reshape(16, 14, 16, 14).mean(axis=(1, 3))
        object_patch_map = (patch_means > 0.5).astype(np.float32).reshape(1, -1)

        attention_on_image = attn_weights.sum(axis=-1)
        attention_on_object_patches = (attn_weights * object_patch_map).sum(axis=-1)
        attention_on_object = attention_on_object_patches / attention_on_image

        metric = attention_on_object[:-1] + self.cutoff_metric.float().numpy() * 0.2
        selected_layer = np.argmin(metric)

        return selected_layer
    
    def _expand_shared_kv_caches(self, kv_caches, L, B):
        import copy
        expanded_caches = {}
        
        for modal_name, cache_obj in kv_caches.items():
            new_cache = copy.copy(cache_obj)
            
            for attr_name in dir(new_cache):
                if attr_name.startswith('__'): 
                    continue
                
                attr_value = getattr(new_cache, attr_name)
                
                if isinstance(attr_value, (list, tuple)) and len(attr_value) > 0 and isinstance(attr_value[0], torch.Tensor):
                    expanded_list = []
                    for tensor in attr_value:
                        expanded_tensor = tensor.expand(L * B, *tensor.shape[1:])
                        expanded_list.append(expanded_tensor)
                    
                    setattr(new_cache, attr_name, type(attr_value)(expanded_list))
                    
                elif isinstance(attr_value, torch.Tensor):
                    expanded_tensor = attr_value.expand(L * B, *attr_value.shape[1:])
                    setattr(new_cache, attr_name, expanded_tensor)
                    
            expanded_caches[modal_name] = new_cache
            
        return expanded_caches

    def infer_action_batch(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_text_proprio_mask: torch.FloatTensor,
        action_mask: torch.FloatTensor,
        vlm_position_ids: torch.LongTensor,
        proprio_position_ids: torch.LongTensor,
        action_position_ids: torch.LongTensor,
        proprios: torch.FloatTensor,
    ) -> torch.FloatTensor:
        dtype, device = pixel_values.dtype, pixel_values.device
        bsz = pixel_values.size(0)

        kv_caches = self.joint_model.build_mixture_caches()
        inputs_embeds = self._forward_siglip_and_text_embedding(input_ids, pixel_values)
        proprio_embeds = self.proprio_encoder(proprios)

        kv_caches = self.joint_model(
            attention_mask=image_text_proprio_mask,
            position_ids_all={"vlm": vlm_position_ids, "proprio": proprio_position_ids},
            embeds_all={"vlm": inputs_embeds, "proprio": proprio_embeds},
            kv_caches=kv_caches,
            return_caches=True,
        )['kv_caches']

        action_noise = torch.randn((bsz, self.horizon_steps, self.action_dim), device=device, dtype=dtype)
        delta_t = 1.0 / self.num_inference_steps

        t_0 = torch.zeros(bsz, device=device, dtype=dtype)
        time_cond_0 = self.time_embedding(t_0)
        action_embeds_0 = self.action_encoder(action_noise, time_cond_0) if not self.action_expert_adaptive_mode else self.action_encoder(action_noise)
        
        outputs_0 = self.joint_model(
            attention_mask=action_mask,
            position_ids_all={"action": action_position_ids},
            embeds_all={"action": action_embeds_0},
            time_cond=time_cond_0,
            kv_caches=kv_caches,
            cache_mode="append_non_active", 
            return_all_hidden_states=True,
        )
        
        action_all_hidden_states = outputs_0['all_hidden_states']['action']
        L, B, S, D_emb = action_all_hidden_states.shape
        
        flat_embeds = action_all_hidden_states.view(L * B, S, D_emb)
        v_all_layers_0 = self.action_decoder(flat_embeds).view(L, B, S, -1)

        action_batch = action_noise.unsqueeze(0).repeat(L, 1, 1, 1)
        
        action_batch += delta_t * v_all_layers_0
        
        expanded_kv_caches = self._expand_shared_kv_caches(kv_caches, L, bsz)
        expanded_action_mask = action_mask.expand(L * bsz, *action_mask.shape[1:])
        expanded_action_pos_ids = action_position_ids.expand(L * bsz, *action_position_ids.shape[1:])

        for denoising_step in range(1, self.num_inference_steps):
            t_current = torch.full((L * bsz,), denoising_step * delta_t, device=device, dtype=dtype)
            time_cond = self.time_embedding(t_current)

            flat_action_batch = action_batch.view(L * bsz, S, -1)
            action_embeds = self.action_encoder(flat_action_batch, time_cond) if not self.action_expert_adaptive_mode else self.action_encoder(flat_action_batch)
            
            outputs = self.joint_model(
                attention_mask=expanded_action_mask,
                position_ids_all={"action": expanded_action_pos_ids},
                embeds_all={"action": action_embeds},
                time_cond=time_cond,
                kv_caches=expanded_kv_caches,
                cache_mode="append_non_active",
                return_all_hidden_states=True,
            )
            
            action_all_hidden_states = outputs['all_hidden_states']['action']
            L_out, LB, S, D_emb = action_all_hidden_states.shape
            flat_embeds = action_all_hidden_states.view(L_out * LB, S, D_emb)
            flat_v_all_layers = self.action_decoder(flat_embeds)
            v_all_layers_batch = flat_v_all_layers.view(L_out, L, B, S, -1)
            target_vels = v_all_layers_batch[torch.arange(L), torch.arange(L)]
            action_batch += delta_t * target_vels

        self.get_action_similarity(action_batch)

    
    def get_action_similarity(self, action_batch):

        def euler_to_quaternion_torch(eulers):
            r, p, y = eulers[..., 0], eulers[..., 1], eulers[..., 2]

            rx, ry, rz = r * 0.5, p * 0.5, y * 0.5

            cx, sx = torch.cos(rx), torch.sin(rx)
            cy, sy = torch.cos(ry), torch.sin(ry)
            cz, sz = torch.cos(rz), torch.sin(rz)

            w = cx*cy*cz + sx*sy*sz
            x = sx*cy*cz - cx*sy*sz
            y = cx*sy*cz + sx*cy*sz
            z = cx*cy*sz - sx*sy*cz

            q = torch.stack([w, x, y, z], dim=-1)
            q = q / torch.norm(q, dim=-1, keepdim=True)
            return q
        action_batch = torch.clamp(action_batch, -self.final_action_clip_value, self.final_action_clip_value).detach().cpu()
        last_layer_vel = action_batch[-1:, :, :, :]
        middle_layer_vel = action_batch[:-1, :, :, :]  #[L, B, chunk_size, Action_Dim]

        L2_dist = torch.linalg.norm(middle_layer_vel[..., 0:3] - last_layer_vel[..., 0:3], dim=-1)
        middle_layer_quaternion = euler_to_quaternion_torch(middle_layer_vel[..., 3:6])
        last_layer_quaternion = euler_to_quaternion_torch(last_layer_vel[..., 3:6])
        dot_prod = torch.sum(middle_layer_quaternion * last_layer_quaternion, dim=-1)
        dot_prod = torch.clamp(torch.abs(dot_prod), -1.0 + 1e-7, 1.0 - 1e-7)
        rot_dist = 2 * torch.acos(dot_prod)
        raw_dist = (L2_dist + rot_dist).mean(-1).squeeze()
        self.cutoff_metric = raw_dist
        return raw_dist



    def infer_action_naive(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        causal_mask: torch.FloatTensor,
        vlm_position_ids: torch.LongTensor,
        proprio_position_ids: torch.LongTensor,
        action_position_ids: torch.LongTensor,
        proprios: torch.FloatTensor,
    ) -> torch.FloatTensor:
        dtype, device = pixel_values.dtype, pixel_values.device
        bsz = pixel_values.size(0)

        kv_caches = self.joint_model.build_mixture_caches()

        # merge the text tokens and the image tokens
        inputs_embeds = self._forward_siglip_and_text_embedding(input_ids, pixel_values)

        # encode proprio
        proprio_embeds = self.proprio_encoder(proprios)

        # sample pure action noise
        action = torch.randn(
            (bsz, self.horizon_steps, self.action_dim), device=device, dtype=dtype
        )

        # forward euler integration --- run vlm in each step, which is unnecessary
        delta_t = 1.0 / self.num_inference_steps
        t = torch.zeros(bsz, device=device, dtype=dtype)
        for _ in range(self.num_inference_steps):
            # encode action and time into embedding
            time_cond = self.time_embedding(t)
            # [Batch_Size, Horizon_Steps, Embed_Dim]
            if self.action_expert_adaptive_mode:
                action_embeds = self.action_encoder(action)
            else:
                action_embeds = self.action_encoder(action, time_cond)
            action_embeds = self.joint_model(
                attention_mask=causal_mask,
                position_ids_all={
                    "vlm": vlm_position_ids,
                    "proprio": proprio_position_ids,
                    "action": action_position_ids,
                },
                embeds_all={
                    "vlm": inputs_embeds.clone(),  # clone needed due to modified in-place
                    "proprio": proprio_embeds.clone(),
                    "action": action_embeds,
                },
                time_cond=time_cond,
                kv_caches=kv_caches,
                cache_mode="no_append",  # no new tokens
            )['embeds']["action"]
            # decode action: [Batch_Size, Horizon_Steps, Action_Dim]
            action_vel = self.action_decoder(action_embeds)
            action += delta_t * action_vel
            t += delta_t

        # clamp final output if specified
        if self.final_action_clip_value is not None:
            action = torch.clamp(
                action,
                -self.final_action_clip_value,
                self.final_action_clip_value,
            )
        return action

    def infer_text(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        q_len = input_ids.size(1)

        # text tokens + image tokens
        inputs_embeds = self._forward_siglip_and_text_embedding(input_ids, pixel_values)

        # build causal mask and position ids for text
        (
            causal_mask,
            position_ids,
        ) = self.build_causal_mask_and_position_ids_for_text(
            q_len, attention_mask, kv_cache
        )

        hidden_states = self.joint_model(
            attention_mask=causal_mask,
            position_ids_all={"vlm": position_ids},
            embeds_all={"vlm": inputs_embeds},
            kv_caches={"vlm": kv_cache},
            cache_mode="append",  # new tokens for the active mixture
            final_layer_post_attn_skip_names=[],  # do not skip vlm last layer
        )['embeds']["vlm"]
        logits = self.lm_head(hidden_states)
        output = {
            "logits": logits,
        }
        if kv_cache is not None:
            output["kv_cache"] = kv_cache
        return output

    # ---------- Flow matching training ----------#

    def psi_t(
        self,
        x: torch.FloatTensor,
        x1: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Conditional Flow"""
        t = t[:, None, None]  # (B, 1, 1)
        return (1 - (1 - self.flow_sig_min) * t) * x + t * x1

    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.ByteTensor,
        causal_mask: torch.FloatTensor,
        vlm_position_ids: torch.LongTensor,
        proprio_position_ids: torch.LongTensor,
        action_position_ids: torch.LongTensor,
        proprios: torch.FloatTensor,
        actions: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """flow matching loss for action prediction, no use of kv cache"""
        # noisy action
        # [Batch_Size, Horizon_Steps, Action_Dim]
        x0 = torch.randn_like(actions, device=t.device, dtype=t.dtype)
        x1 = actions
        psi_t = self.psi_t(x0, x1, t)

        # text tokens + image tokens
        inputs_embeds = self._forward_siglip_and_text_embedding(input_ids, pixel_values)

        # proprio
        proprio_embeds = self.proprio_encoder(proprios)

        # inference with noisy action
        # [Batch_Size, Embed_Dim]
        time_cond = self.time_embedding(t)
        # [Batch_Size, Horizon_Steps, Embed_Dim]
        if self.action_expert_adaptive_mode:
            action_embeds = self.action_encoder(psi_t)
        else:
            action_embeds = self.action_encoder(psi_t, time_cond)
        action_embeds = self.joint_model(
            attention_mask=causal_mask,
            position_ids_all={
                "vlm": vlm_position_ids,
                "proprio": proprio_position_ids,
                "action": action_position_ids,
            },
            embeds_all={
                "vlm": inputs_embeds,
                "proprio": proprio_embeds,
                "action": action_embeds,
            },
            time_cond=time_cond,
            kv_caches={},  # no caching during training
        )["embeds"]["action"]

        # [Batch_Size, Horizon_Steps, Action_Dim]
        v_psi = self.action_decoder(action_embeds)

        # compare to true velocity
        d_psi = x1 - (1 - self.flow_sig_min) * x0
        return torch.mean((v_psi - d_psi) ** 2)
