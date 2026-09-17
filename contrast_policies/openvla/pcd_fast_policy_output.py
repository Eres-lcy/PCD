from typing import Optional
import json
import os
import numpy as np
from transforms3d.euler import euler2axangle
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
import cv2 as cv
import random
from torch.nn import functional as F
from accelerate import init_empty_weights

_TOKEN_DIM = 32064

def setup_torch_seed(seed):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class OpenVLAPCDFastPolicyOutput:
    def __init__(
        self,
        saved_model_path: str = "openvla-7b",
        unnorm_key: Optional[str] = None,
        policy_setup: str = "widowx_bridge",
        horizon: int = 1,
        pred_action_horizon: int = 1,
        exec_horizon: int = 1,
        image_size: list[int] = [224, 224],
        action_scale: float = 1.0,
        gpu_id: int = 0,
        early_exit_layer: int = 32,
        alpha: float = 0.8,
        reset_frequency: int = 4,
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if policy_setup == "widowx_bridge":
            unnorm_key = "bridge_orig" if unnorm_key is None else unnorm_key
            self.sticky_gripper_num_repeat = 1
        elif policy_setup == "google_robot":
            unnorm_key = "fractal20220817_data" if unnorm_key is None else unnorm_key
            self.sticky_gripper_num_repeat = 15
        else:
            raise NotImplementedError(
                f"Policy setup {policy_setup} not supported by this OpenVLA policy. The other datasets can be found in the huggingface config.json file."
            )
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key
        self.gpu_id = gpu_id
        self.early_exit_layer = early_exit_layer
        self.alpha = alpha
        self.reset_frequency = reset_frequency

        self.processor = AutoProcessor.from_pretrained(saved_model_path, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            saved_model_path,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        
        from .pcd_fast_llama_early_exit import LlamaForCausalLM
        from .pcd_fast_action_generation import OpenVLAForActionPrediction
        with init_empty_weights():
            llama_model = LlamaForCausalLM(self.vla.language_model.config)
            openvla_pcd_fast = OpenVLAForActionPrediction(self.vla.config)
        llama_model.load_state_dict(self.vla.language_model.state_dict(), assign=True)
        openvla_pcd_fast.load_state_dict(self.vla.state_dict(), assign=True)
        # PCD-Fast only projects the 256 action-token rows. Compact the output
        # head while it is still on CPU so the unused full-vocabulary head is
        # never copied to GPU.
        llama_model.compact_action_head()
        self.vla = openvla_pcd_fast
        self.vla.language_model = llama_model
        self.vla.eval()

        self.vla = self.vla.to(f"cuda:0")

        self.image_size = image_size
        self.action_scale = action_scale
        self.horizon = horizon
        self.pred_action_horizon = pred_action_horizon
        self.exec_horizon = exec_horizon

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task = None
        self.task_description = None
        self.num_image_history = 0
        self.bins = torch.linspace(-1, 1, 256)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self._cached_segmentation_mask = None
        self._cached_object_patch_map = None

    def reset(self, task_description: str, seed=None) -> None:
        self.task_description = task_description
        self.num_image_history = 0
        self.vla.reset_frequency = self.reset_frequency

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self._cached_segmentation_mask = None
        self._cached_object_patch_map = None
        
        if seed is not None:
            setup_torch_seed(seed)

    def step(
        self, image: np.ndarray, task_description: Optional[str] = None, timestep: int = 0, segmentation_mask = None, logger = None, episode = None, *args, **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
            task_description: Optional[str], task description; if different from previous task description, policy state is reset
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
                - 'world_vector': np.ndarray of shape (3,), xyz translation of robot end-effector
                - 'rot_axangle': np.ndarray of shape (3,), axis-angle representation of end-effector rotation
                - 'gripper': np.ndarray of shape (1,), gripper action
                - 'terminate_episode': np.ndarray of shape (1,), 1 if episode should be terminated, 0 otherwise
        """
        if timestep == 0:
            logger.info(f"*** episode {episode} start ***")

        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description=task_description)

        if segmentation_mask is not self._cached_segmentation_mask:
            mask_tensor = torch.as_tensor(
                segmentation_mask,
                dtype=torch.float32,
                device="cuda:0",
            )
            resized_mask = F.interpolate(mask_tensor[None, None, :, :], size=(224, 224), mode='nearest')
            patch_means = F.avg_pool2d(resized_mask, kernel_size=14, stride=14)
            self._cached_object_patch_map = (patch_means > 0.5).float().reshape(1, -1)
            self._cached_segmentation_mask = segmentation_mask
        object_patch_map = self._cached_object_patch_map

        assert image.dtype == np.uint8

        image = self._resize_image(image)
        
        image: Image.Image = Image.fromarray(image)
        prompt = task_description
        
        # predict action (7-dof; un-normalize for bridgev2)
        inputs = self.processor(prompt, image).to(f"cuda:0", dtype=torch.bfloat16)
        input_ids, pixel_values = inputs["input_ids"], inputs["pixel_values"]

        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat((input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1)

        if 'proprio' in kwargs:
            del kwargs['proprio']


        if timestep == 0:
            output_pure_list = []
            output_pure = self.vla.generate(
                input_ids=input_ids,                            # Shape: [1, seq]
                pixel_values=pixel_values,                      # Shape: [1, 3, res, res] or Dict[str, ...]
                max_new_tokens=6,
                output_scores=False,
                return_dict_in_generate=True,
                alpha = self.alpha,
                segmentation_mask = None,
                timestep = timestep,
                episode = episode,
                get_cut_off=True,
                **kwargs,
            )
            output_pure_list.append(output_pure)
            self.get_cut_off_index(output_pure_list)


        output = self.vla.generate(
            input_ids=input_ids,                            # Shape: [1, seq]
            pixel_values=pixel_values,                      # Shape: [1, 3, res, res] or Dict[str, ...]
            max_new_tokens=self.get_action_dim(self.unnorm_key),
            output_scores=False,
            return_dict_in_generate=True,
            alpha = self.alpha,
            segmentation_mask = object_patch_map,
            timestep = timestep,
            episode = episode,
            vla_logger = logger,
            **kwargs,
        )

        predicted_action_token_ids = output.sequences[
            0, -self.vla.get_action_dim(self.unnorm_key):
        ]
        raw_actions = self._decode_actions(predicted_action_token_ids, self.unnorm_key)[None]

        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # range [0, 1]; 1 = open; 0 = close
        }

        # process raw_action to obtain the action to be sent to the maniskill2 environment
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action["rot_axangle"] = action_rotation_axangle * self.action_scale

        if self.policy_setup == "google_robot":
            current_gripper_action = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
            else:
                relative_gripper_action = self.previous_gripper_action - current_gripper_action
            self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_on):
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action["gripper"] = relative_gripper_action

        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0

        action["terminate_episode"] = np.array([0.0])

        return raw_action, action
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    def _decode_actions(self, action_token_ids, unnorm_key):
        discretized_actions = self.vla.vocab_size - action_token_ids.cpu()
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.vla.bin_centers.shape[0] - 1)
        normalized_actions = self.vla.bin_centers[discretized_actions]

        # Unnormalize actions
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        return np.where(mask, 0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low, normalized_actions)
    
    def get_action_stats(self, unnorm_key):
        if unnorm_key == 'finetune_dataset':
            with open('pretrained/dataset_statistics.json') as f:
                action_norm_stats = json.load(f)['finetune_dataset']['action']
        else:
            action_norm_stats = self.vla.get_action_stats(unnorm_key)
        return action_norm_stats

    def get_action_dim(self, unnorm_key):
        norm_stats = self.get_action_stats(unnorm_key)
        return len(norm_stats['q01'])

    def get_cut_off_index(self, output_pure_list):
        score_pure_list = [output_pure['all_logits'].squeeze().cpu() for output_pure in output_pure_list]
        score_pure = torch.cat(score_pure_list, dim=1)
        score_pure = score_pure[:,:,:]

        raw_action_pure = torch.argmax(score_pure, dim=-1)

        discretized_pure = (256 - raw_action_pure).sub_(1).clamp_(min=0, max=self.bin_centers.size(0) - 1)
        normalized_pure = self.bin_centers[discretized_pure]

        hidden_action = normalized_pure[:, :-1]
        output_action = normalized_pure[:, -1:]

        action_diff = hidden_action[:3, :] - output_action[:3, :]
        l2_dist = torch.norm(action_diff, dim=0) 
        rot_dist = self.calculate_similarity_tensor(output_action, hidden_action)
        
        metric = l2_dist + rot_dist
        metric = metric.squeeze()
        metric = np.array(metric)   
          
        self.vla.cutoff_metric = torch.as_tensor(
            metric,
            device="cuda:0",
        )

    def euler_to_quaternion_torch(self, eulers):
        r, p, y = eulers[0, ...], eulers[1, ...], eulers[2, ...]
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

    def calculate_similarity_tensor(self, one_action, batch_actions):
        euler_one = one_action[3:, ...]    # (3)
        euler_batch = batch_actions[3:, ...] # (31, 3)

        q_one = self.euler_to_quaternion_torch(euler_one)       # (4)
        q_batch = self.euler_to_quaternion_torch(euler_batch)   # (31, 4)

        dot_prod = torch.sum(q_one * q_batch, dim=-1)
        dot_prod = torch.clamp(torch.abs(dot_prod), -1.0 + 1e-7, 1.0 - 1e-7)
        dist_rot = 2 * torch.acos(dot_prod)

        return dist_rot
