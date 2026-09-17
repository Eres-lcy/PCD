import numpy as np
import torch

from .pizero_model import PiZeroInference


class PiZeroBaselineInference(PiZeroInference):
    """PiZero inference without attention export or visualization."""

    def step(self, image, instruction, proprio, *args, **kwargs):
        inputs = self.preprocess_inputs(image, instruction, proprio)
        with torch.inference_mode():
            raw_actions = self.model.infer_action(**inputs)

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
