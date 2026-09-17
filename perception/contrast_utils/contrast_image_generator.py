"""PCD-Mask contrast-image generation."""

import cv2
import numpy as np
import os
import torch

from .inpainters import build_inpainter
from .instruction_templates import get_objects_from_instruction
from .mask_predictors import build_predictor, predict_masks_with_predictor
from .properties import _ROBOT_NAMES


def mask_to_points(mask):
    if not mask.any():
        return None
    points = np.argwhere(mask)
    # sample 5 points
    if len(points) > 5:
        points = points[np.random.choice(len(points), 5, replace=False)]
    # y,x -> x,y
    return points[:, [1, 0]]


def mask_to_bbox(mask):
    if not mask.any():
        return None
    y, x = np.where(mask)
    return np.array([x.min(), y.min(), x.max(), y.max()])
    

def name_to_alias(name):
    s = name.split('_')
    rm_list = ['opened', 'light', 'generated', 'modified', 'objaverse', 'bridge', 'baked', 'v2']
    cleaned = []
    for w in s:
        if w[-2:] == "cm":
            # object size in object name
            continue
        if w not in rm_list:
            cleaned.append(w)
    return ' '.join(cleaned)


class ContrastImageGenerator:
    def __init__(self, 
                 env, 
                 camera_name=None, 
                 by="gt",
                 inpaint_mode="lama",
                 version=2,
                 get_all_parts=False):
        self.env = env
        self.camera_name = camera_name
        self.by = by
        assert version in [2, 3]
        self.version = version
        self.get_all_parts = get_all_parts
        
        self.mask_objects = None
        self.keep_objects = None
        self.task_description = None
        self.predictor = None
        self.inpainter = build_inpainter(inpaint_mode)
    
    def generate(self, obs, task_description, task, return_image=True):
        """Advance the mask predictor and optionally materialize an inpainted image."""
        if task_description != self.task_description:
            self.reset_mask_and_keep_object_names(task_description)
            self.task_description = task_description
            if self.by != "gt":
                if self.predictor is None:
                    self.predictor = build_predictor(self.by)
                elif hasattr(self.predictor, "reset"):
                    self.predictor.reset()
                if self.by in ["point_tracking", "box_tracking"]:
                    self._set_points_or_boxes(obs)
        
        if self.by == "gt":
            if not return_image:
                return None
            mask, excluded_mask = self.get_mask_by_gt(obs)
        else:
            mask, excluded_mask = self.get_mask_by_predictor(
                obs,
                task,
                return_masks=return_image,
            )
            if not return_image:
                return None
        
        image = self.inpainter.inpaint(self._get_rgb_image(obs), mask, excluded_mask)
        return image
    
    def reset(self):
        self.task_description = None
        
    def reset_mask_and_keep_object_names(self, task_description):
        self.mask_objects = get_objects_from_instruction(task_description, self.get_all_parts)
        self.keep_objects = _ROBOT_NAMES if self.by == "gt" else ["robot"]
    
    def get_mask_by_gt(self, obs):
        seg = self._get_segmentation(obs)
        name2id = self._get_name_to_id()
        
        masks = [self._get_object_mask_by_gt(seg, name2id, obj_name) for obj_name in self.mask_objects]
        keep_masks = [self._get_object_mask_by_gt(seg, name2id, obj_name) for obj_name in self.keep_objects]
        mask = self._add_reserve_keep_mask(seg.shape, masks, keep_masks)

        robot_mask = np.zeros_like(seg, dtype=bool)
        for robot_name in _ROBOT_NAMES:
            robot_mask |= self._get_object_mask_by_gt(seg, name2id, robot_name)

        return mask, robot_mask
    
    def get_mask_by_predictor(self, obs, task, return_masks=True):
        image = self._get_rgb_image(obs)
        objs = self.mask_objects + self.keep_objects
        masks = predict_masks_with_predictor(
            image,
            objs,
            self.predictor,
            task,
            return_masks=return_masks,
        )
        if not return_masks:
            return None, None
        mask_obj_masks, keep_obj_masks = masks[:len(self.mask_objects)], masks[len(self.mask_objects):len(self.mask_objects) + len(self.keep_objects)]
        robot_mask = masks[objs.index('robot')] if 'robot' in objs else None
        mask = self._add_reserve_keep_mask(
            image.shape[:2], mask_obj_masks, keep_obj_masks
        )
        return mask, robot_mask
    
    def _set_points_or_boxes(self, obs):
        assert self.by in ["point_tracking", "box_tracking"]
        assert self.predictor is not None, "Predictor is not initialized"
        seg = self._get_segmentation(obs)
        name2id = self._get_name_to_id()

        masks = []
        for obj_name in self.mask_objects + self.keep_objects:
            if obj_name == 'robot':
                robot_mask = np.zeros_like(seg, dtype=bool)
                for robot_name in _ROBOT_NAMES:
                    robot_mask |= self._get_object_mask_by_gt(seg, name2id, robot_name)
                masks.append(robot_mask)
            else:
                masks.append(self._get_object_mask_by_gt(seg, name2id, obj_name))

        if self.by == "point_tracking":
            points = [mask_to_points(mask) for mask in masks]
            self.predictor.predictor.set_points(points)
        elif self.by == "box_tracking":
            boxes = [mask_to_bbox(mask) for mask in masks]
            self.predictor.predictor.set_boxes(boxes)
    
    def _get_name_to_id(self):
        if self.version == 2:
            actor_name2id = {name_to_alias(actor.name): actor.id for actor in self.env.unwrapped.get_actors()}
            robot_name2id = {link.name: link.id for link in self.env.unwrapped.agent.robot.get_links()}
            art_name2id = {}
            for art_obj in self.env.unwrapped.get_articulations():
                if art_obj.name in ['cabinet']:
                    for link in art_obj.get_links():
                        art_name2id[name_to_alias(link.name)] = link.id
            return {**actor_name2id, **robot_name2id, **art_name2id}
        else:
            name2id = {}
            for k, v in self.env.unwrapped.segmentation_id_map.items():
                name2id[v.name] = k
            return name2id

    def _get_object_mask_by_gt(self, seg, name2id, obj_name):
        # 1. check if object name is in assets
        if os.path.exists('assets'):
            filenames = os.listdir('assets')
            if obj_name + '.png' in filenames:
                return cv2.imread(f'assets/{obj_name}.png', cv2.IMREAD_GRAYSCALE) > 0
        
        # 2. check if object name is in name2id
        if obj_name not in name2id:
            return np.zeros_like(seg, dtype=bool)
        return seg == name2id[obj_name]
    
    def _add_reserve_keep_mask(self, shape, masks, keep_masks):
        mask = np.zeros(shape, dtype=bool)
        for obj_mask in masks:
            if obj_mask is not None:
                mask |= obj_mask

        for obj_mask in keep_masks:
            if obj_mask is not None:
                mask[obj_mask] = False

        return mask
    
    def _get_rgb_image(self, obs):
        image = self._get_camera_images(obs)['rgb']
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if len(image.shape) == 4 and image.shape[0] == 1:
            image = image[0]    
        return image

    def _get_segmentation(self, obs):
        seg_key = "Segmentation" if self.version == 2 else "segmentation"
        if self.version == 2:
            seg = self._get_camera_images(obs)[seg_key][..., 1].copy()
        else:
            seg = self._get_camera_images(obs)[seg_key][0, :, : ,0]
        if isinstance(seg, torch.Tensor):
            seg = seg.cpu().numpy()
        return seg

    def _get_camera_images(self, obs):
        robot = self.env.unwrapped.robot_uid if self.version == 2 else self.env.unwrapped.robot_uids
        if not isinstance(robot, list):
            robot = ''.join(robot)
        
        camera_name = self.camera_name
        if camera_name is None:
            if "google_robot" in robot:
                camera_name = "overhead_camera"
            elif "widowx" in robot:
                camera_name = "3rd_view_camera"
            elif "panda" in robot:
                camera_name = "base_camera"
            elif "panda_wristcam" in robot:
                camera_name = "base_camera"
            else:
                raise NotImplementedError()
        
        key = "image" if self.version == 2 else "sensor_data"
        return obs[key][camera_name]
