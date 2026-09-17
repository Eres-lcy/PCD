import cv2
import json
import numpy as np
import os
import torch
from PIL import Image

from .utils import *


_CLASSNAMES = ['robot', 'coke can', 'pepsi can', 'redbull can', '7up can', 'blue plastic bottle', 
               'apple', 'orange', 'sponge', 'bottom drawer', 'middle drawer', 'top drawer',
               'eggplant', 'spoon', 'carrot', 'plate larger', 'table cloth shorter', 
               'yellow basket', 'green cube', 'yellow cube', 'goal_site']

_BACKGROUND_CLASSNAMES = ['floor', 'wall', 'ceiling']

_NAME_TO_ALIAS_GDINO = {
    '7up can': 'white 7up pop can',
    'redbull can': 'redbull pop can',
    'coke can': 'coke pop can',
    'pepsi can': 'pepsi pop can',
    'sponge': 'green sponge',
    'plate larger': 'plate',
    'table cloth shorter': 'blue towel cloth',
    'top drawer': 'dresser',
    'middle drawer': 'dresser',
    'bottom drawer': 'dresser',
    'robot': 'robotic arm',
}

_NAME_TO_ALIAS_YOLO_WORLD = {
    '7up can': '7up pop can',
    'redbull can': 'redbull pop can',
    'coke can': 'coke pop can',
    'pepsi can': 'pepsi pop can',
    'blue plastic bottle': 'mineral water bottle with label',
    'top drawer': 'cabinet with drawers',
    'middle drawer': 'cabinet with drawers',
    'bottom drawer': 'cabinet with drawers',
    'plate larger': 'plate',
    'table cloth shorter': 'blue towel cloth',
}

_NAME_TO_ALIAS_SED = {
    'blue plastic bottle': 'mineral water bottle with label',
    'top drawer': 'cabinet with drawers',
    'middle drawer': 'cabinet with drawers',
    'bottom drawer': 'cabinet with drawers',
    'plate larger': 'plate',
    'robot': 'robot manipulator',
    'table cloth shorter': 'blue towel cloth',
}

_SAM2_MODEL_CFG = os.path.join('configs', 'sam2.1', 'sam2.1_hiera_l.yaml')
_PCD_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_CHECKPOINT_ROOT = os.environ.get(
    'PCD_CHECKPOINT_ROOT',
    os.path.join(_PCD_ROOT, 'checkpoint'),
)
_SAM2_CHECKPOINT = os.path.join(_CHECKPOINT_ROOT, 'sam2.1_hiera_large.pt')

_YOLO_WORLD_CFG = os.path.join(_PCD_ROOT, 'third_party', 'yolo_world', 'configs', 'pretrain', 'yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py')
_YOLO_WORLD_CHECKPOINT = os.path.join(_PCD_ROOT, 'pretrained', 'l_stage1-7d280586.pth')

_GROUNDING_DINO_CHECKPOINT = os.path.join(_CHECKPOINT_ROOT, 'grounding-dino-base')

def postprocess_mask(mask):
    if mask is None or not mask.any():
        return None
    
    # only keep the largest connected component
    mask = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    largest_area = stats[1:, cv2.CC_STAT_AREA].max()
    if num_labels > 1:
        # for each component, if area is smaller than 0.1 * largest_area, set it to 0
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 0.1 * largest_area:
                mask[labels == i] = 0

    # get counter of mask and fill poly
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(mask)
    cv2.fillPoly(mask, contours, 1)
    return (mask > 0)


def filter_boxes_by_smallest_area(boxes, scores, labels, iou_threshold=0.5):
    if not isinstance(boxes, torch.Tensor):
        boxes = torch.tensor(boxes)
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores)

    if len(boxes) == 0:
        return boxes, scores, labels

    valid_indices = [
        index
        for index, label in enumerate(labels)
        if label is not None and label.strip()
    ]
    if not valid_indices:
        return boxes[:0], scores[:0], []

    valid_indices = torch.tensor(valid_indices, device=boxes.device)
    boxes = boxes[valid_indices]
    scores = scores[valid_indices]
    labels = [labels[index] for index in valid_indices.tolist()]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    final_keep_indices = []
    for label in set(labels):
        indices = torch.tensor(
            [index for index, value in enumerate(labels) if value == label],
            device=boxes.device,
        )
        sorted_indices = torch.argsort(areas[indices])
        remaining = indices[sorted_indices]

        while len(remaining) > 0:
            current = remaining[0]
            final_keep_indices.append(current.item())
            if len(remaining) == 1:
                break

            rest = remaining[1:]
            intersection_top_left = torch.max(boxes[current, :2], boxes[rest, :2])
            intersection_bottom_right = torch.min(
                boxes[current, 2:], boxes[rest, 2:]
            )
            intersection_size = (
                intersection_bottom_right - intersection_top_left
            ).clamp(min=0)
            intersection_area = intersection_size[:, 0] * intersection_size[:, 1]
            overlap_ratio = intersection_area / (areas[current] + 1e-6)
            remaining = rest[overlap_ratio < iou_threshold]

    final_keep_indices = sorted(final_keep_indices)
    return (
        boxes[final_keep_indices],
        scores[final_keep_indices],
        [labels[index] for index in final_keep_indices],
    )


def expand_boxes(boxes, image_height, image_width, ratio=0.1):
    if len(boxes) == 0:
        return boxes

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    expand_width = widths * (ratio / 2)
    expand_height = heights * (ratio / 2)
    return torch.stack(
        [
            (boxes[:, 0] - expand_width).clamp(min=0),
            (boxes[:, 1] - expand_height).clamp(min=0),
            (boxes[:, 2] + expand_width).clamp(max=image_width),
            (boxes[:, 3] + expand_height).clamp(max=image_height),
        ],
        dim=1,
    )


def get_center_points_from_boxes(boxes, ratio_x=0.6, ratio_y=0.4):
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    point_coords = torch.stack(
        [boxes[:, 0] + widths * ratio_x, boxes[:, 1] + heights * ratio_y],
        dim=1,
    ).unsqueeze(1)
    point_labels = torch.ones(
        point_coords.shape[:2], dtype=torch.int, device=boxes.device
    )
    return point_coords, point_labels


def predict_sam2_masks_only(
    image_predictor,
    point_coords=None,
    point_labels=None,
    box=None,
    mask_input=None,
    multimask_output=False,
    normalize_coords=True,
):
    """Run SAM2 without copying unused IoU and low-resolution outputs to CPU."""
    if not image_predictor._is_image_set:
        raise RuntimeError(
            "An image must be set with .set_image(...) before mask prediction."
        )

    mask_input, unnorm_coords, labels, unnorm_box = image_predictor._prep_prompts(
        point_coords,
        point_labels,
        box,
        mask_input,
        normalize_coords,
    )
    masks, _, _ = image_predictor._predict(
        unnorm_coords,
        labels,
        unnorm_box,
        mask_input,
        multimask_output,
        return_logits=False,
    )
    return masks.squeeze(0)


class GroundedSAMPredictor:
    def __init__(self, sam2_model=None):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        if sam2_model is None:
            sam2_model = build_sam2(_SAM2_MODEL_CFG, _SAM2_CHECKPOINT)
        sam2_model.to(dtype=torch.bfloat16)
        self.image_predictor = SAM2ImagePredictor(sam2_model)
        self.processor = AutoProcessor.from_pretrained(_GROUNDING_DINO_CHECKPOINT)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            _GROUNDING_DINO_CHECKPOINT
        ).to(device='cuda:0', dtype=torch.bfloat16)
    
    @torch.no_grad()
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def predict(self, image, prompts, task):
        image_pil = Image.fromarray(image.copy())
        target_list = [
            _NAME_TO_ALIAS_GDINO.get(prompt, prompt) for prompt in prompts
        ]
        robot_index = target_list.index(_NAME_TO_ALIAS_GDINO['robot'])
        if not task.startswith('google_robot'):
            target_list[robot_index] = 'black ' + target_list[robot_index]
        text = '. '.join(target_list) + '.'

        inputs = self.processor(
            image_pil, text=text, return_tensors="pt"
        ).to('cuda:0')
        outputs = self.grounding_model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.25,
            text_threshold=0.25,
            target_sizes=[image_pil.size[::-1]],
        )[0]
        input_boxes, scores, input_labels = filter_boxes_by_smallest_area(
            result['boxes'], result['scores'], result['labels']
        )
        input_boxes = expand_boxes(
            input_boxes, image.shape[0], image.shape[1], ratio=0.02
        )
        point_coords, point_labels = get_center_points_from_boxes(input_boxes)
        labels = []
        for label in input_labels:
            if label.endswith('pop'):
                label += ' can'
            labels.append(label)

        del outputs, inputs, result

        self.image_predictor.set_image(image.copy())
        masks = predict_sam2_masks_only(
            self.image_predictor,
            point_coords=point_coords,
            point_labels=point_labels,
            box=input_boxes,
            multimask_output=False,
        )

        if masks.ndim == 4:
            masks = masks.squeeze(1)
        masks = masks.detach().cpu().numpy()
        self.image_predictor.reset_predictor()

        output_masks = []
        for name in target_list:
            if name in labels:
                indexs = index_all(labels, name)
                selected_scores = scores[indexs]
                argmax_scores = selected_scores.argmax()
                output_masks.append(masks[indexs[argmax_scores]] > 0)
            else:
                output_masks.append(None)
        output_masks = [postprocess_mask(mask) for mask in output_masks]

        return output_masks

    def reset(self):
        self.image_predictor.reset_predictor()


class SEDPredictor:
    def __init__(self):
        from detectron2.config import get_cfg
        from detectron2.projects.deeplab import add_deeplab_config
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from third_party.SED.demo.predictor import VisualizationDemo as SEDDemo
        from third_party.SED.sed import add_sed_config

        with open('contrast_utils/SED/datasets/simpler.json', 'w') as f:
            self.classnames = [_NAME_TO_ALIAS_SED.get(name, name) for name in _CLASSNAMES] + _BACKGROUND_CLASSNAMES
            self.classnames = list(set(self.classnames))
            json.dump(self.classnames, f)

        def setup_cfg():
            # load config from file and command-line arguments
            cfg = get_cfg()
            add_deeplab_config(cfg)
            add_sed_config(cfg)
            cfg.merge_from_file('contrast_utils/SED/configs/convnextL_768.yaml')
            cfg.merge_from_list(['MODEL.WEIGHTS', 'pretrained/sed.pth'])
            cfg.freeze()
            return cfg
        cfg = setup_cfg()
        self.model = SEDDemo(cfg)

        sam2_checkpoint = "pretrained/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
        self.image_predictor = SAM2ImagePredictor(sam2_image_model)
    
    @torch.no_grad()
    def predict(self, image, prompts):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        sem_seg = self.model.predictor(image)['sem_seg'].argmax(dim=0)
        
        masks = []
        for prompt in prompts:
            idx = self.classnames.index(_NAME_TO_ALIAS_SED.get(prompt, prompt))
            mask = (sem_seg == idx).detach().cpu().numpy()
            if mask.any():
                masks.append(postprocess_mask(mask))
            else:
                masks.append(None)
        
        self.image_predictor.set_image(image)
        # if mask is None, set box [0, 0, 1, 1]
        input_boxes = np.array([mask_to_bbox(mask) if mask is not None 
                                else np.array([0.0, 0.0, 1.0, 1.0]) for mask in masks])
        refined_masks, _, _ = self.image_predictor.predict(point_coords=None, point_labels=None,
                                                            box=input_boxes, multimask_output=False)
        if refined_masks.ndim == 4:
            refined_masks = refined_masks.squeeze(1)
        
        # add None to refined_masks
        refined_masks = [(refined_masks[i] > 0) if mask is not None else None for i, mask in enumerate(masks)]
        return [postprocess_mask(mask) for mask in refined_masks]


class YoloWorldPredictor:
    def __init__(self):
        # load config
        from mmengine.config import Config
        from mmengine.dataset import Compose
        from mmdet.apis import init_detector
        from mmdet.utils import get_test_pipeline_cfg

        cfg = Config.fromfile(_YOLO_WORLD_CFG)
        cfg.work_dir = os.path.join('./work_dirs')
        cfg.load_from = _YOLO_WORLD_CHECKPOINT
        self.model = init_detector(cfg, checkpoint=_YOLO_WORLD_CHECKPOINT, device='cpu')
        test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
        test_pipeline_cfg[0].type = 'mmdet.LoadImageFromNDArray'
        self.test_pipeline = Compose(test_pipeline_cfg)
        
        from sam2.build_sam import build_sam2_video_predictor, build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.video_predictor = build_sam2_video_predictor(_SAM2_MODEL_CFG, _SAM2_CHECKPOINT)
        sam2_image_model = build_sam2(_SAM2_MODEL_CFG, _SAM2_CHECKPOINT, device='cpu')
        self.image_predictor = SAM2ImagePredictor(sam2_image_model)
    
    @torch.no_grad()
    def predict(self, image, prompts):
        prompts = [_NAME_TO_ALIAS_YOLO_WORLD.get(p, p) for p in prompts]
        result = self.yolo_world_inference(self.model, image, prompts, self.test_pipeline)

        input_boxes = result['boxes']
        scores = result['scores']
        labels = []
        
        if len(input_boxes) == 0:
            return [None] * len(prompts)

        # avoid one alias to multi name mapping
        alias_to_name = {v: k for k, v in _NAME_TO_ALIAS_YOLO_WORLD.items() if k in prompts}

        for l in result['label_texts']:
            labels.append(alias_to_name.get(l, l))

        self.image_predictor.set_image(image.copy())
        masks, _, _ = self.image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        output_masks = []
        for name in prompts:
            if name in labels:
                indexs = index_all(labels, name)
                selected_scores = scores[indexs]
                argmax_scores = selected_scores.argmax()
                output_masks.append(masks[indexs[argmax_scores]] > 0)
            else:
                output_masks.append(None)

        return [postprocess_mask(mask) for mask in output_masks]
    
    def yolo_world_inference(self, model, image, texts, test_pipeline, score_thr=0.3, max_dets=100):
        texts.append(' ')
        texts = [[t] for t in texts]

        data_info = dict(img=image, img_id=0, texts=texts)
        data_info = test_pipeline(data_info)
        data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                        data_samples=[data_info['data_samples']])
        with torch.no_grad():
            output = model.test_step(data_batch)[0]
        pred_instances = output.pred_instances
        # score thresholding
        pred_instances = pred_instances[pred_instances.scores.float() > score_thr]
        # max detections
        if len(pred_instances.scores) > max_dets:
            indices = pred_instances.scores.float().topk(max_dets)[1]
            pred_instances = pred_instances[indices]

        pred_instances = pred_instances.cpu().numpy()
        boxes = pred_instances['bboxes']
        labels = pred_instances['labels']
        scores = pred_instances['scores']
        label_texts = [texts[x][0] for x in labels]
        return {
            'boxes': boxes,
            'labels': labels,
            'scores': scores,
            'label_texts': label_texts,
        }


class VisualPromptPredictor:
    def __init__(self):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        sam2_image_model = build_sam2(_SAM2_MODEL_CFG, _SAM2_CHECKPOINT)
        self.image_predictor = SAM2ImagePredictor(sam2_image_model)
        self.points = None
        self.boxes = None
    
    def set_points(self, points):
        # points is a list contains None
        self.points = np.array([point for point in points if point is not None])
        self.mask_index = [point is not None for point in points]
        
    def set_boxes(self, boxes):
        self.boxes = np.array([box for box in boxes if box is not None])
        self.mask_index = [box is not None for box in boxes]
    
    def predict(self, image, prompts):
        # only support one of points or boxes
        assert self.points is not None or self.boxes is not None, 'points or boxes must be provided!'
        assert self.points is None or self.boxes is None, 'only one of points or boxes can be provided!'
        
        self.image_predictor.set_image(image.copy())
        if self.points is not None:
            point_labels = np.ones(self.points.shape[:-1], dtype=int)
            masks, _, _ = self.image_predictor.predict(point_coords=self.points, point_labels=point_labels, 
                                                       box=None, multimask_output=False)
        elif self.boxes is not None:
            masks, _, _ = self.image_predictor.predict(point_coords=None, point_labels=None, 
                                                       box=self.boxes, multimask_output=False)
        
        if masks.ndim == 4:
            masks = masks.squeeze(1)
            
        out_masks = []
        count = 0
        for mask_index in self.mask_index:
            if mask_index:
                out_masks.append(masks[count])
                count += 1
            else:
                out_masks.append(None)
        return [postprocess_mask(mask) for mask in out_masks]


class TrackingPredictorV2:
    def __init__(self, predictor, video_predictor=None):
        self.predictor = predictor
        if video_predictor is None:
            from sam2.build_sam import build_sam2_camera_predictor
            video_predictor = build_sam2_camera_predictor(
                _SAM2_MODEL_CFG,
                _SAM2_CHECKPOINT,
            )
            video_predictor.to(dtype=torch.bfloat16)
        self.video_predictor = video_predictor
        self.start_tracking = False

    @torch.inference_mode()
    def add_new_mask_without_output(self, frame_idx, obj_id, mask):
        """Add an initial mask without constructing an unused video-size mask."""
        predictor = self.video_predictor
        obj_idx = predictor._obj_id_to_idx(obj_id)
        point_inputs_per_frame = predictor.condition_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = predictor.condition_state["mask_inputs_per_obj"][obj_idx]

        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.bool)
        assert mask.dim() == 2
        mask_h, mask_w = mask.shape
        mask_inputs_orig = mask[None, None].float().to(
            predictor.condition_state["device"]
        )

        if mask_h != predictor.image_size or mask_w != predictor.image_size:
            mask_inputs = torch.nn.functional.interpolate(
                mask_inputs_orig,
                size=(predictor.image_size, predictor.image_size),
                align_corners=False,
                mode="bilinear",
                antialias=True,
            )
            mask_inputs = (mask_inputs >= 0.5).float()
        else:
            mask_inputs = mask_inputs_orig

        mask_inputs_per_frame[frame_idx] = mask_inputs
        point_inputs_per_frame.pop(frame_idx, None)
        is_init_cond_frame = (
            frame_idx not in predictor.condition_state["frames_already_tracked"]
        )
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = predictor.condition_state["frames_already_tracked"][frame_idx][
                "reverse"
            ]
        obj_output_dict = predictor.condition_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = predictor.condition_state["temp_output_dict_per_obj"][obj_idx]
        is_cond = is_init_cond_frame or predictor.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        current_out, _ = predictor._run_single_frame_inference(
            output_dict=obj_output_dict,
            frame_idx=frame_idx,
            batch_size=1,
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=None,
            mask_inputs=mask_inputs,
            reverse=reverse,
            run_mem_encoder=False,
            prev_sam_mask_logits=None,
        )
        obj_temp_output_dict[storage_key][frame_idx] = current_out
        
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def predict(self, image, prompts, task, return_masks=True):
        if not self.start_tracking:
            if isinstance(self.predictor, GroundedSAMPredictor):
                masks = self.predictor.predict(image, prompts, task)
            else:
                masks = self.predictor.predict(image, prompts)
            if all(mask is None for mask in masks):
                return masks if return_masks else None

            self.video_predictor.load_first_frame(image)
            for mask, prompt in zip(masks, prompts):
                obj_id = _CLASSNAMES.index(prompt) + 1
                if mask is not None:
                    self.add_new_mask_without_output(0, obj_id, mask)
                    
            self.start_tracking = True
            return masks if return_masks else None
        
        first_tracking_call = not self.video_predictor.condition_state.get(
            "tracking_has_started",
            False,
        )
        obj_ids, mask_logits = self.video_predictor.track(
            image,
            return_masks=return_masks,
        )

        # The first tracking preflight has encoded the conditioning frame into
        # memory, so its FP32 image and backbone cache are no longer needed.
        if first_tracking_call:
            self.video_predictor.condition_state["cached_features"].clear()

        if not return_masks:
            return None

        binary_masks = (mask_logits > 0).squeeze(1).cpu().numpy()
        name2mask = dict()
        for idx, obj_id in enumerate(obj_ids):
            classname = _CLASSNAMES[obj_id - 1]
            mask = binary_masks[idx]
            if not mask.any():
                mask = None
            name2mask[classname] = mask
        masks = [name2mask.get(p, None) for p in prompts]
        return masks
    
    def reset(self):
        self.start_tracking = False
        self.video_predictor.frame_idx = 0
        self.video_predictor.condition_state = {}
        image_predictor = getattr(self.predictor, "image_predictor", None)
        if image_predictor is not None:
            image_predictor.reset_predictor()


def build_predictor(predictor_name):
    if predictor_name == 'grounded_sam_tracking':
        from sam2.build_sam import build_sam2_camera_predictor

        shared_sam2_model = build_sam2_camera_predictor(
            _SAM2_MODEL_CFG,
            _SAM2_CHECKPOINT,
        )
        shared_sam2_model.to(dtype=torch.bfloat16)
        return TrackingPredictorV2(
            GroundedSAMPredictor(sam2_model=shared_sam2_model),
            video_predictor=shared_sam2_model,
        )
    if predictor_name == 'sed':
        return SEDPredictor()
    if predictor_name == 'yolo_world':
        return YoloWorldPredictor()
    if predictor_name == 'sed_tracking':
        return TrackingPredictorV2(SEDPredictor())
    if predictor_name == 'yolo_world_tracking':
        return TrackingPredictorV2(YoloWorldPredictor())
    if predictor_name == 'point_tracking':
        return TrackingPredictorV2(VisualPromptPredictor())
    if predictor_name == 'box_tracking':
        return TrackingPredictorV2(VisualPromptPredictor())
    raise ValueError(f'predictor_name {predictor_name} is not supported')


def predict_masks_with_predictor(
    image, prompts, predictor, task, return_masks=True
):
    if isinstance(predictor, TrackingPredictorV2):
        return predictor.predict(
            image,
            prompts,
            task,
            return_masks=return_masks,
        )
    if not return_masks:
        return None
    masks = predictor.predict(image, prompts)
    return masks
