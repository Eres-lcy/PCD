import cv2
import numpy as np

from mani_skill2_real2sim import ASSET_DIR
from mani_skill2_real2sim.utils.registration import register_env

from .grasp_single_in_scene import GraspSingleOpenedCokeCanInSceneEnv


def _load_rgb_overlay(filename: str) -> tuple[str, np.ndarray]:
    overlay_path = ASSET_DIR / "real_inpainting" / filename
    overlay_bgr = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
    if overlay_bgr is None:
        raise FileNotFoundError(f"OOD RGB overlay is missing: {overlay_path}")
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB) / 255
    return str(overlay_path), overlay_rgb


class _FixedCokeCanOverlayMixin:
    overlay_filename: str

    def _additional_prepackaged_config_reset(self, options):
        reconfigure = super()._additional_prepackaged_config_reset(options)
        self.rgb_overlay_path, self.rgb_overlay_img = _load_rgb_overlay(
            self.overlay_filename
        )
        return reconfigure


@register_env("GraspSingleCokeCanDarkerInScene-v0", max_episode_steps=80)
class GraspSingleCokeCanDarkerInSceneEnv(GraspSingleOpenedCokeCanInSceneEnv):
    def get_obs(self):
        obs = super().get_obs()
        for camera_name in self.rgb_overlay_cameras:
            obs["image"][camera_name]["Color"] = np.power(
                obs["image"][camera_name]["Color"], 2.0
            )
        return obs


@register_env(
    "GraspSingleOpenedCokeCanDrawerVariantInScene-v0", max_episode_steps=80
)
class GraspSingleOpenedCokeCanDrawerVariantInSceneEnv(
    _FixedCokeCanOverlayMixin, GraspSingleOpenedCokeCanInSceneEnv
):
    overlay_filename = "google_coke_can_real_eval_1_drawer_variant.png"


@register_env(
    "GraspSingleOpenedCokeCanLightVariantInScene-v0", max_episode_steps=80
)
class GraspSingleOpenedCokeCanLightVariantInSceneEnv(
    _FixedCokeCanOverlayMixin, GraspSingleOpenedCokeCanInSceneEnv
):
    overlay_filename = "google_coke_can_real_eval_1_light_variant.png"


@register_env(
    "GraspSingleOpenedCokeCanTablePaperVariantInScene-v0", max_episode_steps=80
)
class GraspSingleOpenedCokeCanTablePaperVariantInSceneEnv(
    _FixedCokeCanOverlayMixin, GraspSingleOpenedCokeCanInSceneEnv
):
    overlay_filename = "google_coke_can_real_eval_1_paper.png"


@register_env(
    "GraspSingleOpenedCokeCanTableStoneVariantInScene-v0", max_episode_steps=80
)
class GraspSingleOpenedCokeCanTableStoneVariantInSceneEnv(
    _FixedCokeCanOverlayMixin, GraspSingleOpenedCokeCanInSceneEnv
):
    overlay_filename = "google_coke_can_real_eval_1_stone.png"
