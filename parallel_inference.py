"""Unified multi-GPU evaluator for baseline, PCD-Mask, and PCD-Fast."""

from __future__ import annotations

import argparse
import multiprocessing
import os
import os.path as osp
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np

from utils import (
    Logger,
    convert_numpy_or_torch_to_python,
    reset_logging,
    stat_final,
    stat_first,
    stat_info,
    summarize,
    transform_mask,
    write_video,
)


METHODS = ("baseline", "pcd-mask", "pcd-fast")
POLICIES = ("openvla", "pizero", "gr00t")
MASK_METHODS = ("gt", "point_tracking", "box_tracking", "grounded_sam_tracking")
PCD_ROOT = Path(__file__).resolve().parent
THIRD_PARTY_ROOT = PCD_ROOT / "third_party"
for source_dir in ("pi0", "openvla"):
    source_path = str(THIRD_PARTY_ROOT / source_dir)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
os.environ.setdefault("PCD_PROJECT_ROOT", str(PCD_ROOT))
os.environ.setdefault("PCD_CHECKPOINT_ROOT", str(PCD_ROOT / "checkpoint"))


def get_image_from_maniskill2_obs_dict(env, obs, camera_name=None):
    if camera_name is None:
        robot_uid = env.unwrapped.robot_uid
        if "google_robot" in robot_uid:
            camera_name = "overhead_camera"
        elif "widowx" in robot_uid:
            camera_name = "3rd_view_camera"
        elif "panda" in robot_uid:
            camera_name = "overhead_camera"
        else:
            raise NotImplementedError(f"Unsupported robot UID: {robot_uid}")
    return obs["image"][camera_name]["rgb"]


class ParallelRunner:
    def __init__(
        self,
        *,
        policy,
        method,
        checkpoint,
        task,
        result_root,
        n_trajs=100,
        seed=0,
        num_gpus=1,
        gpu_ids=(0,),
        alpha=None,
        mask_by="gt",
        inpaint_mode="lama",
    ):
        self.policy = policy
        self.method = method
        self.checkpoint = checkpoint
        self.task = task
        self.result_root = result_root
        self.n_trajs = n_trajs
        self.seed = seed
        self.num_gpus = num_gpus
        self.gpu_ids = list(gpu_ids)
        self.alpha = alpha
        self.mask_by = mask_by
        self.inpaint_mode = inpaint_mode
        self.segmented_image_generator = None
        self.contrast_image_generator = None

        if self.num_gpus < 1:
            raise ValueError("--num-gpus must be at least one")
        if self.num_gpus > len(self.gpu_ids):
            raise ValueError(
                f"--num-gpus={self.num_gpus} exceeds --gpu-ids={self.gpu_ids}"
            )
        self.active_gpu_ids = self.gpu_ids[: self.num_gpus]

    @property
    def is_pcd_mask(self):
        return self.method == "pcd-mask"

    @property
    def is_pcd_fast(self):
        return self.method == "pcd-fast"

    def run(self):
        if self.num_gpus == 1:
            self._run_serial()
        else:
            self._run_parallel()

    def _run_serial(self):
        if not self._set_result_dir():
            return
        self._build_logger()
        infos = self.run_episodes(
            self.active_gpu_ids[0], range(self.n_trajs), show_detail=True
        )
        self._finish(infos)

    def _run_parallel(self):
        if not self._set_result_dir():
            return
        self._build_logger()
        info_queue = multiprocessing.Queue()
        episode_groups = [
            list(range(worker_index, self.n_trajs, self.num_gpus))
            for worker_index in range(self.num_gpus)
        ]
        processes = []
        for worker_index, (gpu_id, episodes) in enumerate(
            zip(self.active_gpu_ids, episode_groups)
        ):
            self.logger.info(f"Allocating episodes for GPU {gpu_id}: {episodes}.")
            process = multiprocessing.Process(
                target=self.run_episodes,
                args=(gpu_id, episodes, info_queue, worker_index == 0),
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join()
        failed = [process.pid for process in processes if process.exitcode != 0]
        if failed:
            raise RuntimeError(f"Inference workers failed: {failed}")

        infos = [info_queue.get() for _ in range(self.n_trajs)]
        self._build_logger(mode="a")
        self._finish(infos)

    def _finish(self, infos):
        info = stat_info(infos)
        self.logger.infos("Results", info)
        source = osp.join(self.result_dir, "000.log")
        target = osp.join(
            self.result_dir,
            f"000_success_{round(float(info['success']), 4)}.log",
        )
        os.rename(source, target)

    def run_episodes(self, gpu_id, episodes, info_queue=None, show_detail=False):
        episodes = list(episodes)
        env, policy = self._build_episode(gpu_id, show_detail)
        infos = []
        for episode_index, episode in enumerate(episodes):
            self.logger.info(f"Running episode {episode} on GPU {gpu_id}.")
            try:
                info = self.run_episode(
                    env, policy, episode, show_detail=show_detail
                )
                should_reload_pi0 = (
                    self.policy == "pizero"
                    and self.method in {"baseline", "pcd-fast"}
                    and episode_index + 1 < len(episodes)
                )
                if should_reload_pi0:
                    self.logger.info("Reloading Pi0 after the episode.")
                    policy.model.to("cpu")
                    del policy.model
                    del policy
                    policy = self._build_policy(show_detail, gpu_id)
            except Exception as error:
                self.logger.error(
                    f"Episode {episode} failed with error: {error}."
                )
                self.logger.error(traceback.format_exc())
                self._write_error(episode, error)
                raise
            infos.append(info)
            if info_queue is not None:
                info_queue.put(info)
        return infos

    def run_episode(self, env, policy, episode, show_detail=False):
        if self.segmented_image_generator is not None:
            self.segmented_image_generator.reset()
        if self.contrast_image_generator is not None:
            self.contrast_image_generator.reset()

        obs, _ = env.reset(seed=episode + self.seed)
        instruction = env.unwrapped.get_language_instruction()
        policy.reset(instruction, seed=episode + self.seed)
        self.logger.info(f"Episode {episode} initial instruction: {instruction}")

        predicted_terminated = False
        truncated = False
        timestep = 0
        frames = []
        step_infos = []

        image = get_image_from_maniskill2_obs_dict(env, obs)
        segmentation_mask = None
        contrast_image = None
        if self.segmented_image_generator is not None:
            mask = self.segmented_image_generator.generate(
                obs, instruction, self.task
            )
            segmentation_mask = transform_mask(mask, image.shape[:2])
        if self.contrast_image_generator is not None:
            contrast_image = self.contrast_image_generator.generate(
                obs, instruction, self.task
            )
        frames.append(image)

        while not (predicted_terminated or truncated):
            raw_action, actions = self._policy_step(
                policy,
                image=image,
                contrast_image=contrast_image,
                instruction=instruction,
                timestep=timestep,
                segmentation_mask=segmentation_mask,
                proprio=obs["agent"]["eef_pos"],
                episode=episode,
                env=env,
            )
            del raw_action
            if not isinstance(actions, list):
                actions = [actions]

            instruction_changed = False
            for action_index, action in enumerate(actions):
                obs, _, _, truncated, info = env.step(
                    np.concatenate(
                        (
                            action["world_vector"],
                            action["rot_axangle"],
                            action["gripper"],
                        )
                    )
                )
                image = get_image_from_maniskill2_obs_dict(env, obs)
                frames.append(image)
                timestep += 1
                info = convert_numpy_or_torch_to_python(info)
                step_infos.append(info)
                predicted_terminated = bool(action["terminate_episode"][0] > 0)
                is_final_subtask = env.unwrapped.is_final_subtask()
                terminal_transition = truncated or (
                    predicted_terminated and is_final_subtask
                )

                if show_detail:
                    self.logger.info(f"Step {timestep}: {info}")

                # Preserve the original OpenVLA/Pi0 tracker cadence. These
                # generators consume the instruction that produced the action.
                if self.is_pcd_mask and self.policy != "gr00t" and not terminal_transition:
                    # Pi0 emits an action chunk. Intermediate observations must
                    # update SAM2 memory, but their contrast images are never
                    # consumed by the policy, so only materialize the last one.
                    return_contrast_image = action_index + 1 == len(actions)
                    next_contrast_image = self.contrast_image_generator.generate(
                        obs,
                        instruction,
                        self.task,
                        return_image=return_contrast_image,
                    )
                    if return_contrast_image:
                        contrast_image = next_contrast_image
                elif (
                    self.is_pcd_fast
                    and self.policy == "openvla"
                    and not terminal_transition
                    and timestep % 4 == 0
                ):
                    should_export = timestep % policy.reset_frequency == 0
                    mask = self.segmented_image_generator.generate(
                        obs,
                        instruction,
                        self.task,
                        return_masks=should_export,
                    )
                    if should_export:
                        segmentation_mask = transform_mask(mask, image.shape[:2])

                if predicted_terminated and not is_final_subtask:
                    predicted_terminated = False
                    env.advance_to_next_subtask()

                new_instruction = env.unwrapped.get_language_instruction()
                if new_instruction != instruction:
                    instruction = new_instruction
                    instruction_changed = True
                    if show_detail:
                        self.logger.info(f"New instruction: {instruction}")

                if predicted_terminated or truncated:
                    break

            terminal = predicted_terminated or truncated
            if self.is_pcd_mask and self.policy == "gr00t" and not terminal:
                contrast_image = self.contrast_image_generator.generate(
                    obs, instruction, self.task
                )
            elif self.is_pcd_fast and self.policy == "pizero" and not terminal:
                mask = self.segmented_image_generator.generate(
                    obs, instruction, self.task
                )
                segmentation_mask = transform_mask(mask, image.shape[:2])
            elif (
                self.is_pcd_fast
                and self.policy == "gr00t"
                and not terminal
                and (instruction_changed or timestep % 4 == 0)
            ):
                mask = self.segmented_image_generator.generate(
                    obs, instruction, self.task
                )
                segmentation_mask = transform_mask(mask, image.shape[:2])

        if not step_infos:
            raise RuntimeError("The environment terminated without producing a step")
        info = summarize(step_infos)
        info.update(stat_first(step_infos))
        info.update(stat_final(step_infos))
        success = info["success"]
        self.logger.info(f"Episode {episode} finished with success {success}.")
        write_video(
            frames,
            osp.join(
                self.result_dir,
                f"episode_{episode}_success_{success}.mp4",
            ),
        )
        return info

    def _policy_step(
        self,
        policy,
        *,
        image,
        contrast_image,
        instruction,
        timestep,
        segmentation_mask,
        proprio,
        episode,
        env,
    ):
        if self.method == "baseline":
            if self.policy == "openvla":
                result = policy.step(image, instruction)
            elif self.policy == "pizero":
                result = policy.step(image, instruction, proprio=proprio)
            else:
                result = policy.step(
                    image,
                    instruction,
                    timestep=timestep,
                    segmentation_mask=None,
                    proprio=proprio,
                )
        elif self.is_pcd_mask:
            if self.policy == "gr00t":
                result = policy.step(
                    image,
                    instruction,
                    timestep=timestep,
                    segmentation_mask=None,
                    contrast_image=contrast_image,
                    proprio=proprio,
                )
            else:
                result = policy.step(
                    image, contrast_image, instruction, proprio=proprio
                )
        else:
            common = dict(
                timestep=timestep,
                segmentation_mask=segmentation_mask,
                logger=self.logger,
                episode=episode,
            )
            if self.policy == "openvla":
                result = policy.step(image, instruction, **common)
            elif self.policy == "pizero":
                result = policy.step(
                    image,
                    instruction,
                    proprio=proprio,
                    env=env,
                    **common,
                )
            else:
                result = policy.step(
                    image,
                    instruction,
                    proprio=proprio,
                    **common,
                )
        return result[0], result[1]

    def _build_episode(self, gpu_id, show_detail):
        self._set_gpu(gpu_id)
        uses_torch_compile = self.policy == "pizero"
        if uses_torch_compile:
            from torch_compile_utils import configure_torch_compile_cache

            compile_cache_dir = configure_torch_compile_cache(PCD_ROOT)
            if show_detail:
                self.logger.info(f"Torch compile cache: {compile_cache_dir}")
        env = self._build_environment()
        self.segmented_image_generator = None
        self.contrast_image_generator = None
        if self.is_pcd_fast:
            self._build_segmented_image_generator(env, show_detail)
        elif self.is_pcd_mask:
            self._build_contrast_image_generator(env, show_detail)
        policy = self._build_policy(show_detail, gpu_id)
        return env, policy

    def _set_gpu(self, gpu_id):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        import tensorflow as tf

        tf.config.list_physical_devices("GPU")
        try:
            tf.config.experimental.set_visible_devices([], "GPU")
        except RuntimeError:
            pass

    def _build_environment(self):
        import simpler_env

        return simpler_env.make(self.task)

    def _policy_opts(self):
        opts = {}
        if self.method != "baseline" and self.alpha is not None:
            opts["alpha"] = self.alpha
        if self.is_pcd_mask:
            opts["by"] = self.mask_by
            opts["inpaint_mode"] = self.inpaint_mode
        elif self.is_pcd_fast:
            opts["by"] = self.mask_by
        return opts

    def _build_policy(self, show_detail=False, gpu_id=0):
        from properties import (
            GR00T_SERVER_BASE_PORT,
            GR00T_SERVER_HOST,
            get_policy_config,
        )

        config = get_policy_config(
            self.policy,
            self.checkpoint,
            self.task,
            self._policy_opts(),
            contrast=self.method != "baseline",
            early_exit=self.is_pcd_fast,
        )
        if self.policy == "gr00t":
            worker_index = self.active_gpu_ids.index(gpu_id)
            config["server_address"] = (
                f"tcp://{GR00T_SERVER_HOST}:"
                f"{GR00T_SERVER_BASE_PORT + worker_index}"
            )
        if show_detail:
            self.logger.infos("Policy Config", config)

        if self.policy == "openvla":
            if self.method == "baseline":
                from simpler_env.policies.openvla.openvla_model import (
                    OpenVLAInference,
                )

                policy = OpenVLAInference(gpu_id=gpu_id, **config)
            elif self.is_pcd_mask:
                from contrast_policies.openvla.pcd_mask_policy_output import (
                    OpenVLAPCDMaskPolicyOutput,
                )

                policy = OpenVLAPCDMaskPolicyOutput(gpu_id=gpu_id, **config)
            else:
                from contrast_policies.openvla.pcd_fast_policy_output import (
                    OpenVLAPCDFastPolicyOutput,
                )

                policy = OpenVLAPCDFastPolicyOutput(gpu_id=gpu_id, **config)
        elif self.policy == "pizero":
            if self.method == "baseline":
                from simpler_env.policies.pizero.pizero_model_baseline import (
                    PiZeroBaselineInference,
                )

                policy = PiZeroBaselineInference(**config)
            elif self.is_pcd_mask:
                from contrast_policies.pizero.pcd_mask_policy_output import (
                    PiZeroPCDMaskPolicyOutput,
                )

                policy = PiZeroPCDMaskPolicyOutput(**config)
            else:
                from contrast_policies.pizero.pcd_fast_policy_output import (
                    PiZeroPCDFastPolicyOutput,
                )

                policy = PiZeroPCDFastPolicyOutput(**config)
        else:
            from contrast_policies.gr00t.remote_policy import Gr00tRemotePolicy

            policy = Gr00tRemotePolicy(**config)

        policy.task = self.task
        reset_logging()
        self._build_logger(mode="a")
        return policy

    def _build_segmented_image_generator(self, env, show_detail=False):
        from properties import get_segmented_image_generator_config
        from perception.segment_utils import get_segmented_image_generator
        from perception.segment_utils.mask_predictors import build_predictor

        config = get_segmented_image_generator_config({"by": self.mask_by})
        if show_detail:
            self.logger.infos("Segmentation Config", config)
        config["env"] = env
        generator = get_segmented_image_generator(config)
        generator.predictor = build_predictor(config["by"])
        self.segmented_image_generator = generator
        reset_logging()
        self._build_logger(mode="a", level="warning")

    def _build_contrast_image_generator(self, env, show_detail=False):
        from perception.contrast_utils import get_contrast_image_generator
        from properties import get_contrast_image_generator_config

        config = get_contrast_image_generator_config(
            {"by": self.mask_by, "inpaint_mode": self.inpaint_mode}
        )
        if show_detail:
            self.logger.infos("Contrast Image Config", config)
        config["env"] = env
        self.contrast_image_generator = get_contrast_image_generator(config)
        reset_logging()
        self._build_logger(mode="a", level="warning")

    def _build_logger(self, mode="w", level="info"):
        self.logger = Logger(
            osp.join(self.result_dir, "000.log"), mode, log_level=level
        )

    def _set_result_dir(self):
        model_name = self.checkpoint.replace("\\", "/").rstrip("/").split("/")[-1]
        parts = [self.result_root, model_name, self.method]
        if self.method != "baseline" and self.alpha is not None:
            parts.append(f"alpha={self.alpha}")
        if self.method != "baseline":
            parts.append(f"by={self.mask_by}")
        if self.is_pcd_mask:
            parts.append(f"inpaint={self.inpaint_mode}")
        parts.extend((f"seed_{self.seed}", self.task))
        self.result_dir = osp.join(*parts)

        if not osp.exists(self.result_dir):
            os.makedirs(self.result_dir)
            return True
        logs = [
            name
            for name in os.listdir(self.result_dir)
            if name.startswith("000") and name.endswith(".log")
        ]
        if any(name.startswith("000_success_") for name in logs):
            print(f"Completed result exists, skipping: {self.result_dir}")
            return False

        task_name = Path(self.task)
        result_root = Path(self.result_root).expanduser().resolve()
        task_parent = Path(osp.join(*parts[:-1])).expanduser().resolve()
        task_dir = Path(self.result_dir).expanduser()
        resolved_task_dir = task_dir.resolve()
        pcd_root = PCD_ROOT.resolve()
        if (
            task_name.name != self.task
            or self.task in {".", ".."}
            or resolved_task_dir.parent != task_parent
            or resolved_task_dir.name != self.task
            or not task_parent.is_relative_to(result_root)
            or not resolved_task_dir.is_relative_to(pcd_root)
        ):
            raise RuntimeError(
                f"Refusing to clear a path other than the current task directory: "
                f"{task_dir}"
            )

        print(f"Incomplete result exists, restarting: {self.result_dir}")
        shutil.rmtree(task_dir)
        task_dir.mkdir(parents=True)
        return True

    def _write_error(self, episode, error):
        path = osp.join(self.result_dir, f"000_episode_{episode}_error.log")
        with open(path, "w", encoding="utf-8") as file:
            file.write(f"{error}\n")
            traceback.print_exc(file=file)


def _parse_gpu_ids(value):
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values or any(not item.isdigit() for item in values):
        raise argparse.ArgumentTypeError(
            "GPU IDs must be a comma-separated list of non-negative integers"
        )
    gpu_ids = [int(item) for item in values]
    if len(gpu_ids) != len(set(gpu_ids)):
        raise argparse.ArgumentTypeError("GPU IDs must not contain duplicates")
    return gpu_ids


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--n-trajs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--gpu-ids", type=_parse_gpu_ids, default=[0])
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="PCD-Mask/PCD-Fast strength; use the model default when omitted.",
    )
    parser.add_argument(
        "--mask-by",
        choices=MASK_METHODS,
        default=None,
        help="PCD-Mask/PCD-Fast target-mask source (default: gt).",
    )
    parser.add_argument(
        "--inpaint-mode",
        choices=("lama", "telea", "ns"),
        default=None,
        help="PCD-Mask inpainting method (default: lama).",
    )
    args = parser.parse_args(argv)
    if args.n_trajs < 1:
        parser.error("--n-trajs must be at least one")
    if args.num_gpus is None:
        args.num_gpus = len(args.gpu_ids)
    if args.method == "baseline":
        if args.alpha is not None or args.mask_by is not None or args.inpaint_mode is not None:
            parser.error("baseline does not accept PCD algorithm options")
    elif args.method == "pcd-fast" and args.inpaint_mode is not None:
        parser.error("--inpaint-mode is only valid for pcd-mask")
    args.mask_by = args.mask_by or "gt"
    args.inpaint_mode = args.inpaint_mode or "lama"
    return args


def main():
    args = parse_args()
    runner = ParallelRunner(
        policy=args.policy,
        method=args.method,
        checkpoint=args.checkpoint,
        task=args.task,
        result_root=args.result_root,
        n_trajs=args.n_trajs,
        seed=args.seed,
        num_gpus=args.num_gpus,
        gpu_ids=args.gpu_ids,
        alpha=args.alpha,
        mask_by=args.mask_by,
        inpaint_mode=args.inpaint_mode,
    )
    runner.run()


if __name__ == "__main__":
    os.environ["DISPLAY"] = ""
    os.environ["MUJOCO_GL"] = "egl"
    multiprocessing.set_start_method("spawn", force=True)
    main()
