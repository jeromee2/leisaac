# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a leisaac teleoperation with leisaac manipulation environments."""

"""Launch Isaac Sim Simulator first."""
import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)
import argparse
import signal

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="leisaac teleoperation for leisaac environments."
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to simulate."
)
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    choices=[
        "keyboard",
        "gamepad",
        "so101leader",
        "bi-so101leader",
        "lekiwi-keyboard",
        "lekiwi-gamepad",
        "lekiwi-leader",
        "handtracking",
        "quest3-controller",
        "quest3-controller-v2",
    ],
    help="Device for interacting with environment",
)
parser.add_argument(
    "--port",
    type=str,
    default="/dev/ttyACM0",
    help="Port for the teleop device:so101leader, default is /dev/ttyACM0",
)
parser.add_argument(
    "--remote_endpoint",
    type=str,
    default=None,
    help=(
        "ZMQ endpoint for remote so101leader (e.g. tcp://192.168.1.10:5556). Uses so101_joint_state_server.py on the"
        " remote machine."
    ),
)
parser.add_argument(
    "--left_arm_port",
    type=str,
    default="/dev/ttyACM0",
    help="Port for the left teleop device:bi-so101leader, default is /dev/ttyACM0",
)
parser.add_argument(
    "--right_arm_port",
    type=str,
    default="/dev/ttyACM1",
    help="Port for the right teleop device:bi-so101leader, default is /dev/ttyACM1",
)
parser.add_argument(
    "--task",
    type=str,
    default="LeIsaac-OpenArm-Bimanual-Physics01-QuestV2-v0",
    help="Name of the task (default: Physics01 OpenArm Quest V2).",
)
parser.add_argument("--seed", type=int, default=None, help="Seed for the environment.")
parser.add_argument(
    "--sensitivity", type=float, default=1.0, help="Sensitivity factor."
)
parser.add_argument(
    "--openarm_bimanual_pos_scale",
    type=float,
    default=1.0,
    help="Quest position gain for absolute OpenArm bimanual tracking.",
)
parser.add_argument(
    "--openarm_bimanual_rot_scale",
    type=float,
    default=1.0,
    help="Quest rotation gain for absolute OpenArm bimanual tracking.",
)
parser.add_argument(
    "--openarm_bimanual_rot_deadzone",
    type=float,
    default=0.0001,
    help="Quest rotation deadzone in radians for the OpenArm bimanual task.",
)
parser.add_argument(
    "--openarm_bimanual_max_linear_speed",
    type=float,
    default=1.0,
    help="Maximum OpenArm absolute-target translation speed in meters per second.",
)
parser.add_argument(
    "--openarm_bimanual_max_angular_speed",
    type=float,
    default=6.0,
    help="Maximum OpenArm absolute-target rotation speed in radians per second.",
)
parser.add_argument(
    "--openarm_bimanual_elbow",
    type=float,
    default=1.0,
    help="Initial joint4 angle in radians for both OpenArm bimanual arms.",
)
parser.add_argument(
    "--openarm_v2_pos_scale",
    type=float,
    default=1.0,
    help="Quest V2 relative translation scale.",
)
parser.add_argument(
    "--openarm_v2_rot_scale",
    type=float,
    default=1.0,
    help="Quest V2 relative rotation scale.",
)
parser.add_argument(
    "--openarm_v2_max_linear_speed",
    type=float,
    default=1.0,
    help="Quest V2 maximum filtered target speed in meters per second.",
)
parser.add_argument(
    "--openarm_v2_max_angular_speed",
    type=float,
    default=4.5,
    help="Quest V2 maximum filtered target angular speed in radians per second.",
)
parser.add_argument(
    "--openarm_v2_joint_target_lookahead_s",
    type=float,
    default=0.08,
    help="Quest V2 joint target lookahead in seconds (0 < value <= 0.09); larger values track faster.",
)
parser.add_argument(
    "--openarm_v2_debug_log",
    type=str,
    default=None,
    help="Optional JSONL path for Quest V2 targets, joints, and state transitions.",
)
parser.add_argument(
    "--xr_start_paused",
    action="store_true",
    help="Start XR teleoperation paused until a START command or the Quest 3 Left X button resumes it.",
)

# recorder_parameter
parser.add_argument(
    "--record", action="store_true", help="whether to enable record function"
)
parser.add_argument(
    "--step_hz", type=int, default=60, help="Environment stepping rate in Hz."
)
parser.add_argument(
    "--dataset_file",
    type=str,
    default="./datasets/dataset.hdf5",
    help="File path to export recorded demos.",
)
parser.add_argument(
    "--resume",
    action="store_true",
    help="whether to resume recording in the existing dataset file",
)
parser.add_argument(
    "--num_demos",
    type=int,
    default=0,
    help="Number of demonstrations to record. Set to 0 for infinite.",
)

parser.add_argument(
    "--recalibrate",
    action="store_true",
    help="recalibrate SO101-Leader or Bi-SO101Leader",
)
parser.add_argument(
    "--quality", action="store_true", help="whether to enable quality render mode."
)
parser.add_argument(
    "--use_lerobot_recorder",
    action="store_true",
    help="whether to use lerobot recorder.",
)
parser.add_argument(
    "--lerobot_dataset_repo_id",
    type=str,
    default=None,
    help="Lerobot Dataset repository ID.",
)
parser.add_argument(
    "--lerobot_dataset_fps",
    type=int,
    default=30,
    help="Lerobot Dataset frames per second.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)
if args_cli.teleop_device in {
    "handtracking",
    "quest3-controller",
    "quest3-controller-v2",
}:
    app_launcher_args["xr"] = True

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import os
import time

import gymnasium as gym
import torch
from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import DatasetExportMode, TerminationTermCfg
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.enhance.managers import EnhanceDatasetExportMode, StreamingRecorderManager
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        """
        Args:
            hz (int): frequency to enforce
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def manual_terminate(env: ManagerBasedRLEnv | DirectRLEnv, success: bool):
    if hasattr(env, "termination_manager"):
        if success:
            env.termination_manager.set_term_cfg(
                "success",
                TerminationTermCfg(
                    func=lambda env: torch.ones(
                        env.num_envs, dtype=torch.bool, device=env.device
                    )
                ),
            )
        else:
            env.termination_manager.set_term_cfg(
                "success",
                TerminationTermCfg(
                    func=lambda env: torch.zeros(
                        env.num_envs, dtype=torch.bool, device=env.device
                    )
                ),
            )
        env.termination_manager.compute()
    elif hasattr(env, "_get_dones"):
        env.cfg.return_success_status = success


def main():  # noqa: C901
    """Running lerobot teleoperation with leisaac manipulation environment."""

    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    env_cfg.use_teleop_device(args_cli.teleop_device)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())
    task_name = args_cli.task
    is_openarm_bimanual = getattr(env_cfg, "robot_name", "").startswith(
        "openarm_bimanual"
    )
    if is_openarm_bimanual:
        if not 0.0 <= args_cli.openarm_bimanual_elbow <= 2.443461:
            raise ValueError(
                "--openarm_bimanual_elbow must be in [0.0, 2.443461] radians"
            )
        env_cfg.scene.robot.init_state.joint_pos["openarm_left_joint4"] = (
            args_cli.openarm_bimanual_elbow
        )
        env_cfg.scene.robot.init_state.joint_pos["openarm_right_joint4"] = (
            args_cli.openarm_bimanual_elbow
        )

    if args_cli.xr:
        env_cfg = remove_camera_configs(env_cfg)
        for event_name in dir(env_cfg.events):
            event_cfg = getattr(env_cfg.events, event_name)
            asset_cfg = (
                getattr(event_cfg, "params", {}).get("asset_cfg") if event_cfg else None
            )
            if asset_cfg and not hasattr(env_cfg.scene, asset_cfg.name):
                setattr(env_cfg.events, event_name, None)

    if args_cli.quality:
        env_cfg.sim.render.antialiasing_mode = "FXAA"
        env_cfg.sim.render.rendering_mode = "quality"

    # precheck task and teleop device
    if "BiArm" in task_name or "Bimanual" in task_name:
        allowed_bimanual_devices = (
            {"handtracking", "quest3-controller", "quest3-controller-v2"}
            if is_openarm_bimanual
            else {"bi-so101leader"}
        )
        assert (
            args_cli.teleop_device in allowed_bimanual_devices
        ), f"supported devices for this bimanual task: {sorted(allowed_bimanual_devices)}"
    if "LeKiwi" in task_name:
        assert args_cli.teleop_device in [
            "lekiwi-leader",
            "lekiwi-keyboard",
            "lekiwi-gamepad",
        ], "only support lekiwi-leader, lekiwi-keyboard, lekiwi-gamepad for lekiwi task"
    is_direct_env = "Direct" in task_name
    if is_direct_env:
        assert args_cli.teleop_device in [
            "so101leader",
            "bi-so101leader",
        ], "only support so101leader or bi-so101leader for direct task"

    # timeout and terminate preprocess
    if is_direct_env:
        env_cfg.never_time_out = True
        env_cfg.manual_terminate = True
    else:
        # modify configuration
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
        if hasattr(env_cfg.terminations, "success"):
            env_cfg.terminations.success = None
    # recorder preprocess & manual success terminate preprocess
    if args_cli.record:
        if args_cli.use_lerobot_recorder:
            if args_cli.resume:
                env_cfg.recorders.dataset_export_mode = (
                    EnhanceDatasetExportMode.EXPORT_SUCCEEDED_ONLY_RESUME
                )
            else:
                env_cfg.recorders.dataset_export_mode = (
                    DatasetExportMode.EXPORT_SUCCEEDED_ONLY
                )
        else:
            if args_cli.resume:
                env_cfg.recorders.dataset_export_mode = (
                    EnhanceDatasetExportMode.EXPORT_ALL_RESUME
                )
                assert os.path.exists(
                    args_cli.dataset_file
                ), "the dataset file does not exist, please don't use '--resume' if you want to record a new dataset"
            else:
                env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
                assert not os.path.exists(
                    args_cli.dataset_file
                ), "the dataset file already exists, please use '--resume' to resume recording"
        env_cfg.recorders.dataset_export_dir_path = output_dir
        env_cfg.recorders.dataset_filename = output_file_name
        if is_direct_env:
            env_cfg.return_success_status = False
        else:
            if not hasattr(env_cfg.terminations, "success"):
                setattr(env_cfg.terminations, "success", None)
            env_cfg.terminations.success = TerminationTermCfg(
                func=lambda env: torch.zeros(
                    env.num_envs, dtype=torch.bool, device=env.device
                )
            )
    else:
        env_cfg.recorders = None

    # create environment
    env: ManagerBasedRLEnv | DirectRLEnv = gym.make(task_name, cfg=env_cfg).unwrapped
    # replace the original recorder manager with the streaming recorder manager or lerobot recorder manager
    if args_cli.record:
        del env.recorder_manager
        if args_cli.use_lerobot_recorder:
            from leisaac.enhance.datasets.lerobot_dataset_handler import (
                LeRobotDatasetCfg,
            )
            from leisaac.enhance.managers.lerobot_recorder_manager import (
                LeRobotRecorderManager,
            )

            dataset_cfg = LeRobotDatasetCfg(
                repo_id=args_cli.lerobot_dataset_repo_id,
                fps=args_cli.lerobot_dataset_fps,
            )
            env.recorder_manager = LeRobotRecorderManager(
                env_cfg.recorders, dataset_cfg, env
            )
        else:
            env.recorder_manager = StreamingRecorderManager(env_cfg.recorders, env)
            env.recorder_manager.flush_steps = 100
            env.recorder_manager.compression = "lzf"

    # add teleoperation key for env reset
    should_reset_recording_instance = False

    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True

    # add teleoperation key for task success
    should_reset_task_success = False

    def reset_task_success():
        nonlocal should_reset_task_success
        should_reset_task_success = True
        reset_recording_instance()

    xr_teleop_devices = {"handtracking", "quest3-controller", "quest3-controller-v2"}
    # The CloudXR client does not always emit the optional ``teleop_command``
    # START event. Start XR devices by default; their first valid pose is
    # still used as a zero-motion baseline. Use --xr_start_paused to wait for
    # an explicit start command instead.
    teleoperation_active = not (
        args_cli.teleop_device in xr_teleop_devices and args_cli.xr_start_paused
    )

    def start_teleoperation():
        nonlocal teleoperation_active
        teleoperation_active = True

    def stop_teleoperation():
        nonlocal teleoperation_active
        teleoperation_active = False

    # create controller
    if args_cli.teleop_device == "handtracking":
        print(
            "Quest 3 hand tracking is active. Put the Touch Plus controllers down and use bare hands; "
            "hand joints are unavailable while OpenXR binds the controller profile."
        )
        teleop_callbacks = {
            "START": start_teleoperation,
            "STOP": stop_teleoperation,
            "RESET": reset_recording_instance,
        }
        teleop_interface = create_teleop_device(
            args_cli.teleop_device, env_cfg.teleop_devices.devices, teleop_callbacks
        )
    elif args_cli.teleop_device == "quest3-controller":
        from leisaac.devices import Quest3Controller

        teleop_callbacks = {
            "START": start_teleoperation,
            "STOP": stop_teleoperation,
            "RESET": reset_recording_instance,
        }
        initial_target_poses = None
        if is_openarm_bimanual:
            from isaaclab.utils.math import subtract_frame_transforms
            from leisaac.assets.robots.openarm import OPENARM_BIMANUAL_EE_BODY_NAMES

            robot = env.scene["robot"]
            initial_target_poses = {}
            for side, body_name in OPENARM_BIMANUAL_EE_BODY_NAMES.items():
                body_id = robot.data.body_names.index(body_name)
                position, quaternion = subtract_frame_transforms(
                    robot.data.root_pos_w[:1],
                    robot.data.root_quat_w[:1],
                    robot.data.body_pos_w[:1, body_id],
                    robot.data.body_quat_w[:1, body_id],
                )
                initial_target_poses[side] = (
                    torch.cat((position[0], quaternion[0])).cpu().numpy()
                )

        teleop_interface = Quest3Controller(
            sim_device=env.device,
            callbacks=teleop_callbacks,
            xr_cfg=env_cfg.xr,
            sensitivity=args_cli.sensitivity,
            delta_pos_scale_factor=(
                args_cli.openarm_bimanual_pos_scale if is_openarm_bimanual else 4.0
            ),
            delta_rot_scale_factor=(
                args_cli.openarm_bimanual_rot_scale if is_openarm_bimanual else 2.0
            ),
            rotation_threshold=(
                args_cli.openarm_bimanual_rot_deadzone if is_openarm_bimanual else 0.01
            ),
            position_threshold=0.0001 if is_openarm_bimanual else 0.001,
            zero_out_xy_rotation=not getattr(env_cfg, "robot_name", "").startswith(
                "openarm"
            ),
            start_active=teleoperation_active,
            bimanual=is_openarm_bimanual,
            initial_target_poses=initial_target_poses,
            max_linear_speed=args_cli.openarm_bimanual_max_linear_speed,
            max_angular_speed=args_cli.openarm_bimanual_max_angular_speed,
        )
    elif args_cli.teleop_device == "quest3-controller-v2":
        if not is_openarm_bimanual:
            raise ValueError("quest3-controller-v2 requires an OpenArm bimanual task.")
        from leisaac.devices import Quest3OpenArmTeleopV2

        teleop_callbacks = {
            "START": start_teleoperation,
            "STOP": stop_teleoperation,
            "RESET": reset_recording_instance,
        }
        teleop_interface = Quest3OpenArmTeleopV2(
            robot=env.scene["robot"],
            sim_device=env.device,
            callbacks=teleop_callbacks,
            xr_cfg=env_cfg.xr,
            start_active=teleoperation_active,
            control_hz=args_cli.step_hz,
            position_scale=args_cli.openarm_v2_pos_scale,
            rotation_scale=args_cli.openarm_v2_rot_scale,
            max_linear_speed=args_cli.openarm_v2_max_linear_speed,
            max_angular_speed=args_cli.openarm_v2_max_angular_speed,
            joint_target_lookahead_s=args_cli.openarm_v2_joint_target_lookahead_s,
            ee_body_names=getattr(env_cfg, "openarm_ee_body_names", None),
            debug_log_path=args_cli.openarm_v2_debug_log,
        )
    elif args_cli.teleop_device == "keyboard":
        from leisaac.devices import SO101Keyboard

        teleop_interface = SO101Keyboard(env, sensitivity=args_cli.sensitivity)
    elif args_cli.teleop_device == "gamepad":
        from leisaac.devices import SO101Gamepad

        teleop_interface = SO101Gamepad(env, sensitivity=args_cli.sensitivity)
    elif args_cli.teleop_device == "so101leader":
        if args_cli.remote_endpoint:
            from leisaac.devices import SO101LeaderRemote

            teleop_interface = SO101LeaderRemote(env, endpoint=args_cli.remote_endpoint)
        else:
            from leisaac.devices import SO101Leader

            teleop_interface = SO101Leader(
                env, port=args_cli.port, recalibrate=args_cli.recalibrate
            )
    elif args_cli.teleop_device == "bi-so101leader":
        from leisaac.devices import BiSO101Leader

        teleop_interface = BiSO101Leader(
            env,
            left_port=args_cli.left_arm_port,
            right_port=args_cli.right_arm_port,
            recalibrate=args_cli.recalibrate,
        )
    elif args_cli.teleop_device == "lekiwi-keyboard":
        from leisaac.devices import LeKiwiKeyboard

        teleop_interface = LeKiwiKeyboard(env, sensitivity=args_cli.sensitivity)
    elif args_cli.teleop_device == "lekiwi-leader":
        from leisaac.devices import LeKiwiLeader

        teleop_interface = LeKiwiLeader(
            env, port=args_cli.port, recalibrate=args_cli.recalibrate
        )
    elif args_cli.teleop_device == "lekiwi-gamepad":
        from leisaac.devices import LeKiwiGamepad

        teleop_interface = LeKiwiGamepad(env, sensitivity=args_cli.sensitivity)
    else:
        raise ValueError(
            f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'gamepad', 'so101leader',"
            " 'bi-so101leader', 'lekiwi-keyboard', 'lekiwi-leader', 'lekiwi-gamepad', 'handtracking',"
            " 'quest3-controller', 'quest3-controller-v2'."
        )

    if args_cli.teleop_device not in xr_teleop_devices:
        teleop_interface.add_callback("R", reset_recording_instance)
        teleop_interface.add_callback("N", reset_task_success)
    if hasattr(teleop_interface, "display_controls"):
        teleop_interface.display_controls()

    hand_tracking_required_joints = {"wrist", "thumb_tip", "index_tip"}
    hand_tracking_required_pose_flags = None
    hand_tracking_primed = args_cli.teleop_device != "handtracking"
    hand_tracking_waiting_reported = False
    xr_core = None
    if args_cli.teleop_device == "handtracking":
        from omni.kit.xr.core import XRCore, XRPoseValidityFlags

        xr_core = XRCore.get_singleton()
        hand_tracking_required_pose_flags = (
            XRPoseValidityFlags.POSITION_VALID | XRPoseValidityFlags.ORIENTATION_VALID
        )

    def hand_tracking_is_ready() -> bool:
        """Return whether the active OpenXR profile exposes the joints used by the retargeters."""
        if xr_core is None:
            return False
        try:
            hand_paths = ["/user/hand/right"]
            if is_openarm_bimanual:
                hand_paths.insert(0, "/user/hand/left")
            for hand_path in hand_paths:
                hand = xr_core.get_input_device(hand_path)
                if hand is None:
                    return False
                poses = hand.get_all_virtual_world_poses()
                if not all(
                    joint_name in poses
                    and (
                        poses[joint_name].validity_flags
                        & hand_tracking_required_pose_flags
                    )
                    == hand_tracking_required_pose_flags
                    for joint_name in hand_tracking_required_joints
                ):
                    return False
            return True
        except Exception:
            return False

    rate_limiter = RateLimiter(args_cli.step_hz)

    # reset environment
    if hasattr(env, "initialize"):
        env.initialize()
    env.reset()
    teleop_interface.reset()

    resume_recorded_demo_count = 0
    if args_cli.record and args_cli.resume:
        resume_recorded_demo_count = (
            env.recorder_manager._dataset_file_handler.get_num_episodes()
        )
        print(
            f"Resume recording from existing dataset file with {resume_recorded_demo_count} demonstrations."
        )
    current_recorded_demo_count = resume_recorded_demo_count

    start_record_state = False

    interrupted = False

    def signal_handler(signum, frame):
        """Handle SIGINT (Ctrl+C) signal."""
        nonlocal interrupted
        interrupted = True
        print("\n[INFO] KeyboardInterrupt (Ctrl+C) detected. Cleaning up resources...")

    original_sigint_handler = signal.signal(signal.SIGINT, signal_handler)

    try:
        while simulation_app.is_running() and not interrupted:
            # run everything in inference mode
            with torch.inference_mode():
                if env.cfg.dynamic_reset_gripper_effort_limit:
                    dynamic_reset_gripper_effort_limit_sim(env, args_cli.teleop_device)
                if (
                    args_cli.teleop_device == "handtracking"
                    and not hand_tracking_is_ready()
                ):
                    if not hand_tracking_waiting_reported:
                        print(
                            "Waiting for Quest 3 hand joints (wrist, thumb_tip, index_tip). "
                            "Put both controllers down or turn them off, then show "
                            f"{'both hands' if is_openarm_bimanual else 'your right hand'} to the headset."
                        )
                        hand_tracking_waiting_reported = True
                    hand_tracking_primed = False
                    actions = None
                elif (
                    args_cli.teleop_device == "handtracking"
                    and not hand_tracking_primed
                ):
                    # Discard one valid frame after an input-profile change so the
                    # retargeters learn the current hand pose without a jump.
                    teleop_interface.reset()
                    teleop_interface.advance()
                    hand_tracking_primed = True
                    hand_tracking_waiting_reported = False
                    print(
                        "Quest 3 hand joints detected; captured a zero-motion baseline."
                    )
                    actions = None
                else:
                    actions = teleop_interface.advance()
                if isinstance(actions, torch.Tensor) and actions.ndim == 1:
                    actions = actions.repeat(env.num_envs, 1)
                if should_reset_task_success:
                    print("Task Success!!!")
                    should_reset_task_success = False
                    if args_cli.record:
                        manual_terminate(env, True)
                if should_reset_recording_instance:
                    env.reset()
                    if args_cli.teleop_device == "quest3-controller-v2":
                        teleop_interface.reset()
                    should_reset_recording_instance = False
                    if start_record_state:
                        if args_cli.record:
                            print("Stop Recording!!!")
                        start_record_state = False
                    if args_cli.record:
                        manual_terminate(env, False)
                    # print out the current demo count if it has changed
                    if (
                        args_cli.record
                        and env.recorder_manager.exported_successful_episode_count
                        + resume_recorded_demo_count
                        > current_recorded_demo_count
                    ):
                        current_recorded_demo_count = (
                            env.recorder_manager.exported_successful_episode_count
                            + resume_recorded_demo_count
                        )
                        print(
                            f"Recorded {current_recorded_demo_count} successful demonstrations."
                        )
                    if (
                        args_cli.record
                        and args_cli.num_demos > 0
                        and env.recorder_manager.exported_successful_episode_count
                        + resume_recorded_demo_count
                        >= args_cli.num_demos
                    ):
                        print(
                            f"All {args_cli.num_demos} demonstrations recorded. Exiting the app."
                        )
                        break

                elif not teleoperation_active or actions is None:
                    env.render()
                # apply actions
                else:
                    if not start_record_state:
                        if args_cli.record:
                            print("Start Recording!!!")
                        start_record_state = True
                    env.step(actions)
                if rate_limiter:
                    rate_limiter.sleep(env)
            if interrupted:
                break
    except Exception as e:
        import traceback

        print(f"\n[ERROR] An error occurred: {e}\n")
        traceback.print_exc()
        print("[INFO] Cleaning up resources...")
    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, original_sigint_handler)
        # finalize the recorder manager
        if args_cli.record and hasattr(env.recorder_manager, "finalize"):
            env.recorder_manager.finalize()
        if hasattr(teleop_interface, "close"):
            teleop_interface.close()
        # close the simulator
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    # run the main function
    main()
