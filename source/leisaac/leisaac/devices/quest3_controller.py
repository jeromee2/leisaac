# Copyright (c) 2026, LeIsaac Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Quest 3 Touch Plus controller interface for relative SE(3) teleoperation."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

import carb
import numpy as np
import torch
from isaaclab.devices.openxr import XrCfg
from isaacsim.core.prims import SingleXFormPrim
from scipy.spatial.transform import Rotation

from .device_base import DeviceBase

# Keep imports optional so normal (non-XR) device imports do not require an XR runtime.
XRCore = None
XRPoseValidityFlags = None
with contextlib.suppress(ModuleNotFoundError):
    from omni.kit.xr.core import XRCore, XRPoseValidityFlags


class Quest3Controller(DeviceBase):
    """Map Quest 3 Touch Plus controller input to a relative SE(3) arm command.

    In single-arm mode the right controller drives the end effector.  In
    bimanual mode each controller drives the matching arm.  Each squeeze input
    is an independent clutch and each trigger controls its matching gripper.
    The left X and Y buttons toggle teleoperation and reset the environment.
    """

    TELEOP_COMMAND_EVENT_TYPE = "teleop_command"
    RIGHT_HAND_PATH = "/user/hand/right"
    LEFT_HAND_PATH = "/user/hand/left"

    def __init__(
        self,
        sim_device: str | torch.device,
        callbacks: dict[str, Callable[[], Any]] | None = None,
        xr_cfg: XrCfg | None = None,
        sensitivity: float = 1.0,
        delta_pos_scale_factor: float = 4.0,
        delta_rot_scale_factor: float = 2.0,
        alpha_pos: float = 0.5,
        alpha_rot: float = 0.35,
        position_threshold: float = 0.001,
        rotation_threshold: float = 0.01,
        clutch_engage_threshold: float = 0.55,
        clutch_release_threshold: float = 0.45,
        gripper_close_threshold: float = 0.65,
        gripper_open_threshold: float = 0.35,
        zero_out_xy_rotation: bool = True,
        start_active: bool = True,
        bimanual: bool = False,
    ):
        """Create the controller interface.

        Args:
            sim_device: PyTorch device for the generated action tensor.
            callbacks: Optional ``START``, ``STOP``, and ``RESET`` callbacks.
            xr_cfg: XR anchor configuration used by the environment.
            sensitivity: Multiplier for both position and rotation command scales.
            delta_pos_scale_factor: Position delta amplification before sensitivity.
            delta_rot_scale_factor: Rotation delta amplification before sensitivity.
            alpha_pos: Position exponential smoothing factor in the range ``(0, 1]``.
            alpha_rot: Rotation exponential smoothing factor in the range ``(0, 1]``.
            position_threshold: Ignore filtered position deltas below this many meters.
            rotation_threshold: Ignore filtered rotation deltas below this many radians.
            clutch_engage_threshold: Right squeeze value required to engage arm motion.
            clutch_release_threshold: Right squeeze value below which arm motion releases.
            gripper_close_threshold: Right trigger value at which to close the gripper.
            gripper_open_threshold: Right trigger value below which to open the gripper.
            zero_out_xy_rotation: Keep only the yaw component for the SO-101's 5-DoF arm.
            start_active: Whether commands are enabled immediately. When false, use Left X or a START command.
            bimanual: Emit left and right 7D commands as one 14D action.
        """
        if XRCore is None or XRPoseValidityFlags is None:
            raise RuntimeError(
                "Quest3Controller requires the Isaac Sim OpenXR extension. Launch with --xr and connect the headset."
            )
        if sensitivity < 0.0:
            raise ValueError("sensitivity must be non-negative")
        if not 0.0 < alpha_pos <= 1.0 or not 0.0 < alpha_rot <= 1.0:
            raise ValueError("alpha_pos and alpha_rot must be in the range (0, 1]")
        if clutch_release_threshold > clutch_engage_threshold:
            raise ValueError("clutch_release_threshold must be less than or equal to clutch_engage_threshold")
        if gripper_open_threshold > gripper_close_threshold:
            raise ValueError("gripper_open_threshold must be less than or equal to gripper_close_threshold")

        self._sim_device = sim_device
        self._additional_callbacks = dict(callbacks or {})
        self._xr_cfg = xr_cfg or XrCfg()

        self._delta_pos_scale_factor = delta_pos_scale_factor * sensitivity
        self._delta_rot_scale_factor = delta_rot_scale_factor * sensitivity
        self._alpha_pos = alpha_pos
        self._alpha_rot = alpha_rot
        self._position_threshold = position_threshold
        self._rotation_threshold = rotation_threshold
        self._clutch_engage_threshold = clutch_engage_threshold
        self._clutch_release_threshold = clutch_release_threshold
        self._gripper_close_threshold = gripper_close_threshold
        self._gripper_open_threshold = gripper_open_threshold
        self._zero_out_xy_rotation = zero_out_xy_rotation
        self._bimanual = bimanual

        self._xr_core = XRCore.get_singleton()
        self._vc_subscription = self._xr_core.get_message_bus().create_subscription_to_pop_by_type(
            carb.events.type_from_string(self.TELEOP_COMMAND_EVENT_TYPE), self._on_teleop_command
        )
        self._configure_xr_anchor()

        self._teleoperation_active = start_active
        self._hand_states = {
            hand_path: {
                "previous_pose": None,
                "smoothed_delta_pos": np.zeros(3, dtype=np.float32),
                "smoothed_delta_rot": np.zeros(3, dtype=np.float32),
                "clutch_active": False,
                "gripper_closed": False,
            }
            for hand_path in (self.LEFT_HAND_PATH, self.RIGHT_HAND_PATH)
        }
        self._left_x_pressed = False
        self._left_y_pressed = False

    def __del__(self):
        """Release the XR message subscription held by this interface."""
        if hasattr(self, "_vc_subscription"):
            self._vc_subscription = None

    def __str__(self) -> str:
        return (
            "Quest 3 Touch Plus Controller\n"
            f"\tPosition scale: {self._delta_pos_scale_factor:g}\n"
            f"\tRotation scale: {self._delta_rot_scale_factor:g}\n"
            f"\t{'Both grip poses' if self._bimanual else 'Right grip pose'} with squeeze clutch\n"
            "\tLeft X: start/stop, Left Y: reset"
        )

    def reset(self):
        """Clear pose, smoothing, and button-edge state without changing teleop state."""
        for state in self._hand_states.values():
            state["previous_pose"] = None
            state["smoothed_delta_pos"].fill(0.0)
            state["smoothed_delta_rot"].fill(0.0)
            state["clutch_active"] = False
            state["gripper_closed"] = False
        self._left_x_pressed = False
        self._left_y_pressed = False

    def add_callback(self, key: str, func: Callable[[], Any]):
        """Register a callback for an XR teleoperation command."""
        self._additional_callbacks[key] = func

    def display_controls(self):
        """Print the physical controller mapping."""
        squeeze_label = "Both squeezes " if self._bimanual else "Right squeeze  "
        grip_label = "Both grips    " if self._bimanual else "Right grip     "
        trigger_label = "Both triggers " if self._bimanual else "Right trigger  "
        print(
            "\nQuest 3 controller controls:\n"
            "  Left X         Pause / resume teleoperation\n"
            "  Left Y         Reset environment\n"
            f"  {squeeze_label} Hold to move the matching robot arm (clutch)\n"
            f"  {grip_label} Move the matching end effector\n"
            f"  {trigger_label} Close / open the matching gripper\n"
            "  Ctrl+C         Quit\n"
        )

    def advance(self) -> torch.Tensor:
        """Return one 7D command, or left+right commands in bimanual mode."""
        right_controller = self._xr_core.get_input_device(self.RIGHT_HAND_PATH)
        left_controller = self._xr_core.get_input_device(self.LEFT_HAND_PATH)
        self._handle_button_edges(left_controller)

        if self._bimanual:
            left_action = self._advance_controller(left_controller, self.LEFT_HAND_PATH)
            right_action = self._advance_controller(right_controller, self.RIGHT_HAND_PATH)
            return torch.cat((left_action, right_action), dim=0)
        return self._advance_controller(right_controller, self.RIGHT_HAND_PATH)

    def _advance_controller(self, controller: Any, hand_path: str) -> torch.Tensor:
        """Advance one controller and return its 7D arm/gripper command."""
        state = self._hand_states[hand_path]
        trigger_value = self._get_input_value(controller, "trigger", "value")
        self._update_gripper_state(trigger_value, state)

        pose = self._get_grip_pose(controller)
        if pose is None:
            state["previous_pose"] = None
            self._clear_motion_state(state)
            return self._make_action(np.zeros(6, dtype=np.float32), state)

        # Always update the baseline while paused.  Starting or engaging the
        # clutch then produces a zero command instead of a pose jump.
        if state["previous_pose"] is None:
            state["previous_pose"] = pose
            return self._make_action(np.zeros(6, dtype=np.float32), state)

        delta_pose = self._calculate_delta_pose(pose, state["previous_pose"])
        state["previous_pose"] = pose

        if not self._teleoperation_active:
            self._clear_motion_state(state)
            return self._make_action(np.zeros(6, dtype=np.float32), state)

        clutch_engaged = self._is_clutch_engaged(
            self._get_input_value(controller, "squeeze", "value"), state
        )
        if not clutch_engaged:
            state["clutch_active"] = False
            self._clear_motion_state(state)
            return self._make_action(np.zeros(6, dtype=np.float32), state)
        if not state["clutch_active"]:
            state["clutch_active"] = True
            self._clear_motion_state(state)
            return self._make_action(np.zeros(6, dtype=np.float32), state)

        return self._make_action(self._filter_and_scale_delta(delta_pose, state), state)

    def _configure_xr_anchor(self):
        """Match the anchor setup used by Isaac Lab's native OpenXR device."""
        xr_anchor = SingleXFormPrim(
            "/XRAnchor", position=self._xr_cfg.anchor_pos, orientation=self._xr_cfg.anchor_rot
        )
        settings = carb.settings.get_settings()
        settings.set_float("/persistent/xr/profile/ar/render/nearPlane", self._xr_cfg.near_plane)
        settings.set_string("/persistent/xr/profile/ar/anchorMode", "custom anchor")
        settings.set_string("/xrstage/profile/ar/customAnchor", xr_anchor.prim_path)

    def _handle_button_edges(self, left_controller: Any):
        x_pressed = self._get_input_value(left_controller, "x", "click") >= 0.5
        y_pressed = self._get_input_value(left_controller, "y", "click") >= 0.5

        if x_pressed and not self._left_x_pressed:
            if self._teleoperation_active:
                self._stop_teleoperation()
            else:
                self._start_teleoperation()
        if y_pressed and not self._left_y_pressed:
            self.reset()
            self._invoke_callback("RESET")

        self._left_x_pressed = x_pressed
        self._left_y_pressed = y_pressed

    def _start_teleoperation(self):
        self._teleoperation_active = True
        for state in self._hand_states.values():
            state["previous_pose"] = None
            self._clear_motion_state(state)
        self._invoke_callback("START")

    def _stop_teleoperation(self):
        self._teleoperation_active = False
        for state in self._hand_states.values():
            state["previous_pose"] = None
            self._clear_motion_state(state)
        self._invoke_callback("STOP")

    def _on_teleop_command(self, event: carb.events.IEvent):
        """Handle the same OpenXR client start, stop, and reset commands as hand tracking."""
        message = str(event.payload.get("message", "")).lower()
        if "start" in message:
            self._start_teleoperation()
        elif "stop" in message:
            self._stop_teleoperation()
        elif "reset" in message:
            self.reset()
            self._invoke_callback("RESET")

    def _invoke_callback(self, key: str):
        callback = self._additional_callbacks.get(key)
        if callback is not None:
            callback()

    def _get_grip_pose(self, input_device: Any) -> np.ndarray | None:
        """Read a valid Quest controller grip pose as ``[x, y, z, qw, qx, qy, qz]``."""
        if input_device is None:
            return None

        try:
            pose_desc = input_device.get_virtual_world_pose_desc("grip")
        except Exception:
            return None

        required_flags = XRPoseValidityFlags.POSITION_VALID | XRPoseValidityFlags.ORIENTATION_VALID
        if (pose_desc.validity_flags & required_flags) != required_flags:
            return None

        pose = pose_desc.pose_matrix
        position = pose.ExtractTranslation()
        rotation = pose.ExtractRotationQuat()
        imaginary = rotation.GetImaginary()
        return np.asarray(
            [
                position[0],
                position[1],
                position[2],
                rotation.GetReal(),
                imaginary[0],
                imaginary[1],
                imaginary[2],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _get_input_value(input_device: Any, input_name: str, gesture_name: str) -> float:
        """Return an OpenXR input value, or zero if it is unavailable."""
        if input_device is None:
            return 0.0
        try:
            if input_device.has_input_gesture(input_name, gesture_name):
                return float(np.clip(input_device.get_input_gesture_value(input_name, gesture_name), 0.0, 1.0))
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _calculate_delta_pose(current_pose: np.ndarray, previous_pose: np.ndarray) -> np.ndarray:
        """Calculate ``[dx, dy, dz, rx, ry, rz]`` from two world-space poses."""
        delta_pos = current_pose[:3] - previous_pose[:3]
        current_rot = Rotation.from_quat([*current_pose[4:7], current_pose[3]])
        previous_rot = Rotation.from_quat([*previous_pose[4:7], previous_pose[3]])
        delta_rot = (current_rot * previous_rot.inv()).as_rotvec()
        return np.concatenate([delta_pos, delta_rot]).astype(np.float32)

    def _filter_and_scale_delta(self, delta_pose: np.ndarray, state: dict[str, Any]) -> np.ndarray:
        position = delta_pose[:3]
        rotation = delta_pose[3:].copy()
        if self._zero_out_xy_rotation:
            rotation[:2] = 0.0

        state["smoothed_delta_pos"] = (
            self._alpha_pos * position + (1.0 - self._alpha_pos) * state["smoothed_delta_pos"]
        )
        if np.linalg.norm(state["smoothed_delta_pos"]) < self._position_threshold:
            state["smoothed_delta_pos"].fill(0.0)

        state["smoothed_delta_rot"] = (
            self._alpha_rot * rotation + (1.0 - self._alpha_rot) * state["smoothed_delta_rot"]
        )
        if np.linalg.norm(state["smoothed_delta_rot"]) < self._rotation_threshold:
            state["smoothed_delta_rot"].fill(0.0)

        return np.concatenate(
            [
                state["smoothed_delta_pos"] * self._delta_pos_scale_factor,
                state["smoothed_delta_rot"] * self._delta_rot_scale_factor,
            ]
        ).astype(np.float32)

    def _is_clutch_engaged(self, squeeze_value: float, state: dict[str, Any]) -> bool:
        if state["clutch_active"]:
            return squeeze_value >= self._clutch_release_threshold
        return squeeze_value >= self._clutch_engage_threshold

    def _update_gripper_state(self, trigger_value: float, state: dict[str, Any]):
        if trigger_value >= self._gripper_close_threshold:
            state["gripper_closed"] = True
        elif trigger_value <= self._gripper_open_threshold:
            state["gripper_closed"] = False

    @staticmethod
    def _clear_motion_state(state: dict[str, Any]):
        state["smoothed_delta_pos"].fill(0.0)
        state["smoothed_delta_rot"].fill(0.0)

    def _make_action(self, ee_command: np.ndarray, state: dict[str, Any]) -> torch.Tensor:
        gripper_command = -1.0 if state["gripper_closed"] else 1.0
        action = np.concatenate([ee_command, np.asarray([gripper_command], dtype=np.float32)])
        return torch.tensor(action, dtype=torch.float32, device=self._sim_device)
