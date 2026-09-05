# Copyright (c) 2026, LeIsaac Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Thin OpenXR input source for Quest Touch controllers."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from typing import Any

import carb
import numpy as np
from isaaclab.devices.openxr import XrCfg
from isaacsim.core.prims import SingleXFormPrim

from .core import OpenXRInputFrame, TrackedPoseSample, read_openxr_input_value


XRCore = None
XRPoseValidityFlags = None
with contextlib.suppress(ModuleNotFoundError):
    from omni.kit.xr.core import XRCore, XRPoseValidityFlags


class Quest3OpenXRSourceV2:
    """Read HMD/controller poses and buttons without performing robot mapping."""

    TELEOP_COMMAND_EVENT_TYPE = "teleop_command"
    HAND_PATHS = {"left": "/user/hand/left", "right": "/user/hand/right"}

    def __init__(
        self,
        *,
        callbacks: dict[str, Callable[[], Any]] | None = None,
        xr_cfg: XrCfg | None = None,
        start_active: bool = True,
    ) -> None:
        if XRCore is None or XRPoseValidityFlags is None:
            raise RuntimeError("Quest V2 requires Isaac Sim OpenXR. Launch teleop with --xr.")
        self._callbacks = dict(callbacks or {})
        self._active = bool(start_active)
        self._epoch = 0
        self._left_x_pressed = False
        self._left_y_pressed = False
        self._xr_core = XRCore.get_singleton()
        self._subscription = self._xr_core.get_message_bus().create_subscription_to_pop_by_type(
            carb.events.type_from_string(self.TELEOP_COMMAND_EVENT_TYPE), self._on_teleop_command
        )
        self._configure_anchor(xr_cfg or XrCfg())

    @property
    def active(self) -> bool:
        return self._active

    @property
    def epoch(self) -> int:
        return self._epoch

    def reset_motion(self) -> None:
        self._epoch += 1

    def poll(self) -> OpenXRInputFrame:
        now = time.perf_counter()
        devices = {
            side: self._xr_core.get_input_device(path) for side, path in self.HAND_PATHS.items()
        }
        self._handle_button_edges(devices["left"])
        hands = {}
        for side, device in devices.items():
            pose = self._get_pose(device, "grip")
            hands[side] = TrackedPoseSample(
                timestamp_s=now,
                pose=pose,
                valid=pose is not None,
                squeeze=read_openxr_input_value(device, "squeeze", "value"),
                trigger=read_openxr_input_value(device, "trigger", "value", "force", "click"),
            )
        head_device = self._xr_core.get_input_device("/user/head")
        head_pose = self._get_pose(head_device, "")
        head = TrackedPoseSample(now, head_pose, head_pose is not None)
        return OpenXRInputFrame(now, head, hands)

    def display_controls(self) -> None:
        print(
            "Quest 3 OpenArm V2 controls:\n"
            "  Left/Right squeeze  Hold to clutch the matching arm\n"
            "  Left/Right trigger  Close/open the matching gripper\n"
            "  Left X              Pause/resume and recapture HMD yaw\n"
            "  Left Y              Reset the environment"
        )

    def close(self) -> None:
        self._subscription = None

    def _configure_anchor(self, xr_cfg: XrCfg) -> None:
        anchor = SingleXFormPrim("/XRAnchor", position=xr_cfg.anchor_pos, orientation=xr_cfg.anchor_rot)
        settings = carb.settings.get_settings()
        settings.set_float("/persistent/xr/profile/ar/render/nearPlane", xr_cfg.near_plane)
        settings.set_string("/persistent/xr/profile/ar/anchorMode", "custom anchor")
        settings.set_string("/xrstage/profile/ar/customAnchor", anchor.prim_path)

    def _handle_button_edges(self, left_controller: Any) -> None:
        x_pressed = read_openxr_input_value(left_controller, "x", "click") >= 0.5
        y_pressed = read_openxr_input_value(left_controller, "y", "click") >= 0.5
        if x_pressed and not self._left_x_pressed:
            if self._active:
                self._stop()
            else:
                self._start()
        if y_pressed and not self._left_y_pressed:
            self.reset_motion()
            self._invoke("RESET")
        self._left_x_pressed = x_pressed
        self._left_y_pressed = y_pressed

    def _start(self) -> None:
        self._active = True
        self.reset_motion()
        self._invoke("START")

    def _stop(self) -> None:
        self._active = False
        self.reset_motion()
        self._invoke("STOP")

    def _on_teleop_command(self, event: carb.events.IEvent) -> None:
        message = str(event.payload.get("message", "")).lower()
        if "start" in message:
            self._start()
        elif "stop" in message:
            self._stop()
        elif "reset" in message:
            self.reset_motion()
            self._invoke("RESET")

    def _invoke(self, key: str) -> None:
        callback = self._callbacks.get(key)
        if callback is not None:
            callback()

    @staticmethod
    def _get_pose(input_device: Any, pose_name: str) -> np.ndarray | None:
        if input_device is None:
            return None
        try:
            description = input_device.get_virtual_world_pose_desc(pose_name)
            required = XRPoseValidityFlags.POSITION_VALID | XRPoseValidityFlags.ORIENTATION_VALID
            if (description.validity_flags & required) != required:
                return None
            matrix = description.pose_matrix
        except Exception:
            try:
                matrix = input_device.get_virtual_world_pose(pose_name)
            except Exception:
                return None
        try:
            position = matrix.ExtractTranslation()
            rotation = matrix.ExtractRotationQuat()
            imaginary = rotation.GetImaginary()
            return np.asarray(
                [position[0], position[1], position[2], rotation.GetReal(), *imaginary], dtype=np.float32
            )
        except Exception:
            return None
