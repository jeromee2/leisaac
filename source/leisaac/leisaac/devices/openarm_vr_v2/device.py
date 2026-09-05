# Copyright (c) 2026, LeIsaac Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Quest 3 to OpenArm joint-command coordinator for teleoperation V2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from leisaac.assets.robots.openarm import OPENARM_BIMANUAL_ARM_JOINT_NAMES

from .core import ClutchedPoseMapper, HandTeleopState, OperatorFrame, rotation_from_wxyz
from .isaac_qp_controller import (
    EXPECTED_JOINT_LIMITS_RAD,
    SIDES,
    OpenArmIsaacQPControllerV2,
)
from .openxr_source import Quest3OpenXRSourceV2


class Quest3OpenArmTeleopV2:
    """Generate direct bimanual OpenArm joint targets from Quest Touch input."""

    def __init__(
        self,
        *,
        robot: Any,
        sim_device: str | torch.device,
        callbacks: dict[str, Any] | None = None,
        xr_cfg: Any = None,
        start_active: bool = True,
        control_hz: float = 60.0,
        ee_body_names: dict[str, str] | None = None,
        position_scale: float = 1.0,
        rotation_scale: float = 1.0,
        max_linear_speed: float = 1.0,
        max_angular_speed: float = 4.5,
        joint_target_lookahead_s: float = 0.08,
        debug_log_path: str | None = None,
    ) -> None:
        self._robot = robot
        self._sim_device = sim_device
        self._control_hz = float(control_hz)
        self._ee_body_names = ee_body_names
        self._max_linear_speed = float(max_linear_speed)
        self._max_angular_speed = float(max_angular_speed)
        self._joint_target_lookahead_s = float(joint_target_lookahead_s)
        if self._robot.data.joint_pos.shape[0] != 1:
            raise ValueError("Quest V2 supports exactly one Isaac environment.")
        self._joint_ids: dict[str, list[int]] = {}
        for side in SIDES:
            ids, names = robot.find_joints(list(OPENARM_BIMANUAL_ARM_JOINT_NAMES[side]), preserve_order=True)
            if len(ids) != 7:
                raise RuntimeError(f"Expected seven {side} OpenArm joints, found {names}.")
            self._joint_ids[side] = list(ids)
        self._validate_robot_schema()

        self._source = Quest3OpenXRSourceV2(
            callbacks=callbacks,
            xr_cfg=xr_cfg,
            start_active=start_active,
        )
        self._mappers = {
            side: ClutchedPoseMapper(
                position_scale=position_scale,
                rotation_scale=rotation_scale,
                max_linear_speed=max_linear_speed,
                max_angular_speed=max_angular_speed,
            )
            for side in SIDES
        }
        self._solver: OpenArmIsaacQPControllerV2 | None = None
        self._last_command: dict[str, np.ndarray] = {}
        self._divergence_frames = {side: 0 for side in SIDES}
        self._operator_frame: OperatorFrame | None = None
        self._previous_head_pose: np.ndarray | None = None
        self._previous_head_time: float | None = None
        self._seen_epoch = self._source.epoch
        self._debug_stream = None
        if debug_log_path:
            path = Path(debug_log_path).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            self._debug_stream = path.open("a", encoding="utf-8", buffering=1)

    def reset(self) -> None:
        """Synchronize native IK to Isaac and invalidate every clutch/reference."""
        measured = self._read_measured()
        if self._solver is None:
            self._solver = OpenArmIsaacQPControllerV2(
                self._robot,
                measured["left"],
                measured["right"],
                control_hz=self._control_hz,
                ee_body_names=self._ee_body_names,
                max_linear_speed=self._max_linear_speed,
                max_angular_speed=self._max_angular_speed,
                joint_target_lookahead_s=self._joint_target_lookahead_s,
            )
        else:
            self._solver.sync(measured["left"], measured["right"])
        self._last_command = {side: measured[side].copy() for side in SIDES}
        self._divergence_frames = {side: 0 for side in SIDES}
        self._operator_frame = None
        self._previous_head_pose = None
        self._previous_head_time = None
        for mapper in self._mappers.values():
            mapper.reset()
        self._source.reset_motion()
        self._seen_epoch = self._source.epoch
        self._log({"event": "reset"})

    def advance(self) -> torch.Tensor:
        """Poll XR, solve both arms independently, and return one 16D Isaac action."""
        frame = self._source.poll()
        measured = self._read_measured()
        self._ensure_solver(measured)
        assert self._solver is not None

        if self._seen_epoch != self._source.epoch:
            self._seen_epoch = self._source.epoch
            self._operator_frame = None
            self._previous_head_pose = None
            self._previous_head_time = None
            self._solver.sync(measured["left"], measured["right"])
            self._last_command = {side: measured[side].copy() for side in SIDES}
            for mapper in self._mappers.values():
                mapper.reset()

        self._update_operator_frame(frame.head, frame.timestamp_s, measured)
        log_hands: dict[str, Any] = {}
        for side in SIDES:
            mapper = self._mappers[side]
            robot_pose = self._solver.fk(side)
            result = mapper.step(
                frame.hands[side],
                now_s=frame.timestamp_s,
                operator_frame=self._operator_frame,
                robot_pose=robot_pose,
                enabled=self._source.active,
            )
            solver_ok = None
            if result.target_pose is not None:
                if result.clutch_started:
                    self._solver.sync(measured["left"], measured["right"], side=side)
                command = self._solver.solve(
                    side,
                    result.target_pose,
                    measured["left"],
                    measured["right"],
                )
                solver_ok = command is not None
                if command is None:
                    mapper.mark_hold(frame.timestamp_s, fault=True)
                else:
                    self._last_command[side] = command

            if mapper.state == HandTeleopState.CLUTCHED:
                divergence = float(np.max(np.abs(self._last_command[side] - measured[side])))
                self._divergence_frames[side] = self._divergence_frames[side] + 1 if divergence > 0.25 else 0
                if self._divergence_frames[side] >= 6:
                    mapper.mark_hold(frame.timestamp_s, fault=True)
                    self._last_command[side] = measured[side].copy()
                    self._solver.sync(measured["left"], measured["right"], side=side)
            else:
                self._divergence_frames[side] = 0

            log_hands[side] = {
                "state": mapper.state.value,
                "reason": result.reason,
                "trigger": frame.hands[side].trigger,
                "gripper_closed": mapper.gripper_closed,
                "solver_ok": solver_ok,
                "raw_pose": frame.hands[side].pose.tolist() if frame.hands[side].pose is not None else None,
                "target_pose": result.target_pose.tolist() if result.target_pose is not None else None,
                "measured_q": measured[side].tolist(),
                "command_q": self._last_command[side].tolist(),
            }

        action = np.concatenate((
            self._last_command["left"],
            [-1.0 if self._mappers["left"].gripper_closed else 1.0],
            self._last_command["right"],
            [-1.0 if self._mappers["right"].gripper_closed else 1.0],
        )).astype(np.float32)
        self._log({
            "event": "frame",
            "timestamp_s": frame.timestamp_s,
            "active": self._source.active,
            "operator_calibrated": self._operator_frame is not None,
            "hands": log_hands,
        })
        return torch.as_tensor(action, dtype=torch.float32, device=self._sim_device)

    def display_controls(self) -> None:
        self._source.display_controls()

    def close(self) -> None:
        self._source.close()
        if self._debug_stream is not None:
            self._debug_stream.close()
            self._debug_stream = None

    def _ensure_solver(self, measured: dict[str, np.ndarray]) -> None:
        if self._solver is None:
            self._solver = OpenArmIsaacQPControllerV2(
                self._robot,
                measured["left"],
                measured["right"],
                control_hz=self._control_hz,
                ee_body_names=self._ee_body_names,
                max_linear_speed=self._max_linear_speed,
                max_angular_speed=self._max_angular_speed,
                joint_target_lookahead_s=self._joint_target_lookahead_s,
            )
            self._last_command = {side: measured[side].copy() for side in SIDES}

    def _read_measured(self) -> dict[str, np.ndarray]:
        return {
            side: self._robot.data.joint_pos[0, self._joint_ids[side]].detach().cpu().numpy().astype(np.float32)
            for side in SIDES
        }

    def _validate_robot_schema(self) -> None:
        for side in SIDES:
            actual_names = [self._robot.data.joint_names[index] for index in self._joint_ids[side]]
            expected_names = list(OPENARM_BIMANUAL_ARM_JOINT_NAMES[side])
            if actual_names != expected_names:
                raise RuntimeError(
                    f"OpenArm {side} joint order mismatch: expected {expected_names}, got {actual_names}."
                )
            actual_limits = self._robot.data.soft_joint_pos_limits[0, self._joint_ids[side]].detach().cpu().numpy()
            if not np.allclose(actual_limits, EXPECTED_JOINT_LIMITS_RAD[side], atol=1e-4, rtol=0.0):
                raise RuntimeError(
                    f"OpenArm {side} limits do not match the verified V2 schema: {actual_limits.tolist()}."
                )

    def _update_operator_frame(self, head: Any, now_s: float, measured: dict[str, np.ndarray]) -> None:
        if not self._source.active:
            return
        if not head.valid or head.pose is None or now_s - head.timestamp_s > 0.1:
            self._operator_frame = None
            for mapper in self._mappers.values():
                mapper.mark_hold(now_s)
            return

        discontinuity = False
        if self._previous_head_pose is not None and self._previous_head_time is not None:
            dt = head.timestamp_s - self._previous_head_time
            if 0.0 < dt <= 0.1:
                translation = float(np.linalg.norm(head.pose[:3] - self._previous_head_pose[:3]))
                rotation = float(
                    (
                        rotation_from_wxyz(head.pose[3:]) * rotation_from_wxyz(self._previous_head_pose[3:]).inv()
                    ).magnitude()
                )
                discontinuity = translation > 0.25 or rotation > 0.5
        self._previous_head_pose = head.pose.copy()
        self._previous_head_time = head.timestamp_s

        if self._operator_frame is None or discontinuity:
            try:
                self._operator_frame = OperatorFrame.from_head_pose(head.pose)
            except ValueError:
                self._operator_frame = None
                return
            if discontinuity:
                assert self._solver is not None
                self._solver.sync(measured["left"], measured["right"])
                self._last_command = {side: measured[side].copy() for side in SIDES}
                for mapper in self._mappers.values():
                    mapper.mark_hold(now_s)
            print("[Quest V2] Captured fixed HMD yaw calibration.")

    def _log(self, record: dict[str, Any]) -> None:
        if self._debug_stream is not None:
            self._debug_stream.write(json.dumps(record, separators=(",", ":")) + "\n")
