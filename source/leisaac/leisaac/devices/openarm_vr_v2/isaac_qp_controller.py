# Copyright (c) 2026, LeIsaac Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Independent constrained IK using the live Isaac articulation and Jacobians."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from leisaac.assets.robots.openarm import (
    OPENARM_BIMANUAL_ARM_JOINT_NAMES,
    OPENARM_BIMANUAL_EE_BODY_NAMES,
)
from qpsolvers import solve_qp

from .core import rotation_from_wxyz, rotation_to_wxyz, validate_pose

SIDES = ("left", "right")
JOINT_VELOCITY_LIMITS_RAD_S = np.asarray([2.0, 2.0, 2.175, 2.175, 2.61, 2.61, 2.61])
DEFAULT_JOINT_TARGET_LOOKAHEAD_S = 0.08
EXPECTED_JOINT_LIMITS_RAD = {
    "left": np.asarray([
        [-3.490659, 1.396263],
        [-3.316126, 0.174533],
        [-1.570796, 1.570796],
        [0.0, 2.443461],
        [-1.570796, 1.570796],
        [-0.785398, 0.785398],
        [-1.570796, 1.570796],
    ]),
    "right": np.asarray([
        [-1.396263, 3.490659],
        [-0.174533, 3.316126],
        [-1.570796, 1.570796],
        [0.0, 2.443461],
        [-1.570796, 1.570796],
        [-0.785398, 0.785398],
        [-1.570796, 1.570796],
    ]),
}


class OpenArmIsaacQPControllerV2:
    """Solve each arm from Isaac's current FK/Jacobian with bounded joint velocity."""

    def __init__(
        self,
        robot: Any,
        initial_left: np.ndarray,
        initial_right: np.ndarray,
        *,
        control_hz: float = 60.0,
        ee_body_names: dict[str, str] | None = None,
        max_linear_speed: float = 1.0,
        max_angular_speed: float = 4.5,
        joint_target_lookahead_s: float = DEFAULT_JOINT_TARGET_LOOKAHEAD_S,
    ) -> None:
        if not np.isfinite(control_hz) or control_hz <= 0.0:
            raise ValueError("control_hz must be finite and positive.")
        if not np.isfinite(max_linear_speed) or max_linear_speed < 0.0:
            raise ValueError("max_linear_speed must be finite and non-negative.")
        if not np.isfinite(max_angular_speed) or max_angular_speed < 0.0:
            raise ValueError("max_angular_speed must be finite and non-negative.")
        if not np.isfinite(joint_target_lookahead_s) or not 0.0 < joint_target_lookahead_s <= 0.09:
            raise ValueError("joint_target_lookahead_s must be in (0, 0.09] seconds.")
        if not robot.is_fixed_base:
            raise ValueError("Quest V2 requires a fixed-base OpenArm articulation.")
        self._robot = robot
        self._dt = 1.0 / float(control_hz)
        self._max_linear_speed = float(max_linear_speed)
        self._max_angular_speed = float(max_angular_speed)
        self.joint_target_lookahead_s = float(joint_target_lookahead_s)
        ee_body_names = ee_body_names or OPENARM_BIMANUAL_EE_BODY_NAMES
        self._joint_ids: dict[str, list[int]] = {}
        self._body_ids: dict[str, int] = {}
        self._jacobian_body_ids: dict[str, int] = {}
        for side in SIDES:
            joint_ids, joint_names = robot.find_joints(
                list(OPENARM_BIMANUAL_ARM_JOINT_NAMES[side]), preserve_order=True
            )
            if joint_names != list(OPENARM_BIMANUAL_ARM_JOINT_NAMES[side]):
                raise RuntimeError(f"Unexpected {side} arm joint order: {joint_names}.")
            body_ids, body_names = robot.find_bodies(ee_body_names[side])
            if len(body_ids) != 1:
                raise RuntimeError(f"Expected one {side} end-effector body, found {body_names}.")
            self._joint_ids[side] = list(joint_ids)
            self._body_ids[side] = body_ids[0]
            self._jacobian_body_ids[side] = body_ids[0] - 1
        self._nominal = {
            "left": _validate_joints(initial_left, "left"),
            "right": _validate_joints(initial_right, "right"),
        }
        self.last_singular_value = {side: float("nan") for side in SIDES}

    def sync(self, left: np.ndarray, right: np.ndarray, side: str | None = None) -> None:
        """Refresh the reset posture; a one-side clutch sync has no hidden solver state."""
        left = _validate_joints(left, "left")
        right = _validate_joints(right, "right")
        if side is None:
            self._nominal = {"left": left, "right": right}
        else:
            self._validate_side(side)

    def fk(self, side: str, _joints: np.ndarray | None = None) -> np.ndarray:
        """Read the selected Isaac hand pose in the articulation root frame."""
        self._validate_side(side)
        root_position = self._robot.data.root_pos_w[0].detach().cpu().numpy()
        root_rotation = rotation_from_wxyz(self._robot.data.root_quat_w[0].detach().cpu().numpy())
        body_id = self._body_ids[side]
        body_position = self._robot.data.body_pos_w[0, body_id].detach().cpu().numpy()
        body_rotation = rotation_from_wxyz(self._robot.data.body_quat_w[0, body_id].detach().cpu().numpy())
        position = root_rotation.inv().apply(body_position - root_position)
        rotation = root_rotation.inv() * body_rotation
        return np.asarray([*position, *rotation_to_wxyz(rotation)], dtype=np.float32)

    def solve(
        self,
        side: str,
        target_pose: np.ndarray,
        measured_left: np.ndarray,
        measured_right: np.ndarray,
    ) -> np.ndarray | None:
        """Return one safe seven-joint target, or ``None`` on a side-local failure."""
        self._validate_side(side)
        try:
            target_pose = validate_pose(target_pose, name=f"{side} target")
            measured = {
                "left": _validate_joints(measured_left, "left"),
                "right": _validate_joints(measured_right, "right"),
            }
            current_pose = self.fk(side)
            position_error = target_pose[:3] - current_pose[:3]
            orientation_error = (
                rotation_from_wxyz(target_pose[3:]) * rotation_from_wxyz(current_pose[3:]).inv()
            ).as_rotvec()
            desired_twist = np.concatenate((
                _limit_norm(position_error / self._dt, self._max_linear_speed),
                _limit_norm(orientation_error / self._dt, self._max_angular_speed),
            ))

            jacobian = self._jacobian_in_root(side)
            singular_values = np.linalg.svd(jacobian, compute_uv=False)
            smallest_singular_value = float(singular_values[-1])
            self.last_singular_value[side] = smallest_singular_value
            singularity_scale = float(np.clip(smallest_singular_value / 0.05, 0.1, 1.0))
            desired_twist *= singularity_scale

            task_weight = np.diag([12.0, 12.0, 12.0, 2.0, 2.0, 2.0])
            damping = 0.005 + 0.08 * max(0.0, 1.0 - smallest_singular_value / 0.08) ** 2
            pseudoinverse = np.linalg.pinv(jacobian, rcond=0.03)
            nullspace = np.eye(7) - pseudoinverse @ jacobian
            posture_velocity = np.clip(1.2 * (self._nominal[side] - measured[side]), -1.0, 1.0)
            nullspace_weight = 1.5
            hessian = (
                jacobian.T @ task_weight @ jacobian + damping * np.eye(7) + nullspace_weight * (nullspace.T @ nullspace)
            )
            gradient = (
                -jacobian.T @ task_weight @ desired_twist
                - nullspace_weight * (nullspace.T @ nullspace) @ posture_velocity
            )
            hessian = 0.5 * (hessian + hessian.T)

            limits = EXPECTED_JOINT_LIMITS_RAD[side]
            margin = 0.01
            lower_velocity = np.maximum(
                -JOINT_VELOCITY_LIMITS_RAD_S,
                (limits[:, 0] + margin - measured[side]) / self.joint_target_lookahead_s,
            )
            upper_velocity = np.minimum(
                JOINT_VELOCITY_LIMITS_RAD_S,
                (limits[:, 1] - margin - measured[side]) / self.joint_target_lookahead_s,
            )
            if np.any(lower_velocity > upper_velocity):
                return None
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                velocity = solve_qp(
                    hessian,
                    gradient,
                    lb=lower_velocity,
                    ub=upper_velocity,
                    solver="daqp",
                )
            if velocity is None or velocity.shape != (7,) or not np.all(np.isfinite(velocity)):
                return None
            # One-step position targets give the high-damping drive too little
            # error to build useful speed. Project the bounded QP velocity ahead
            # without reducing damping or increasing physical velocity limits.
            # At <= 0.09 s the lead remains below the device's 0.25 rad guard.
            command = measured[side] + velocity * self.joint_target_lookahead_s
            if np.any(command < limits[:, 0]) or np.any(command > limits[:, 1]):
                return None
            return command.astype(np.float32)
        except Exception:
            return None

    def _jacobian_in_root(self, side: str) -> np.ndarray:
        jacobians = self._robot.root_physx_view.get_jacobians()
        jacobian = (
            jacobians[0, self._jacobian_body_ids[side]][:, self._joint_ids[side]]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        root_rotation = rotation_from_wxyz(self._robot.data.root_quat_w[0].detach().cpu().numpy())
        rotation_root_from_world = root_rotation.inv().as_matrix()
        jacobian[:3] = rotation_root_from_world @ jacobian[:3]
        jacobian[3:] = rotation_root_from_world @ jacobian[3:]
        return jacobian

    @staticmethod
    def _validate_side(side: str) -> None:
        if side not in SIDES:
            raise ValueError(f"Unknown OpenArm side: {side!r}.")


def _validate_joints(joints: np.ndarray, side: str) -> np.ndarray:
    joints = np.asarray(joints, dtype=np.float64)
    if joints.shape != (7,) or not np.all(np.isfinite(joints)):
        raise ValueError(f"{side} arm joints must be a finite shape-(7,) array.")
    return joints.copy()


def _limit_norm(vector: np.ndarray, limit: float) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    return vector * (limit / norm) if norm > limit else vector
