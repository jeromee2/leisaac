# Copyright (c) 2026, LeIsaac Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Diagnostic-only MuJoCo/Mink controller used to reproduce model mismatch.

The Quest V2 runtime intentionally uses :mod:`isaac_qp_controller` instead.
"""

from __future__ import annotations

import numpy as np


SIDES = ("left", "right")
JOINT_VELOCITY_LIMITS_RAD_S = np.asarray([2.0, 2.0, 2.175, 2.175, 2.61, 2.61, 2.61])
EXPECTED_JOINT_LIMITS_RAD = {
    "left": np.asarray(
        [
            [-3.490659, 1.396263],
            [-3.316126, 0.174533],
            [-1.570796, 1.570796],
            [0.0, 2.443461],
            [-1.570796, 1.570796],
            [-0.785398, 0.785398],
            [-1.570796, 1.570796],
        ]
    ),
    "right": np.asarray(
        [
            [-1.396263, 3.490659],
            [-0.174533, 3.316126],
            [-1.570796, 1.570796],
            [0.0, 2.443461],
            [-1.570796, 1.570796],
            [-0.785398, 0.785398],
            [-1.570796, 1.570796],
        ]
    ),
}


def pack_driver_state(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Pack Isaac left/right arm joints into OpenArm driver's right-first 16D layout."""
    left = _validate_joints(left, "left")
    right = _validate_joints(right, "right")
    return np.concatenate((right, [0.0], left, [0.0])).astype(np.float32)


class OpenArmMinkControllerV2:
    """Run one single-arm QP per side so failures remain independent."""

    def __init__(self, initial_left: np.ndarray, initial_right: np.ndarray, *, control_hz: float = 60.0) -> None:
        if not np.isfinite(control_hz) or control_hz <= 0.0:
            raise ValueError("control_hz must be finite and positive.")
        try:
            import mujoco
            import openarm_mujoco.v2 as openarm_mujoco
            from openarm_control import ArmSetup, IKParams, Kinematics
        except ImportError as error:
            raise RuntimeError(
                "This diagnostic requires openarm-control, openarm-mujoco, mujoco, mink, "
                "qpsolvers, and daqp; Quest V2 runtime does not use this controller."
            ) from error

        initial = {"left": _validate_joints(initial_left, "left"), "right": _validate_joints(initial_right, "right")}
        velocity_limits = {
            f"openarm_{side}_joint{index + 1}": float(limit)
            for side in SIDES
            for index, limit in enumerate(JOINT_VELOCITY_LIMITS_RAD_S)
        }
        params = IKParams(
            position_cost=12.0,
            orientation_cost=1.5,
            lm_damping=0.01,
            damping=0.1,
            solver="daqp",
            posture_cost=0.0,
            dt=1.0 / control_hz,
            max_iters=5,
            velocity_limits=velocity_limits,
            frame_position_error_limit=0.02,
            frame_orientation_error_limit=0.25,
            target_linear_speed_slow=0.6,
            target_linear_speed_fast=0.9,
            position_error_latch_threshold=0.006,
            joint_limit_gain=0.95,
            joint_braking=True,
            joint_braking_distance=0.2,
            joint_braking_exponent=2.0,
            joint_braking_distance_buffer=0.01,
            jacobian_characteristic_length=0.3,
            nullspace_cost=8.5,
            nullspace_return_rate=1.6,
            nullspace_max_speed=1.0,
            nullspace_ratio_low=0.02,
            nullspace_ratio_high=0.05,
            singularity_ratio_stop=0.02,
            singularity_ratio_slow=0.08,
            singularity_max_approach_rate=0.25,
            singularity_braking_exponent=2.0,
            singularity_gradient_epsilon=1e-4,
            kinetic_energy_cost=2e-5,
        )

        xml_path = openarm_mujoco.openarm_cell_xml()
        self._kinematics: dict[str, object] = {}
        for side in SIDES:
            setup = ArmSetup.from_args(
                xml=xml_path,
                mode=side,
                frame_right="right_ee_control_point",
                frame_type_right="site",
                frame_left="left_ee_control_point",
                frame_type_left="site",
                keyframe="home",
                origin_frame="arm_origin",
                origin_frame_type="site",
            )
            setup.joint_resolver.set_qpos(setup.data.qpos, np.append(initial[side], 0.0), side)
            mujoco.mj_forward(setup.model, setup.data)
            self._kinematics[side] = Kinematics(setup, params)
        self.sync(initial_left, initial_right)

    def sync(self, left: np.ndarray, right: np.ndarray, side: str | None = None) -> None:
        """Reset one or both command configurations to the measured joints."""
        packed = pack_driver_state(left, right)
        for active_side in SIDES if side is None else (side,):
            self._validate_side(active_side)
            self._kinematics[active_side].sync(packed)

    def fk(self, side: str, joints: np.ndarray) -> np.ndarray:
        """Return the selected MuJoCo control-point pose in ``arm_origin``."""
        self._validate_side(side)
        return self._kinematics[side].fk(side, np.append(_validate_joints(joints, side), 0.0))

    def solve(
        self,
        side: str,
        target_pose: np.ndarray,
        measured_left: np.ndarray,
        measured_right: np.ndarray,
    ) -> np.ndarray | None:
        """Solve one side and return seven arm joints, or ``None`` on any unsafe result."""
        self._validate_side(side)
        target_pose = np.asarray(target_pose, dtype=np.float64)
        if target_pose.shape != (7,) or not np.all(np.isfinite(target_pose)):
            return None
        kinematics = self._kinematics[side]
        try:
            kinematics.update_measured_state(pack_driver_state(measured_left, measured_right))
            kinematics.set_target(side, target_pose)
            result = kinematics.solve()
        except (ValueError, FloatingPointError):
            return None
        if result is None or result.shape != (16,) or not np.all(np.isfinite(result)):
            return None
        joints = result[8:15] if side == "left" else result[:7]
        limits = EXPECTED_JOINT_LIMITS_RAD[side]
        if np.any(joints < limits[:, 0] - 1e-4) or np.any(joints > limits[:, 1] + 1e-4):
            return None
        return joints.astype(np.float32)

    @staticmethod
    def _validate_side(side: str) -> None:
        if side not in SIDES:
            raise ValueError(f"Unknown OpenArm side: {side!r}.")


def _validate_joints(joints: np.ndarray, side: str) -> np.ndarray:
    joints = np.asarray(joints, dtype=np.float64)
    if joints.shape != (7,) or not np.all(np.isfinite(joints)):
        raise ValueError(f"{side} arm joints must be a finite shape-(7,) array.")
    return joints.copy()
