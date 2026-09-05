# Copyright (c) 2026, LeIsaac Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Pure pose mapping and safety state for OpenArm Quest teleoperation V2."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def read_openxr_input_value(input_device: object, input_name: str, *gesture_names: str) -> float:
    """Return the strongest available value for an OpenXR input."""
    if input_device is None:
        return 0.0
    values = []
    for gesture_name in gesture_names:
        try:
            if input_device.has_input_gesture(input_name, gesture_name):
                values.append(float(input_device.get_input_gesture_value(input_name, gesture_name)))
        except Exception:
            continue
    return float(np.clip(max(values, default=0.0), 0.0, 1.0))


def rotation_from_wxyz(quaternion: np.ndarray) -> Rotation:
    """Create a SciPy rotation from a normalized scalar-first quaternion."""
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("Quaternion must be a finite shape-(4,) wxyz array.")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-8:
        raise ValueError("Quaternion norm is zero.")
    quaternion = quaternion / norm
    return Rotation.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])


def rotation_to_wxyz(rotation: Rotation) -> np.ndarray:
    """Return a normalized scalar-first quaternion with a stable sign."""
    x, y, z, w = rotation.as_quat()
    quaternion = np.asarray([w, x, y, z], dtype=np.float64)
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    return quaternion.astype(np.float32)


def validate_pose(pose: np.ndarray, *, name: str = "pose") -> np.ndarray:
    """Return a finite, copied ``xyz+wxyz`` pose with normalized quaternion."""
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        raise ValueError(f"{name} must be a finite shape-(7,) xyz+wxyz array.")
    output = pose.copy()
    output[3:] = rotation_to_wxyz(rotation_from_wxyz(output[3:]))
    return output.astype(np.float32)


class HandTeleopState(str, Enum):
    """Per-hand control state."""

    WAITING = "waiting"
    READY = "ready"
    CLUTCHED = "clutched"
    HOLD = "hold"
    FAULT = "fault"


@dataclass(frozen=True)
class TrackedPoseSample:
    """One timestamped OpenXR controller or HMD sample."""

    timestamp_s: float
    pose: np.ndarray | None
    valid: bool
    squeeze: float = 0.0
    trigger: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.timestamp_s):
            raise ValueError("Tracked pose timestamp must be finite.")
        object.__setattr__(self, "squeeze", float(np.clip(self.squeeze, 0.0, 1.0)))
        object.__setattr__(self, "trigger", float(np.clip(self.trigger, 0.0, 1.0)))
        if self.pose is not None:
            object.__setattr__(self, "pose", validate_pose(self.pose))


@dataclass(frozen=True)
class OpenXRInputFrame:
    """A synchronized poll of the HMD and both Touch controllers."""

    timestamp_s: float
    head: TrackedPoseSample
    hands: dict[str, TrackedPoseSample]


@dataclass(frozen=True)
class OperatorFrame:
    """Fixed forward-left-up operator frame captured from HMD yaw."""

    origin_w: np.ndarray
    rotation_w_from_operator: Rotation

    @classmethod
    def from_head_pose(cls, pose: np.ndarray) -> OperatorFrame:
        pose = validate_pose(pose, name="head pose")
        head_rotation = rotation_from_wxyz(pose[3:])
        # OpenXR's view forward is local -Z. Project it onto the Isaac Z-up floor.
        forward_w = head_rotation.apply(np.asarray([0.0, 0.0, -1.0]))
        forward_w[2] = 0.0
        norm = float(np.linalg.norm(forward_w))
        if norm < 1e-5:
            raise ValueError("Cannot extract HMD yaw while its forward axis is vertical.")
        forward_w /= norm
        up_w = np.asarray([0.0, 0.0, 1.0])
        left_w = np.cross(up_w, forward_w)
        left_w /= np.linalg.norm(left_w)
        basis = np.column_stack((forward_w, left_w, up_w))
        return cls(pose[:3].astype(np.float64), Rotation.from_matrix(basis))

    def express_pose(self, pose_w: np.ndarray) -> tuple[np.ndarray, Rotation]:
        """Express a virtual-world pose in forward-left-up operator axes."""
        pose_w = validate_pose(pose_w)
        rotation_operator_from_w = self.rotation_w_from_operator.inv()
        position = rotation_operator_from_w.apply(pose_w[:3] - self.origin_w)
        rotation = rotation_operator_from_w * rotation_from_wxyz(pose_w[3:])
        return position, rotation


class OneEuroPoseSmoother:
    """Timestamp-aware One-Euro position filter and quaternion SLERP filter."""

    def __init__(
        self,
        *,
        min_cutoff: float = 2.0,
        beta: float = 0.04,
        d_cutoff: float = 1.5,
        max_linear_speed: float = 0.6,
        max_angular_speed: float = 3.0,
    ) -> None:
        values = (min_cutoff, beta, d_cutoff, max_linear_speed, max_angular_speed)
        if not all(np.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("One-Euro parameters must be finite and non-negative.")
        if min_cutoff == 0.0 or d_cutoff == 0.0:
            raise ValueError("One-Euro cutoff frequencies must be positive.")
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.max_linear_speed = float(max_linear_speed)
        self.max_angular_speed = float(max_angular_speed)
        self.reset()

    def reset(self) -> None:
        self._position: np.ndarray | None = None
        self._quaternion: np.ndarray | None = None
        self._velocity = np.zeros(3, dtype=np.float64)
        self._timestamp: float | None = None

    def suspend(self, timestamp_s: float) -> None:
        """Preserve the last output while discarding derivative history."""
        self._velocity.fill(0.0)
        self._timestamp = float(timestamp_s)

    def smooth(self, timestamp_s: float, target_pose: np.ndarray) -> np.ndarray:
        target_pose = validate_pose(target_pose, name="filter target")
        if self._timestamp is None or self._position is None or self._quaternion is None:
            self._position = target_pose[:3].astype(np.float64)
            self._quaternion = target_pose[3:].astype(np.float64)
            self._timestamp = float(timestamp_s)
            return target_pose.copy()

        dt = float(timestamp_s - self._timestamp)
        if dt <= 0.0:
            return np.asarray([*self._position, *self._quaternion], dtype=np.float32)

        def alpha(cutoff: float) -> float:
            tau = 1.0 / (2.0 * np.pi * cutoff)
            return 1.0 / (1.0 + tau / dt)

        raw_velocity = (target_pose[:3] - self._position) / dt
        derivative_alpha = alpha(self.d_cutoff)
        self._velocity = derivative_alpha * raw_velocity + (1.0 - derivative_alpha) * self._velocity
        pose_alpha = alpha(self.min_cutoff + self.beta * np.linalg.norm(self._velocity))

        position_step = pose_alpha * (target_pose[:3] - self._position)
        position_norm = float(np.linalg.norm(position_step))
        linear_limit = self.max_linear_speed * dt
        if position_norm > linear_limit > 0.0:
            position_step *= linear_limit / position_norm
        elif self.max_linear_speed == 0.0:
            position_step.fill(0.0)
        self._position += position_step

        current_rotation = rotation_from_wxyz(self._quaternion)
        target_rotation = rotation_from_wxyz(target_pose[3:])
        angle = float((target_rotation * current_rotation.inv()).magnitude())
        rotation_fraction = pose_alpha
        angular_limit = self.max_angular_speed * dt
        if angle * rotation_fraction > angular_limit > 0.0:
            rotation_fraction = angular_limit / angle
        elif self.max_angular_speed == 0.0:
            rotation_fraction = 0.0
        rotations = Rotation.from_quat([current_rotation.as_quat(), target_rotation.as_quat()])
        filtered_rotation = Slerp([0.0, 1.0], rotations)([rotation_fraction])[0]
        self._quaternion = rotation_to_wxyz(filtered_rotation).astype(np.float64)
        self._timestamp = float(timestamp_s)
        return np.asarray([*self._position, *self._quaternion], dtype=np.float32)


@dataclass(frozen=True)
class HandMappingResult:
    """Pose target and discrete state generated for one hand."""

    target_pose: np.ndarray | None
    state: HandTeleopState
    gripper_closed: bool
    clutch_started: bool = False
    reason: str | None = None


class ClutchedPoseMapper:
    """Map controller deltas onto the current robot EE pose without head coupling."""

    def __init__(
        self,
        *,
        position_scale: float = 1.0,
        rotation_scale: float = 1.0,
        clutch_engage_threshold: float = 0.55,
        clutch_release_threshold: float = 0.45,
        gripper_close_threshold: float = 0.65,
        gripper_open_threshold: float = 0.35,
        stale_timeout_s: float = 0.1,
        jump_translation_m: float = 0.15,
        jump_rotation_rad: float = 0.75,
        max_linear_speed: float = 0.6,
        max_angular_speed: float = 3.0,
    ) -> None:
        if position_scale < 0.0 or rotation_scale < 0.0:
            raise ValueError("Pose scales must be non-negative.")
        if clutch_release_threshold > clutch_engage_threshold:
            raise ValueError("Clutch release threshold must not exceed engage threshold.")
        if gripper_open_threshold > gripper_close_threshold:
            raise ValueError("Gripper open threshold must not exceed close threshold.")
        self.position_scale = float(position_scale)
        self.rotation_scale = float(rotation_scale)
        self.clutch_engage_threshold = float(clutch_engage_threshold)
        self.clutch_release_threshold = float(clutch_release_threshold)
        self.gripper_close_threshold = float(gripper_close_threshold)
        self.gripper_open_threshold = float(gripper_open_threshold)
        self.stale_timeout_s = float(stale_timeout_s)
        self.jump_translation_m = float(jump_translation_m)
        self.jump_rotation_rad = float(jump_rotation_rad)
        self._smoother = OneEuroPoseSmoother(
            min_cutoff=2.0,
            beta=0.04,
            d_cutoff=1.5,
            max_linear_speed=max_linear_speed,
            max_angular_speed=max_angular_speed,
        )
        self.reset()

    @property
    def state(self) -> HandTeleopState:
        return self._state

    @property
    def gripper_closed(self) -> bool:
        return self._gripper_closed

    def reset(self) -> None:
        self._state = HandTeleopState.WAITING
        self._requires_release = False
        self._gripper_closed = False
        self._controller_origin_position: np.ndarray | None = None
        self._controller_origin_rotation: Rotation | None = None
        self._robot_origin_pose: np.ndarray | None = None
        self._previous_valid_pose: np.ndarray | None = None
        self._previous_valid_timestamp: float | None = None
        self._smoother.reset()

    def mark_hold(self, timestamp_s: float, *, fault: bool = False) -> None:
        """Stop this hand until a release/re-clutch sequence is observed."""
        self._state = HandTeleopState.FAULT if fault else HandTeleopState.HOLD
        self._requires_release = True
        self._controller_origin_position = None
        self._controller_origin_rotation = None
        self._robot_origin_pose = None
        self._smoother.suspend(timestamp_s)

    def step(
        self,
        sample: TrackedPoseSample,
        *,
        now_s: float,
        operator_frame: OperatorFrame | None,
        robot_pose: np.ndarray,
        enabled: bool,
    ) -> HandMappingResult:
        """Advance one hand and return a filtered robot-frame target when clutched."""
        robot_pose = validate_pose(robot_pose, name="robot FK pose")
        self._update_gripper(sample.trigger)

        if not enabled:
            self._clear_motion(HandTeleopState.WAITING)
            return self._result()
        if operator_frame is None:
            self._clear_motion(HandTeleopState.WAITING)
            return self._result(reason="waiting for HMD yaw calibration")
        if not sample.valid or sample.pose is None or now_s - sample.timestamp_s > self.stale_timeout_s:
            self.mark_hold(now_s)
            return self._result(reason="controller tracking invalid or stale")

        if self._is_pose_jump(sample.pose, sample.timestamp_s):
            self._remember_pose(sample.pose, sample.timestamp_s)
            self.mark_hold(now_s)
            return self._result(reason="controller reference-space jump")
        self._remember_pose(sample.pose, sample.timestamp_s)

        if self._requires_release:
            if sample.squeeze <= self.clutch_release_threshold:
                self._requires_release = False
                self._clear_motion(HandTeleopState.READY)
            return self._result(reason="release squeeze before re-clutch")

        clutch_engaged = (
            sample.squeeze >= self.clutch_release_threshold
            if self._state == HandTeleopState.CLUTCHED
            else sample.squeeze >= self.clutch_engage_threshold
        )
        if not clutch_engaged:
            self._clear_motion(HandTeleopState.READY)
            return self._result()

        controller_position, controller_rotation = operator_frame.express_pose(sample.pose)
        if self._state != HandTeleopState.CLUTCHED:
            self._state = HandTeleopState.CLUTCHED
            self._controller_origin_position = controller_position
            self._controller_origin_rotation = controller_rotation
            self._robot_origin_pose = robot_pose.copy()
            self._smoother.reset()
            target = self._smoother.smooth(sample.timestamp_s, robot_pose)
            return self._result(target=target, clutch_started=True)

        assert self._controller_origin_position is not None
        assert self._controller_origin_rotation is not None
        assert self._robot_origin_pose is not None
        target_position = self._robot_origin_pose[:3] + self.position_scale * (
            controller_position - self._controller_origin_position
        )
        controller_delta = controller_rotation * self._controller_origin_rotation.inv()
        if self.rotation_scale != 1.0:
            controller_delta = Rotation.from_rotvec(controller_delta.as_rotvec() * self.rotation_scale)
        target_rotation = controller_delta * rotation_from_wxyz(self._robot_origin_pose[3:])
        raw_target = np.asarray([*target_position, *rotation_to_wxyz(target_rotation)], dtype=np.float32)
        target = self._smoother.smooth(sample.timestamp_s, raw_target)
        return self._result(target=target)

    def _is_pose_jump(self, pose: np.ndarray, timestamp_s: float) -> bool:
        if self._previous_valid_pose is None or self._previous_valid_timestamp is None:
            return False
        dt = timestamp_s - self._previous_valid_timestamp
        if dt <= 0.0 or dt > self.stale_timeout_s:
            return False
        translation = float(np.linalg.norm(pose[:3] - self._previous_valid_pose[:3]))
        rotation = float(
            (rotation_from_wxyz(pose[3:]) * rotation_from_wxyz(self._previous_valid_pose[3:]).inv()).magnitude()
        )
        return translation > self.jump_translation_m or rotation > self.jump_rotation_rad

    def _remember_pose(self, pose: np.ndarray, timestamp_s: float) -> None:
        self._previous_valid_pose = pose.copy()
        self._previous_valid_timestamp = float(timestamp_s)

    def _update_gripper(self, trigger: float) -> None:
        if trigger >= self.gripper_close_threshold:
            self._gripper_closed = True
        elif trigger <= self.gripper_open_threshold:
            self._gripper_closed = False

    def _clear_motion(self, state: HandTeleopState) -> None:
        self._state = state
        self._controller_origin_position = None
        self._controller_origin_rotation = None
        self._robot_origin_pose = None
        self._smoother.reset()

    def _result(
        self,
        *,
        target: np.ndarray | None = None,
        clutch_started: bool = False,
        reason: str | None = None,
    ) -> HandMappingResult:
        return HandMappingResult(target, self._state, self._gripper_closed, clutch_started, reason)
