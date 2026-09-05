"""Deterministic tests for the OpenArm Quest V2 pose mapper."""

import numpy as np
from scipy.spatial.transform import Rotation

from leisaac.devices.openarm_vr_v2.core import (
    ClutchedPoseMapper,
    HandTeleopState,
    OneEuroPoseSmoother,
    OperatorFrame,
    TrackedPoseSample,
    read_openxr_input_value,
    rotation_to_wxyz,
    validate_pose,
)


IDENTITY_POSE = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def sample(timestamp, pose=IDENTITY_POSE, *, valid=True, squeeze=0.0, trigger=0.0):
    return TrackedPoseSample(timestamp, None if pose is None else pose.copy(), valid, squeeze, trigger)


def test_quaternion_sign_is_canonical():
    pose = validate_pose(np.asarray([0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(pose, IDENTITY_POSE)


def test_hmd_yaw_frame_is_forward_left_up():
    head_rotation = Rotation.from_euler("y", -90.0, degrees=True)
    head_pose = np.asarray([1.0, 2.0, 3.0, *rotation_to_wxyz(head_rotation)])
    frame = OperatorFrame.from_head_pose(head_pose)
    basis = frame.rotation_w_from_operator.as_matrix()
    np.testing.assert_allclose(basis[:, 0], [1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(basis[:, 1], [0.0, 1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(basis[:, 2], [0.0, 0.0, 1.0], atol=1e-6)


def test_stationary_controller_target_does_not_depend_on_later_head_motion():
    mapper = ClutchedPoseMapper(max_linear_speed=100.0, max_angular_speed=100.0)
    operator_frame = OperatorFrame(np.zeros(3), Rotation.identity())
    robot_pose = np.asarray([0.2, 0.1, -0.3, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    started = mapper.step(
        sample(1.0, squeeze=1.0),
        now_s=1.0,
        operator_frame=operator_frame,
        robot_pose=robot_pose,
        enabled=True,
    )
    same = mapper.step(
        sample(1.01, squeeze=1.0),
        now_s=1.01,
        operator_frame=operator_frame,
        robot_pose=robot_pose,
        enabled=True,
    )
    assert started.clutch_started
    np.testing.assert_allclose(started.target_pose, robot_pose, atol=1e-7)
    np.testing.assert_allclose(same.target_pose, robot_pose, atol=1e-7)


def test_relative_translation_uses_operator_axes():
    mapper = ClutchedPoseMapper(max_linear_speed=100.0, max_angular_speed=100.0)
    operator_frame = OperatorFrame(np.zeros(3), Rotation.identity())
    robot_pose = IDENTITY_POSE.copy()
    mapper.step(
        sample(1.0, squeeze=1.0),
        now_s=1.0,
        operator_frame=operator_frame,
        robot_pose=robot_pose,
        enabled=True,
    )
    moved_pose = IDENTITY_POSE.copy()
    moved_pose[0] = 0.1
    moved = mapper.step(
        sample(1.1, moved_pose, squeeze=1.0),
        now_s=1.1,
        operator_frame=operator_frame,
        robot_pose=robot_pose,
        enabled=True,
    )
    assert moved.target_pose[0] > 0.0
    np.testing.assert_allclose(moved.target_pose[1:3], 0.0, atol=1e-7)


def test_tracking_loss_requires_release_before_reclutch():
    mapper = ClutchedPoseMapper()
    operator_frame = OperatorFrame(np.zeros(3), Rotation.identity())
    mapper.step(
        sample(1.0, squeeze=1.0),
        now_s=1.0,
        operator_frame=operator_frame,
        robot_pose=IDENTITY_POSE,
        enabled=True,
    )
    lost = mapper.step(
        sample(1.01, None, valid=False, squeeze=1.0),
        now_s=1.01,
        operator_frame=operator_frame,
        robot_pose=IDENTITY_POSE,
        enabled=True,
    )
    held = mapper.step(
        sample(1.02, squeeze=1.0),
        now_s=1.02,
        operator_frame=operator_frame,
        robot_pose=IDENTITY_POSE,
        enabled=True,
    )
    released = mapper.step(
        sample(1.03, squeeze=0.0),
        now_s=1.03,
        operator_frame=operator_frame,
        robot_pose=IDENTITY_POSE,
        enabled=True,
    )
    restarted = mapper.step(
        sample(1.04, squeeze=1.0),
        now_s=1.04,
        operator_frame=operator_frame,
        robot_pose=IDENTITY_POSE,
        enabled=True,
    )
    assert lost.state == HandTeleopState.HOLD
    assert held.state == HandTeleopState.HOLD
    assert held.target_pose is None
    assert released.state == HandTeleopState.READY
    assert restarted.state == HandTeleopState.CLUTCHED
    assert restarted.clutch_started


def test_gripper_hysteresis_does_not_chatter():
    mapper = ClutchedPoseMapper()
    operator_frame = OperatorFrame(np.zeros(3), Rotation.identity())
    robot_pose = IDENTITY_POSE
    mapper.step(sample(1.0, trigger=0.7), now_s=1.0, operator_frame=operator_frame, robot_pose=robot_pose, enabled=True)
    assert mapper.gripper_closed
    mapper.step(sample(1.01, trigger=0.5), now_s=1.01, operator_frame=operator_frame, robot_pose=robot_pose, enabled=True)
    assert mapper.gripper_closed
    mapper.step(sample(1.02, trigger=0.3), now_s=1.02, operator_frame=operator_frame, robot_pose=robot_pose, enabled=True)
    assert not mapper.gripper_closed


def test_openxr_trigger_uses_the_strongest_supported_channel():
    class FakeInputDevice:
        values = {("trigger", "value"): 0.0, ("trigger", "force"): 0.8}

        def has_input_gesture(self, input_name, gesture_name):
            return (input_name, gesture_name) in self.values

        def get_input_gesture_value(self, input_name, gesture_name):
            return self.values[(input_name, gesture_name)]

    assert read_openxr_input_value(FakeInputDevice(), "trigger", "value", "force", "click") == 0.8


def test_one_euro_filter_respects_60hz_linear_and_angular_caps():
    smoother = OneEuroPoseSmoother(max_linear_speed=0.6, max_angular_speed=3.0)
    first = smoother.smooth(1.0, IDENTITY_POSE)
    jump = np.asarray(
        [1.0, 0.0, 0.0, *rotation_to_wxyz(Rotation.from_euler("z", 180.0, degrees=True))],
        dtype=np.float32,
    )
    second = smoother.smooth(1.0 + 1.0 / 60.0, jump)
    linear_step = np.linalg.norm(second[:3] - first[:3])
    angular_step = (
        Rotation.from_quat([second[4], second[5], second[6], second[3]])
        * Rotation.from_quat([first[4], first[5], first[6], first[3]]).inv()
    ).magnitude()
    assert linear_step <= 0.010001
    assert angular_step <= 0.050001
