"""Headless reachability regression check for OpenArm bimanual Quest teleoperation."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=120, help="Simulation steps per probe.")
parser.add_argument("--command", type=float, default=0.25, help="Absolute pose offset for each probe.")
parser.add_argument("--min_translation", type=float, default=0.08, help="Minimum hand translation in meters.")
parser.add_argument("--min_rotation", type=float, default=0.08, help="Minimum hand rotation in radians.")
parser.add_argument("--disable_self_collisions", action="store_true", help="Negative-control collision run.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from isaaclab.utils.math import quat_error_magnitude, subtract_frame_transforms
from isaaclab_tasks.utils import parse_env_cfg
from scipy.spatial.transform import Rotation

import leisaac.tasks  # noqa: F401
from leisaac.devices.quest3_controller import Quest3Controller, _OneEuroPoseSmoother


TASK = "LeIsaac-OpenArm-Bimanual-LiftCube-v0"


def rotation_from_wxyz(quaternion):
    """Create a SciPy rotation from an Isaac-style wxyz quaternion."""
    return Rotation.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])


def rotation_to_wxyz(rotation):
    """Return a SciPy rotation as an Isaac-style wxyz quaternion."""
    x, y, z, w = rotation.as_quat()
    return np.asarray([w, x, y, z], dtype=np.float32)


def check_pose_math():
    """Verify rotation direction, quaternion order, and absolute-target speed caps."""
    cosine = np.sqrt(0.5)
    controller_origin = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    controller_pose = np.array([0.1, 0.0, 0.0, cosine, 0.0, 0.0, cosine], dtype=np.float32)
    robot_origin = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    target = Quest3Controller._compose_absolute_target(
        controller_pose, controller_origin, robot_origin, 1.0, 1.0
    )
    np.testing.assert_allclose(target, controller_pose, atol=1e-5)

    smoother = _OneEuroPoseSmoother(max_linear_speed=1.0, max_angular_speed=6.0)
    smoother.smooth(1.0, robot_origin)
    jump = np.array([10.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    limited = smoother.smooth(1.1, jump)
    linear_step = np.linalg.norm(limited[:3] - robot_origin[:3])
    angular_step = (
        rotation_from_wxyz(limited[3:]) * rotation_from_wxyz(robot_origin[3:]).inv()
    ).magnitude()
    assert linear_step <= 0.10001
    assert angular_step <= 0.60001
    print("pose math: wxyz direction and target speed caps passed")


def probe(env, robot, left_body_id, right_body_id, name, action_index, left_sign, right_sign, minimum, rotation):
    """Apply one absolute pose target and assert both hands respond."""
    env.reset()
    start_pos = torch.stack((robot.data.body_pos_w[0, left_body_id], robot.data.body_pos_w[0, right_body_id])).clone()
    start_quat = torch.stack((robot.data.body_quat_w[0, left_body_id], robot.data.body_quat_w[0, right_body_id])).clone()
    root_pos = robot.data.root_pos_w[:1].repeat(2, 1)
    root_quat = robot.data.root_quat_w[:1].repeat(2, 1)
    target_pos, target_quat = subtract_frame_transforms(root_pos, root_quat, start_pos, start_quat)
    action = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
    action[0, :3] = target_pos[0]
    action[0, 3:7] = target_quat[0]
    action[0, 8:11] = target_pos[1]
    action[0, 11:15] = target_quat[1]
    if rotation:
        axis = np.eye(3)[action_index - 3]
        left_quat = rotation_to_wxyz(
            Rotation.from_rotvec(axis * args_cli.command * left_sign)
            * rotation_from_wxyz(target_quat[0].cpu().numpy())
        )
        right_quat = rotation_to_wxyz(
            Rotation.from_rotvec(axis * args_cli.command * right_sign)
            * rotation_from_wxyz(target_quat[1].cpu().numpy())
        )
        action[0, 3:7] = torch.as_tensor(left_quat, device=env.device)
        action[0, 11:15] = torch.as_tensor(right_quat, device=env.device)
    else:
        action[0, action_index] += args_cli.command * left_sign
        action[0, 8 + action_index] += args_cli.command * right_sign
    action[:, 7] = 1.0
    action[:, 15] = 1.0
    for _ in range(args_cli.steps):
        env.step(action)
    if rotation:
        end_quat = torch.stack((robot.data.body_quat_w[0, left_body_id], robot.data.body_quat_w[0, right_body_id]))
        measured = quat_error_magnitude(end_quat, start_quat)
        unit = "rad"
    else:
        end_pos = torch.stack((robot.data.body_pos_w[0, left_body_id], robot.data.body_pos_w[0, right_body_id]))
        measured = torch.linalg.vector_norm(end_pos - start_pos, dim=-1)
        unit = "m"
    values = measured.cpu().tolist()
    print(f"{name}: left={values[0]:.3f}{unit}, right={values[1]:.3f}{unit}")
    if min(values) < minimum:
        raise AssertionError(f"{name} response below {minimum:.3f}{unit}: {values}")


def current_pose_action(env, robot, left_body_id, right_body_id):
    """Build a 16D absolute command that holds both hands at their current poses."""
    body_pos = torch.stack((robot.data.body_pos_w[0, left_body_id], robot.data.body_pos_w[0, right_body_id]))
    body_quat = torch.stack((robot.data.body_quat_w[0, left_body_id], robot.data.body_quat_w[0, right_body_id]))
    root_pos = robot.data.root_pos_w[:1].repeat(2, 1)
    root_quat = robot.data.root_quat_w[:1].repeat(2, 1)
    target_pos, target_quat = subtract_frame_transforms(root_pos, root_quat, body_pos, body_quat)
    action = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
    action[0, :3] = target_pos[0]
    action[0, 3:7] = target_quat[0]
    action[0, 8:11] = target_pos[1]
    action[0, 11:15] = target_quat[1]
    action[:, 7] = 1.0
    action[:, 15] = 1.0
    return action


def check_collision_geometry():
    """Assert the composed USD contains torso and bilateral arm collision geometry."""
    import omni.usd
    from pxr import UsdPhysics

    stage = omni.usd.get_context().get_stage()
    collision_paths = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if str(prim.GetPath()).startswith("/World/envs/env_0/Robot")
        and prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    required = ("openarm_body_link", "openarm_left_link", "openarm_right_link")
    missing = [name for name in required if not any(name in path for path in collision_paths)]
    if missing:
        raise AssertionError(f"Missing collision geometry for {missing}; found {len(collision_paths)} collision prims")
    print(f"collision geometry: {len(collision_paths)} prims covering torso and both arms")


def check_posture_and_limits(env, robot, left_body_id, right_body_id):
    """Hold the current pose and verify finite, bounded joints and bent elbows."""
    env.reset()
    action = current_pose_action(env, robot, left_body_id, right_body_id)
    for _ in range(args_cli.steps):
        env.step(action)
    names = [f"openarm_{side}_joint{index}" for side in ("left", "right") for index in range(1, 8)]
    joint_ids, _ = robot.find_joints(names, preserve_order=True)
    joint_pos = robot.data.joint_pos[0, joint_ids]
    limits = robot.data.soft_joint_pos_limits[0, joint_ids]
    if not torch.isfinite(joint_pos).all():
        raise AssertionError(f"Non-finite joint positions: {joint_pos}")
    if not ((joint_pos >= limits[:, 0] - 1e-3) & (joint_pos <= limits[:, 1] + 1e-3)).all():
        raise AssertionError(f"Joint limit violation: q={joint_pos}, limits={limits}")
    elbows = joint_pos[[3, 10]]
    if torch.min(elbows) < 0.5:
        raise AssertionError(f"Elbow posture collapsed toward the singular lower limit: {elbows}")
    margin = torch.min(torch.minimum(joint_pos - limits[:, 0], limits[:, 1] - joint_pos))
    print(f"posture/limits: elbows={elbows.tolist()}, minimum_limit_margin={float(margin):.3f}rad")


def check_hand_collision(env, robot, left_body_id, right_body_id):
    """Swap hand targets and verify collision geometry prevents intersection while the arms cross."""
    env.reset()
    action = current_pose_action(env, robot, left_body_id, right_body_id)
    left_position = action[0, :3].clone()
    right_position = action[0, 8:11].clone()
    action[0, :3] = right_position
    action[0, 8:11] = left_position
    minimum_separation = float("inf")
    for _ in range(args_cli.steps * 2):
        env.step(action)
        separation = torch.linalg.vector_norm(
            robot.data.body_pos_w[0, left_body_id] - robot.data.body_pos_w[0, right_body_id]
        )
        minimum_separation = min(minimum_separation, float(separation))
    final_separation = torch.linalg.vector_norm(
        robot.data.body_pos_w[0, left_body_id] - robot.data.body_pos_w[0, right_body_id]
    )
    print(
        f"crossing-hand collision: minimum={minimum_separation:.3f}m, "
        f"final={float(final_separation):.3f}m"
    )
    if minimum_separation < 0.150:
        raise AssertionError(f"Hands interpenetrated while crossing: {minimum_separation:.3f}m")



def main():
    check_pose_math()

    cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=1)
    cfg.seed = 42
    cfg.use_teleop_device("quest3-controller")
    if args_cli.disable_self_collisions:
        cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
    env = gym.make(TASK, cfg=cfg).unwrapped
    robot = env.scene["robot"]
    if env.action_manager.total_action_dim != 16:
        raise AssertionError(f"Expected 16D bimanual action, got {env.action_manager.total_action_dim}")
    left_body_id = robot.data.body_names.index("openarm_left_hand")
    right_body_id = robot.data.body_names.index("openarm_right_hand")
    check_collision_geometry()
    try:
        for spec in (
            ("forward", 0, 1, 1, args_cli.min_translation, False),
            ("backward", 0, -1, -1, args_cli.min_translation, False),
            ("outward", 1, 1, -1, args_cli.min_translation, False),
            ("upward", 2, 1, 1, args_cli.min_translation, False),
            ("roll", 3, 1, 1, args_cli.min_rotation, True),
            ("pitch", 4, 1, 1, args_cli.min_rotation, True),
            ("yaw", 5, 1, 1, args_cli.min_rotation, True),
        ):
            probe(env, robot, left_body_id, right_body_id, *spec)
        check_posture_and_limits(env, robot, left_body_id, right_body_id)
        if args_cli.disable_self_collisions:
            try:
                check_hand_collision(env, robot, left_body_id, right_body_id)
            except AssertionError as error:
                print(f"Negative control reproduced the collision regression: {error}")
            else:
                raise AssertionError("Negative control did not reproduce the expected collision regression")
            print("OpenArm bimanual negative-control check passed.")
        else:
            check_hand_collision(env, robot, left_body_id, right_body_id)
            print("OpenArm bimanual headless physics checks passed.")
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
