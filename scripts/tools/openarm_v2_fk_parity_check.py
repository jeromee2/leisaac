"""Verify OpenArm native QP targets and Isaac USD end-effector axis parity."""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--iterations", type=int, default=60)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab_tasks.utils import parse_env_cfg
from scipy.spatial.transform import Rotation

import leisaac.tasks  # noqa: F401
from leisaac.devices.openarm_vr_v2.core import rotation_from_wxyz, rotation_to_wxyz
from leisaac.devices.openarm_vr_v2.isaac_qp_controller import OpenArmIsaacQPControllerV2


TASK = "LeIsaac-OpenArm-Bimanual-LiftCube-QuestV2-v0"
SIDES = ("left", "right")


def read_joints(robot, joint_ids):
    return {
        side: robot.data.joint_pos[0, joint_ids[side]].detach().cpu().numpy().astype(np.float32)
        for side in SIDES
    }


def read_hand_pose(robot, body_id):
    position, quaternion = subtract_frame_transforms(
        robot.data.root_pos_w[:1],
        robot.data.root_quat_w[:1],
        robot.data.body_pos_w[:1, body_id],
        robot.data.body_quat_w[:1, body_id],
    )
    return np.concatenate((position[0].cpu().numpy(), quaternion[0].cpu().numpy()))


def write_joints(env, robot, joint_ids, joints):
    for side in SIDES:
        position = torch.as_tensor(joints[side], dtype=torch.float32, device=env.device).unsqueeze(0)
        robot.write_joint_state_to_sim(
            position,
            torch.zeros_like(position),
            joint_ids=joint_ids[side],
        )
    env.sim.forward()


def converge(solver, side, target, initial, env, robot, joint_ids):
    measured = {active_side: initial[active_side].copy() for active_side in SIDES}
    solver.sync(measured["left"], measured["right"])
    write_joints(env, robot, joint_ids, measured)
    for _ in range(args_cli.iterations):
        command = solver.solve(side, target, measured["left"], measured["right"])
        if command is None:
            raise AssertionError(f"Isaac QP failed while checking {side} frame parity.")
        # Integrate the velocity-bounded drive target over one control step.
        measured[side] += (command - measured[side]) * ((1.0 / 60.0) / solver.joint_target_lookahead_s)
        write_joints(env, robot, joint_ids, measured)
    return measured


def assert_axis(label, vector, axis, minimum, cosine_limit):
    magnitude = float(np.linalg.norm(vector))
    cosine = float(vector[axis] / magnitude) if magnitude > 0.0 else -1.0
    if magnitude < minimum or cosine < cosine_limit:
        raise AssertionError(
            f"{label} axis mismatch: vector={vector.tolist()}, magnitude={magnitude:.5f}, cosine={cosine:.5f}."
        )
    return magnitude, cosine


def check_parity(env, robot, joint_ids, body_ids):
    initial = read_joints(robot, joint_ids)
    solver = OpenArmIsaacQPControllerV2(
        robot, initial["left"], initial["right"], control_hz=60.0
    )
    axis_names = ("x", "y", "z")
    results = []

    for side in SIDES:
        write_joints(env, robot, joint_ids, initial)
        initial_pose = read_hand_pose(robot, body_ids[side])
        initial_rotation = rotation_from_wxyz(initial_pose[3:])

        for axis, axis_name in enumerate(axis_names):
            write_joints(env, robot, joint_ids, initial)
            target = solver.fk(side)
            target[axis] += 0.03
            measured = converge(solver, side, target, initial, env, robot, joint_ids)
            write_joints(env, robot, joint_ids, measured)
            displacement = read_hand_pose(robot, body_ids[side])[:3] - initial_pose[:3]
            magnitude, cosine = assert_axis(
                f"{side} +{axis_name} translation", displacement, axis, 0.015, 0.90
            )
            results.append(f"{side} +{axis_name} {magnitude:.4f}m cos={cosine:.4f}")

        for axis, axis_name in enumerate(axis_names):
            write_joints(env, robot, joint_ids, initial)
            target = solver.fk(side)
            target_rotation = Rotation.from_rotvec(np.eye(3)[axis] * 0.12) * rotation_from_wxyz(target[3:])
            target[3:] = rotation_to_wxyz(target_rotation)
            measured = converge(solver, side, target, initial, env, robot, joint_ids)
            write_joints(env, robot, joint_ids, measured)
            actual_rotation = rotation_from_wxyz(read_hand_pose(robot, body_ids[side])[3:])
            rotation_vector = (actual_rotation * initial_rotation.inv()).as_rotvec()
            magnitude, cosine = assert_axis(
                f"{side} +{axis_name} rotation", rotation_vector, axis, 0.05, 0.80
            )
            results.append(f"{side} +r{axis_name} {magnitude:.4f}rad cos={cosine:.4f}")

    write_joints(env, robot, joint_ids, initial)
    print("frame parity: " + "; ".join(results))


def main():
    cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=1)
    cfg.use_teleop_device("quest3-controller-v2")
    cfg.recorders = None
    cfg.terminations.time_out = None
    cfg.terminations.success = None
    env = gym.make(TASK, cfg=cfg).unwrapped
    env.reset()
    robot = env.scene["robot"]
    joint_ids = {
        side: list(
            robot.find_joints(
                [f"openarm_{side}_joint{index}" for index in range(1, 8)],
                preserve_order=True,
            )[0]
        )
        for side in SIDES
    }
    body_ids = {
        side: robot.data.body_names.index(f"openarm_{side}_hand") for side in SIDES
    }
    try:
        check_parity(env, robot, joint_ids, body_ids)
        print("OpenArm V2 native QP/Isaac 6-DoF frame parity passed.")
    except Exception:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
