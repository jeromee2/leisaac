"""Headless integration check for OpenArm Quest teleoperation V2."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--motion_steps", type=int, default=120)
parser.add_argument("--joint_target_lookahead_s", type=float, default=0.08)
parser.add_argument("--settle_steps", type=int, default=120)
parser.add_argument("--task", default="LeIsaac-OpenArm-Bimanual-Physics01-QuestV2-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import gymnasium as gym
import leisaac.tasks  # noqa: F401
import numpy as np
import torch
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.devices.openarm_vr_v2.isaac_qp_controller import (
    EXPECTED_JOINT_LIMITS_RAD,
    JOINT_VELOCITY_LIMITS_RAD_S,
    OpenArmIsaacQPControllerV2,
)


def read_joints(robot, joint_ids):
    return {
        side: robot.data.joint_pos[0, joint_ids[side]]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
        for side in ("left", "right")
    }


def pack_action(commands, device, gripper_commands=(1.0, 1.0)):
    values = np.concatenate(
        (
            commands["left"],
            [gripper_commands[0]],
            commands["right"],
            [gripper_commands[1]],
        )
    ).astype(np.float32)
    return torch.as_tensor(values, device=device).unsqueeze(0)


def check_schema(env, robot, joint_ids):
    assert env.action_manager.total_action_dim == 16
    term_types = [type(term).__name__ for term in env.action_manager._terms.values()]
    assert term_types == [
        "JointPositionAction",
        "BinaryJointPositionAction",
        "JointPositionAction",
        "BinaryJointPositionAction",
    ]
    for side in ("left", "right"):
        actual = (
            robot.data.soft_joint_pos_limits[0, joint_ids[side]].detach().cpu().numpy()
        )
        np.testing.assert_allclose(
            actual, EXPECTED_JOINT_LIMITS_RAD[side], atol=1e-4, rtol=0.0
        )
    print("schema: 16D absolute joint action and verified joint limits passed")


def check_hold(env, robot, joint_ids):
    initial = read_joints(robot, joint_ids)
    action = pack_action(initial, env.device)
    velocities = []
    for step in range(args_cli.settle_steps):
        env.step(action)
        if step >= args_cli.settle_steps // 2:
            ids = joint_ids["left"] + joint_ids["right"]
            velocities.append(
                robot.data.joint_vel[0, ids].detach().cpu().numpy().copy()
            )
    final = read_joints(robot, joint_ids)
    error = max(
        float(np.max(np.abs(final[side] - initial[side]))) for side in ("left", "right")
    )
    velocity_samples = np.asarray(velocities)
    rms_velocity = float(np.sqrt(np.mean(np.square(velocity_samples))))
    if not np.isfinite(error) or not np.isfinite(rms_velocity):
        raise AssertionError("Non-finite joint state in hold test.")
    if rms_velocity > 0.02:
        per_joint_rms = np.sqrt(np.mean(np.square(velocity_samples), axis=0))
        joint_names = [
            robot.data.joint_names[index]
            for index in joint_ids["left"] + joint_ids["right"]
        ]
        detail = dict(zip(joint_names, per_joint_rms.tolist(), strict=True))
        raise AssertionError(
            f"Stationary joint RMS velocity {rms_velocity:.5f}rad/s exceeds 0.02rad/s: {detail}"
        )
    print(
        f"hold: max_initial_error={error:.5f}rad, settled_rms_velocity={rms_velocity:.5f}rad/s"
    )


def check_grippers(env, robot, joint_ids):
    arm_commands = read_joints(robot, joint_ids)
    finger_ids = {
        side: robot.find_joints(
            [f"openarm_{side}_finger_joint{index}" for index in range(1, 3)],
            preserve_order=True,
        )[0]
        for side in ("left", "right")
    }

    def positions():
        return {
            side: robot.data.joint_pos[0, finger_ids[side]]
            .detach()
            .cpu()
            .numpy()
            .copy()
            for side in ("left", "right")
        }

    opened = positions()
    for _ in range(120):
        env.step(pack_action(arm_commands, env.device, (-1.0, -1.0)))
    closed = positions()
    for side in ("left", "right"):
        close_delta = np.abs(opened[side]) - np.abs(closed[side])
        if np.any(close_delta < np.maximum(0.01, 0.2 * np.abs(opened[side]))):
            raise AssertionError(
                f"{side} gripper did not close: {opened[side]} -> {closed[side]}"
            )

    for _ in range(120):
        env.step(pack_action(arm_commands, env.device))
    reopened = positions()
    for side in ("left", "right"):
        reopen_delta = np.abs(reopened[side]) - np.abs(closed[side])
        if np.any(reopen_delta < np.maximum(0.01, 0.2 * np.abs(opened[side]))):
            raise AssertionError(
                f"{side} gripper did not reopen: {closed[side]} -> {reopened[side]}"
            )
    print(f"grippers: close/reopen passed; closed={closed}, reopened={reopened}")


def check_native_qp_motion(env, robot, joint_ids, body_ids, ee_body_names):
    measured = read_joints(robot, joint_ids)
    solver = OpenArmIsaacQPControllerV2(
        robot,
        measured["left"],
        measured["right"],
        control_hz=60.0,
        ee_body_names=ee_body_names,
        joint_target_lookahead_s=args_cli.joint_target_lookahead_s,
    )
    targets = {side: solver.fk(side) for side in ("left", "right")}
    for target in targets.values():
        target[0] += 0.05
    start_positions = {
        side: robot.data.body_pos_w[0, body_ids[side]].detach().cpu().numpy().copy()
        for side in ("left", "right")
    }
    max_target_lead_ratio = 0.0
    max_physical_velocity_ratio = 0.0
    physical_velocities = []
    steps_to_90_percent = {side: None for side in ("left", "right")}
    for step_index in range(args_cli.motion_steps):
        measured = read_joints(robot, joint_ids)
        commands = {}
        for side in ("left", "right"):
            command = solver.solve(
                side, targets[side], measured["left"], measured["right"]
            )
            if command is None:
                raise AssertionError(f"Isaac QP failed on reachable {side} target.")
            target_lead = np.abs(command - measured[side])
            max_target_lead_ratio = max(
                max_target_lead_ratio,
                float(
                    np.max(
                        target_lead
                        / (
                            JOINT_VELOCITY_LIMITS_RAD_S
                            * solver.joint_target_lookahead_s
                        )
                    )
                ),
            )
            commands[side] = command
        env.step(pack_action(commands, env.device))
        step_velocities = []
        for side in ("left", "right"):
            physical_velocity = (
                robot.data.joint_vel[0, joint_ids[side]].detach().cpu().numpy().copy()
            )
            step_velocities.append(physical_velocity)
            max_physical_velocity_ratio = max(
                max_physical_velocity_ratio,
                float(np.max(np.abs(physical_velocity) / JOINT_VELOCITY_LIMITS_RAD_S)),
            )
            displacement_x = float(
                robot.data.body_pos_w[0, body_ids[side], 0] - start_positions[side][0]
            )
            if displacement_x >= 0.045 and steps_to_90_percent[side] is None:
                steps_to_90_percent[side] = step_index + 1
        physical_velocities.append(step_velocities)

    movements = {
        side: float(
            torch.linalg.vector_norm(
                robot.data.body_pos_w[0, body_ids[side]]
                - torch.as_tensor(start_positions[side], device=env.device)
            )
        )
        for side in ("left", "right")
    }
    if min(movements.values()) < 0.02:
        raise AssertionError(
            f"Reachable 5cm target produced too little hand motion: {movements}."
        )
    if max_target_lead_ratio > 1.001:
        raise AssertionError(
            f"Isaac QP target lookahead cap exceeded by ratio {max_target_lead_ratio:.4f}."
        )
    if max_physical_velocity_ratio > 1.05:
        raise AssertionError(
            f"Isaac physical joint speed cap exceeded by ratio {max_physical_velocity_ratio:.4f}."
        )
    settled_rms = float(np.sqrt(np.mean(np.square(physical_velocities[-30:]))))
    if args_cli.motion_steps >= 120 and settled_rms > 0.02:
        raise AssertionError(
            f"Post-motion joint RMS velocity {settled_rms:.5f}rad/s exceeds 0.02rad/s."
        )

    current = read_joints(robot, joint_ids)
    bad_target = targets["left"].copy()
    bad_target[0] = np.nan
    if solver.solve("left", bad_target, current["left"], current["right"]) is not None:
        raise AssertionError("Left solver accepted a non-finite target.")
    right_hold = solver.fk("right")
    if solver.solve("right", right_hold, current["left"], current["right"]) is None:
        raise AssertionError(
            "Right solver stopped after an independent left-side failure."
        )
    print(
        f"motion: left={movements['left']:.4f}m, right={movements['right']:.4f}m, "
        f"target_lead_ratio={max_target_lead_ratio:.4f}, "
        f"physical_velocity_ratio={max_physical_velocity_ratio:.4f}, "
        f"steps_to_90_percent={steps_to_90_percent}, settled_rms={settled_rms:.5f}rad/s; "
        "independent failure passed"
    )


def main():
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    cfg.use_teleop_device("quest3-controller-v2")
    cfg.recorders = None
    cfg.terminations.time_out = None
    if hasattr(cfg.terminations, "success"):
        cfg.terminations.success = None
    env = gym.make(args_cli.task, cfg=cfg).unwrapped
    env.reset()
    robot = env.scene["robot"]
    joint_ids = {}
    for side in ("left", "right"):
        names = [f"openarm_{side}_joint{index}" for index in range(1, 8)]
        ids, _ = robot.find_joints(names, preserve_order=True)
        joint_ids[side] = list(ids)
    ee_body_names = getattr(
        cfg,
        "openarm_ee_body_names",
        {side: f"openarm_{side}_hand" for side in ("left", "right")},
    )
    body_ids = {
        side: robot.data.body_names.index(ee_body_names[side])
        for side in ("left", "right")
    }
    try:
        check_schema(env, robot, joint_ids)
        check_hold(env, robot, joint_ids)
        check_grippers(env, robot, joint_ids)
        check_native_qp_motion(env, robot, joint_ids, body_ids, ee_body_names)
        print(f"OpenArm Quest V2 headless integration checks passed: {args_cli.task}")
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
