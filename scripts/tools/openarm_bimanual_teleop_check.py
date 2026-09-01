"""Headless reachability regression check for OpenArm bimanual Quest teleoperation."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=60, help="Simulation steps per probe.")
parser.add_argument("--command", type=float, default=0.05, help="Relative SE(3) command per step.")
parser.add_argument("--min_translation", type=float, default=0.15, help="Minimum hand translation in meters.")
parser.add_argument("--min_rotation", type=float, default=0.10, help="Minimum hand rotation in radians.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.utils.math import quat_error_magnitude
from isaaclab_tasks.utils import parse_env_cfg

import leisaac.tasks  # noqa: F401


TASK = "LeIsaac-OpenArm-Bimanual-LiftCube-v0"


def probe(env, robot, left_body_id, right_body_id, name, action_index, left_sign, right_sign, minimum, rotation):
    """Apply one relative command direction and assert both hands respond."""
    env.reset()
    start_pos = torch.stack((robot.data.body_pos_w[0, left_body_id], robot.data.body_pos_w[0, right_body_id])).clone()
    start_quat = torch.stack((robot.data.body_quat_w[0, left_body_id], robot.data.body_quat_w[0, right_body_id])).clone()
    action = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
    action[:, action_index] = args_cli.command * left_sign
    action[:, 7 + action_index] = args_cli.command * right_sign
    action[:, 6] = 1.0
    action[:, 13] = 1.0
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


def main():
    cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=1)
    cfg.seed = 42
    cfg.use_teleop_device("quest3-controller")
    env = gym.make(TASK, cfg=cfg).unwrapped
    robot = env.scene["robot"]
    if env.action_manager.total_action_dim != 14:
        raise AssertionError(f"Expected 14D bimanual action, got {env.action_manager.total_action_dim}")
    left_body_id = robot.data.body_names.index("openarm_left_hand")
    right_body_id = robot.data.body_names.index("openarm_right_hand")
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
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
