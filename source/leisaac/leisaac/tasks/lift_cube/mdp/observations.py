import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer


def object_grasped(
    env: ManagerBasedRLEnv | DirectRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    diff_threshold: float = 0.02,
    grasp_threshold: float = 0.26,
) -> torch.Tensor:
    """Check if an object is grasped by the specified robot."""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_pos = object.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 1, :]
    pos_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)

    grasped = torch.logical_and(pos_diff < diff_threshold, robot.data.joint_pos[:, -1] < grasp_threshold)

    return grasped


def object_grasped_by_either_hand(
    env: ManagerBasedRLEnv | DirectRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    gripper_joint_names: tuple[str, str],
    ee_target_indices: tuple[int, int] = (0, 1),
    diff_threshold: float = 0.03,
    closed_threshold: float = 0.02,
) -> torch.Tensor:
    """Return whether either OpenArm hand is closed around the object."""

    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    grasped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for target_index, joint_name in zip(ee_target_indices, gripper_joint_names, strict=True):
        joint_index = robot.data.joint_names.index(joint_name)
        distance = torch.linalg.vector_norm(
            object.data.root_pos_w - ee_frame.data.target_pos_w[:, target_index, :], dim=1
        )
        grasped.logical_or_(
            torch.logical_and(distance < diff_threshold, robot.data.joint_pos[:, joint_index] < closed_threshold)
        )
    return grasped
