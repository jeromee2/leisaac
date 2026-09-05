# Copyright (c) 2026, LeIsaac Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Differential IK action with redundant-joint posture and limit protection."""

from __future__ import annotations

import torch
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils import configclass


class PostureDifferentialInverseKinematicsAction(DifferentialInverseKinematicsAction):
    """Keep redundant joints near a preferred posture while tracking an end-effector pose."""

    cfg: PostureDifferentialInverseKinematicsActionCfg

    def __init__(self, cfg: PostureDifferentialInverseKinematicsActionCfg, env):
        super().__init__(cfg, env)
        if len(cfg.posture_target) != self._num_joints:
            raise ValueError(
                f"posture_target has {len(cfg.posture_target)} values, expected {self._num_joints}"
            )
        self._posture_target = torch.tensor(cfg.posture_target, device=self.device).repeat(self.num_envs, 1)

    def apply_actions(self) -> None:
        """Compute a bounded task-space step plus a null-space posture correction."""
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = self._to_torch(self._asset.data.joint_pos)[:, self._joint_ids]
        if ee_quat_curr.norm() == 0:
            joint_pos_des = joint_pos.clone()
        else:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des = self._ik_controller.compute(ee_pos_curr, ee_quat_curr, jacobian, joint_pos)
            joint_pos_des += self._compute_posture_step(jacobian, joint_pos)

        joint_delta = torch.clamp(
            joint_pos_des - joint_pos,
            min=-self.cfg.max_joint_step,
            max=self.cfg.max_joint_step,
        )
        joint_pos_des = joint_pos + joint_delta

        limits = self._to_torch(self._asset.data.soft_joint_pos_limits)[:, self._joint_ids]
        lower = limits[..., 0] + self.cfg.joint_limit_margin
        upper = limits[..., 1] - self.cfg.joint_limit_margin
        joint_pos_des = torch.maximum(torch.minimum(joint_pos_des, upper), lower)
        joint_pos_des = torch.where(torch.isfinite(joint_pos_des), joint_pos_des, joint_pos)
        if hasattr(self._asset, "set_joint_position_target_index"):
            self._asset.set_joint_position_target_index(target=joint_pos_des, joint_ids=self._joint_ids)
        else:
            self._asset.set_joint_position_target(joint_pos_des, self._joint_ids)

    @staticmethod
    def _to_torch(value):
        """Return a torch tensor from either legacy tensors or ProxyArray values."""
        if hasattr(value, "torch"):
            return value.torch
        return value

    def _compute_posture_step(self, jacobian: torch.Tensor, joint_pos: torch.Tensor) -> torch.Tensor:
        """Project a preferred-posture correction into the damped Jacobian null space."""
        lambda_val = float(self.cfg.controller.ik_params.get("lambda_val", 0.01))
        task_identity = torch.eye(jacobian.shape[1], device=self.device).expand(self.num_envs, -1, -1)
        joint_identity = torch.eye(self._num_joints, device=self.device).expand(self.num_envs, -1, -1)
        task_metric = jacobian @ jacobian.transpose(1, 2) + lambda_val**2 * task_identity
        jacobian_pinv = jacobian.transpose(1, 2) @ torch.linalg.solve(task_metric, task_identity)
        nullspace = joint_identity - jacobian_pinv @ jacobian
        posture_error = (self._posture_target - joint_pos).unsqueeze(-1)
        return self.cfg.posture_gain * (nullspace @ posture_error).squeeze(-1)


@configclass
class PostureDifferentialInverseKinematicsActionCfg(DifferentialInverseKinematicsActionCfg):
    """Configuration for posture-regularized differential inverse kinematics."""

    class_type: type[PostureDifferentialInverseKinematicsAction] = (
        PostureDifferentialInverseKinematicsAction
    )

    posture_target: tuple[float, ...] = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    """Preferred controlled-joint positions [rad]."""

    posture_gain: float = 0.08
    """Null-space posture correction gain per control step."""

    max_joint_step: float = 0.12
    """Maximum commanded joint-position change per control step [rad]."""

    joint_limit_margin: float = 0.02
    """Margin maintained inside each soft joint-position limit [rad]."""
