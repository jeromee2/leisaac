"""Collected Physics01 scene with its embedded bimanual OpenArm."""

from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass
from leisaac.assets.robots.openarm import (
    OPENARM_BIMANUAL_CONTROLLED_JOINT_PATTERNS,
    OPENARM_BIMANUAL_FEATURE_JOINT_NAMES,
    OPENARM_BIMANUAL_GRIPPER_JOINT_PATTERNS,
    get_openarm_bimanual_cfg,
)
from leisaac.enhance.envs.mdp import disable_rigid_body_gravity
from leisaac.tasks.template import (
    SingleArmObservationsCfg,
    SingleArmTaskEnvCfg,
    SingleArmTaskSceneCfg,
    SingleArmTerminationsCfg,
)
from leisaac.utils.env_utils import delete_attribute

from .openarm_bimanual_lift_cube_env_cfg import OpenArmBimanualActionsCfg

DEFAULT_PHYSICS01_USD_PATH = (
    Path(__file__).resolve().parents[5]
    / "assets/scenes/Collected_physics01/physics01.usd"
)
PHYSICS01_USD_PATH = str(
    Path(os.environ.get("LEISAAC_PHYSICS01_USD_PATH", DEFAULT_PHYSICS01_USD_PATH))
    .expanduser()
    .resolve()
)
PHYSICS01_GRIPPER_OPEN_POSITIONS = {"left": 0.75, "right": -0.75}
PHYSICS01_EE_BODY_NAMES = {
    "left": "openarm_left_ee_base_link",
    "right": "openarm_right_ee_base_link",
}


@configclass
class OpenArmBimanualPhysics01SceneCfg(SingleArmTaskSceneCfg):
    """Physics01 environment and its existing OpenArm articulation."""

    scene: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene",
        spawn=sim_utils.UsdFileCfg(
            usd_path=PHYSICS01_USD_PATH,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=4,
            ),
        ),
    )
    robot: ArticulationCfg = get_openarm_bimanual_cfg().replace(
        prim_path="{ENV_REGEX_NS}/Scene/openarm_v20",
        spawn=None,
        articulation_root_prim_path="/root_joint",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.00719338, 0.0, 0.34484773),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "openarm_left_joint[1-35-7]": 0.0,
                "openarm_left_joint4": 1.0,
                "openarm_right_joint[1-35-7]": 0.0,
                "openarm_right_joint4": 1.0,
                OPENARM_BIMANUAL_GRIPPER_JOINT_PATTERNS[
                    "left"
                ]: PHYSICS01_GRIPPER_OPEN_POSITIONS["left"],
                OPENARM_BIMANUAL_GRIPPER_JOINT_PATTERNS[
                    "right"
                ]: PHYSICS01_GRIPPER_OPEN_POSITIONS["right"],
            },
        ),
    )
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Scene/openarm_v20/world",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Scene/openarm_v20/openarm_left_ee_base_link",
                name=PHYSICS01_EE_BODY_NAMES["left"],
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Scene/openarm_v20/openarm_right_ee_base_link",
                name=PHYSICS01_EE_BODY_NAMES["right"],
            ),
        ],
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        delete_attribute(self, "wrist")
        delete_attribute(self, "front")


@configclass
class OpenArmBimanualPhysics01ObservationsCfg(SingleArmObservationsCfg):
    """Joint and end-effector observations without unused camera render products."""

    def __post_init__(self) -> None:
        super().__post_init__()
        delete_attribute(self.policy, "wrist")
        delete_attribute(self.policy, "front")


@configclass
class OpenArmBimanualPhysics01EnvCfg(SingleArmTaskEnvCfg):
    """Quest V2 teleoperation environment for the collected Physics01 scene."""

    scene: OpenArmBimanualPhysics01SceneCfg = OpenArmBimanualPhysics01SceneCfg(
        env_spacing=8.0
    )
    actions: OpenArmBimanualActionsCfg = OpenArmBimanualActionsCfg()
    observations: OpenArmBimanualPhysics01ObservationsCfg = (
        OpenArmBimanualPhysics01ObservationsCfg()
    )
    terminations: SingleArmTerminationsCfg = SingleArmTerminationsCfg()
    robot_name: str = "openarm_bimanual_physics01"
    dynamic_reset_gripper_effort_limit: bool = False
    default_feature_joint_names: list[str] = list(OPENARM_BIMANUAL_FEATURE_JOINT_NAMES)
    task_description: str = (
        "Teleoperate the bimanual OpenArm in the Physics01 laboratory scene."
    )
    openarm_ee_body_names: dict[str, str] = PHYSICS01_EE_BODY_NAMES
    gripper_open_positions: dict[str, float] = PHYSICS01_GRIPPER_OPEN_POSITIONS
    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, 0.0),
        anchor_rot=(0.70710678, 0.0, 0.0, -0.70710678),
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.viewer.eye = (1.2, -1.2, 1.2)
        self.viewer.lookat = (0.0, 0.0, 0.7)
        self.teleop_devices = DevicesCfg()
        self.events.disable_robot_gravity = EventTerm(
            func=disable_rigid_body_gravity,
            mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*")},
        )
        self.default_feature_joint_names = list(OPENARM_BIMANUAL_FEATURE_JOINT_NAMES)
        for term_name in (
            "joint_pos",
            "joint_vel",
            "joint_pos_rel",
            "joint_vel_rel",
            "joint_pos_target",
        ):
            term = getattr(self.observations.policy, term_name)
            term.params["asset_cfg"] = SceneEntityCfg(
                "robot",
                joint_names=list(OPENARM_BIMANUAL_CONTROLLED_JOINT_PATTERNS),
                preserve_order=True,
            )

    def use_teleop_device(self, teleop_device) -> None:
        if teleop_device != "quest3-controller-v2":
            raise ValueError("Physics01 currently supports only quest3-controller-v2.")
        super().use_teleop_device(teleop_device)
