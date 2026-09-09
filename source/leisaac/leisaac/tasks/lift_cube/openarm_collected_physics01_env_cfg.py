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
    OPENARM_BIMANUAL_BASE_BODY_NAME,
    OPENARM_BIMANUAL_CONTROLLED_JOINT_PATTERNS,
    OPENARM_BIMANUAL_EE_BODY_NAMES,
    OPENARM_BIMANUAL_FEATURE_JOINT_NAMES,
    OPENARM_BIMANUAL_GRIPPER_JOINT_PATTERNS,
    OPENARM_GRIPPER_OPEN_POSITION,
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
from leisaac.utils.general_assets import parse_usd_and_create_subassets

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
PHYSICS01_V1_BACKGROUND_USD_PATH = str(
    DEFAULT_PHYSICS01_USD_PATH.with_name("physics01_v1_background.usda")
)
PHYSICS01_GRIPPER_OPEN_POSITIONS = {"left": 0.75, "right": -0.75}
PHYSICS01_EE_BODY_NAMES = {
    "left": "openarm_left_ee_base_link",
    "right": "openarm_right_ee_base_link",
}


@sim_utils.clone
def spawn_physics01(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Repair the collected gripper joint frames before PhysX creates the articulation."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    stage = prim.GetStage()
    for side in ("left", "right"):
        joint = UsdPhysics.RevoluteJoint(
            stage.GetPrimAtPath(f"{prim_path}/openarm_v20/joints/openarm_{side}_finger_joint1")
        )
        # Both frames change together: preserve the zero pose but reverse the X axis.
        # Equal joint commands then give symmetric finger motion.
        joint.GetLocalRot0Attr().Set(Gf.Quatf(0.0, 0.0, 0.0, 1.0))
        joint.GetLocalRot1Attr().Set(Gf.Quatf(0.0, 0.0, 0.0, 1.0))
        follower = stage.GetPrimAtPath(
            f"{prim_path}/openarm_v20/joints/openarm_{side}_finger_joint2"
        )
        # Both finger joints have position drives; do not also constrain the follower.
        follower.RemoveAPI(PhysxSchema.PhysxMimicJointAPI, "rotX")
    return prim


@configclass
class OpenArmBimanualPhysics01SceneCfg(SingleArmTaskSceneCfg):
    """Physics01 environment and its existing OpenArm articulation."""

    scene: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene",
        spawn=sim_utils.UsdFileCfg(
            func=spawn_physics01,
            usd_path=PHYSICS01_USD_PATH,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=8,
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
            # Symmetric attention pose: arms down and clear of the table edge.
            joint_pos={
                "openarm_left_joint[235-7]": 0.0,
                "openarm_left_joint1": 0.6,
                "openarm_left_joint4": 0.4,
                "openarm_right_joint[235-7]": 0.0,
                "openarm_right_joint1": -0.6,
                "openarm_right_joint4": 0.4,
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
        # Physics01 fingers are revolute, unlike the official prismatic gripper.
        gripper = self.robot.actuators["openarm_gripper"]
        gripper.velocity_limit_sim = 2.0
        gripper.effort_limit_sim = 5.0
        gripper.stiffness = 100.0
        gripper.damping = 10.0


@configclass
class OpenArmBimanualPhysics01V1SceneCfg(SingleArmTaskSceneCfg):
    """Physics01 background with a separately spawned OpenArm v1.0."""

    scene: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene",
        spawn=sim_utils.UsdFileCfg(usd_path=PHYSICS01_V1_BACKGROUND_USD_PATH),
    )
    robot: ArticulationCfg = get_openarm_bimanual_cfg(local_only=True).replace(
        prim_path="{ENV_REGEX_NS}/Scene/openarm_v1",
        init_state=ArticulationCfg.InitialStateCfg(
            # V1 and the embedded V2 share the same torso and shoulder origins.
            pos=(-0.00719338, 0.0, 0.34484773),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "openarm_left_joint[235-7]": 0.0,
                "openarm_left_joint1": 0.6,
                "openarm_left_joint4": 0.6,
                "openarm_right_joint[235-7]": 0.0,
                "openarm_right_joint1": -0.6,
                "openarm_right_joint4": 0.6,
                OPENARM_BIMANUAL_GRIPPER_JOINT_PATTERNS[
                    "left"
                ]: OPENARM_GRIPPER_OPEN_POSITION,
                OPENARM_BIMANUAL_GRIPPER_JOINT_PATTERNS[
                    "right"
                ]: OPENARM_GRIPPER_OPEN_POSITION,
            },
        ),
    )
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Scene/openarm_v1/{OPENARM_BIMANUAL_BASE_BODY_NAME}",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=(
                    f"{{ENV_REGEX_NS}}/Scene/openarm_v1/{OPENARM_BIMANUAL_EE_BODY_NAMES['left']}"
                ),
                name=OPENARM_BIMANUAL_EE_BODY_NAMES["left"],
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path=(
                    f"{{ENV_REGEX_NS}}/Scene/openarm_v1/{OPENARM_BIMANUAL_EE_BODY_NAMES['right']}"
                ),
                name=OPENARM_BIMANUAL_EE_BODY_NAMES["right"],
            ),
        ],
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        delete_attribute(self, "wrist")
        delete_attribute(self, "front")
        self.robot.spawn.articulation_props.enabled_self_collisions = False


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
        parse_usd_and_create_subassets(
            PHYSICS01_USD_PATH, self, specific_name_list=["/World/Exp/", "/World/Table"]
        )
        self.viewer.eye = (1.2, -1.2, 1.2)
        self.sim.dt = 1.0 / 120.0
        self.decimation = 2
        self.sim.render_interval = self.decimation
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


@configclass
class OpenArmBimanualPhysics01V1EnvCfg(OpenArmBimanualPhysics01EnvCfg):
    """Physics01 Quest V2 teleoperation with OpenArm hardware model v1.0."""

    scene: OpenArmBimanualPhysics01V1SceneCfg = OpenArmBimanualPhysics01V1SceneCfg(
        env_spacing=8.0
    )
    robot_name: str = "openarm_bimanual_v1_0_physics01"
    openarm_ee_body_names: dict[str, str] = dict(OPENARM_BIMANUAL_EE_BODY_NAMES)
    gripper_open_positions: dict[str, float] = {
        "left": OPENARM_GRIPPER_OPEN_POSITION,
        "right": OPENARM_GRIPPER_OPEN_POSITION,
    }
