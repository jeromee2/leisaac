"""OpenArm v1.0 bimanual LiftCube environment for Quest 3 teleoperation."""

from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.devices.openxr.openxr_device import OpenXRDevice, OpenXRDeviceCfg
from isaaclab.devices.openxr.retargeters.manipulator.gripper_retargeter import (
    GripperRetargeterCfg,
)
from isaaclab.devices.openxr.retargeters.manipulator.se3_rel_retargeter import (
    Se3RelRetargeterCfg,
)
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
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

from . import mdp
from .lift_cube_env_cfg import LiftCubeEnvCfg, LiftCubeSceneCfg, TerminationsCfg


@configclass
class OpenArmBimanualActionsCfg:
    """Per-device left arm/gripper actions followed by right arm/gripper actions."""

    left_arm_action: mdp.ActionTermCfg = MISSING
    left_gripper_action: mdp.ActionTermCfg = MISSING
    right_arm_action: mdp.ActionTermCfg = MISSING
    right_gripper_action: mdp.ActionTermCfg = MISSING


@configclass
class OpenArmBimanualLiftCubeSceneCfg(LiftCubeSceneCfg):
    """LiftCube scene with the official OpenArm bimanual articulation."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.robot = get_openarm_bimanual_cfg().replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                # Center the torso between the two hands and the table workspace.
                pos=(0.35, -0.42, 0.01),
                rot=(1.0, 0.0, 0.0, 0.0),
                joint_pos={
                    "openarm_left_joint[1-35-7]": 0.0,
                    "openarm_left_joint4": 1.0,
                    "openarm_right_joint[1-35-7]": 0.0,
                    "openarm_right_joint4": 1.0,
                    OPENARM_BIMANUAL_GRIPPER_JOINT_PATTERNS["left"]: OPENARM_GRIPPER_OPEN_POSITION,
                    OPENARM_BIMANUAL_GRIPPER_JOINT_PATTERNS["right"]: OPENARM_GRIPPER_OPEN_POSITION,
                },
            ),
        )
        self.front.prim_path = "{ENV_REGEX_NS}/Robot/openarm_body_link/front_camera"
        self.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_body_link",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/openarm_left_hand",
                    name=OPENARM_BIMANUAL_EE_BODY_NAMES["left"],
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/openarm_right_hand",
                    name=OPENARM_BIMANUAL_EE_BODY_NAMES["right"],
                ),
            ],
        )


@configclass
class OpenArmBimanualTerminationsCfg(TerminationsCfg):
    """Lift success relative to the fixed OpenArm torso base."""

    success = DoneTerm(
        func=mdp.cube_height_above_base,
        params={
            "cube_cfg": SceneEntityCfg("cube"),
            "robot_cfg": SceneEntityCfg("robot"),
            "robot_base_name": OPENARM_BIMANUAL_BASE_BODY_NAME,
            "height_threshold": 0.20,
        },
    )


@configclass
class OpenArmBimanualLiftCubeEnvCfg(LiftCubeEnvCfg):
    """Dual-hand Quest teleoperation environment for OpenArm v1.0."""

    scene: OpenArmBimanualLiftCubeSceneCfg = OpenArmBimanualLiftCubeSceneCfg(env_spacing=8.0)
    actions: OpenArmBimanualActionsCfg = OpenArmBimanualActionsCfg()
    terminations: OpenArmBimanualTerminationsCfg = OpenArmBimanualTerminationsCfg()
    robot_name: str = "openarm_bimanual_v1_0"
    dynamic_reset_gripper_effort_limit: bool = False
    # Rotate the whole simulation 90 degrees right in XR so the table faces the viewer.
    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, 0.0),
        anchor_rot=(0.70710678, 0.0, 0.0, -0.70710678),
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        # LiftCube's single-arm parent applies its own base placement; restore
        # the torso-centred bimanual placement afterwards.
        self.scene.robot.init_state.pos = (0.35, -0.42, 0.01)
        self.dynamic_reset_gripper_effort_limit = False
        self.default_feature_joint_names = list(OPENARM_BIMANUAL_FEATURE_JOINT_NAMES)

        self.observations.subtask_terms.pick_cube.func = mdp.object_grasped_by_either_hand
        self.observations.subtask_terms.pick_cube.params = {
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "object_cfg": SceneEntityCfg("cube"),
            "gripper_joint_names": (
                "openarm_left_finger_joint1",
                "openarm_right_finger_joint1",
            ),
        }

        for term_name in ("joint_pos", "joint_vel", "joint_pos_rel", "joint_vel_rel", "joint_pos_target"):
            term = getattr(self.observations.policy, term_name)
            term.params["asset_cfg"] = SceneEntityCfg(
                "robot",
                joint_names=list(OPENARM_BIMANUAL_CONTROLLED_JOINT_PATTERNS),
                preserve_order=True,
            )

        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        Se3RelRetargeterCfg(
                            bound_hand=OpenXRDevice.TrackingTarget.HAND_LEFT,
                            use_wrist_rotation=True,
                            use_wrist_position=True,
                            delta_pos_scale_factor=20.0,
                            delta_rot_scale_factor=10.0,
                            alpha_pos=0.35,
                            alpha_rot=0.25,
                            sim_device=self.sim.device,
                        ),
                        GripperRetargeterCfg(
                            bound_hand=OpenXRDevice.TrackingTarget.HAND_LEFT,
                            sim_device=self.sim.device,
                        ),
                        Se3RelRetargeterCfg(
                            bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT,
                            use_wrist_rotation=True,
                            use_wrist_position=True,
                            delta_pos_scale_factor=20.0,
                            delta_rot_scale_factor=10.0,
                            alpha_pos=0.35,
                            alpha_rot=0.25,
                            sim_device=self.sim.device,
                        ),
                        GripperRetargeterCfg(
                            bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT,
                            sim_device=self.sim.device,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            }
        )

    def use_teleop_device(self, teleop_device) -> None:
        super().use_teleop_device(teleop_device)
        if teleop_device == "quest3-controller-v2" and self.scene.robot.spawn is not None:
            # The official bimanual USD reports a false left-arm self-contact at the
            # neutral pose, producing non-zero joint velocity under a fixed target.
            self.scene.robot.spawn.articulation_props.enabled_self_collisions = False
