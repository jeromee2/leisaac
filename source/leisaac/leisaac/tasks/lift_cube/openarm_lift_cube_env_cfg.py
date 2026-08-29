"""OpenArm 1.1 variant of the LeIsaac LiftCube teleoperation environment."""

from isaaclab.assets import ArticulationCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass

from leisaac.assets.robots.openarm import (
    OPENARM_BASE_BODY_NAME,
    OPENARM_CONTROLLED_JOINT_PATTERNS,
    OPENARM_EE_BODY_NAME,
    OPENARM_GRIPPER_OPEN_POSITION,
    OPENARM_TCP_FRAME_NAME,
    get_openarm_unimanual_cfg,
)

from . import mdp
from .lift_cube_env_cfg import LiftCubeEnvCfg, LiftCubeSceneCfg, TerminationsCfg


@configclass
class OpenArmLiftCubeSceneCfg(LiftCubeSceneCfg):
    """LiftCube scene using the official Isaac Lab OpenArm unimanual USD."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.robot = get_openarm_unimanual_cfg().replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.35, -0.64, 0.01),
                # Isaac Lab uses (w, x, y, z) quaternions; identity is w=1.
                rot=(1.0, 0.0, 0.0, 0.0),
                joint_pos={
                    "openarm_joint1": 1.57,
                    "openarm_joint2": 0.0,
                    "openarm_joint3": -1.57,
                    "openarm_joint4": 1.57,
                    "openarm_joint5": 0.0,
                    "openarm_joint6": 0.0,
                    "openarm_joint7": 0.0,
                    "openarm_finger_joint.*": OPENARM_GRIPPER_OPEN_POSITION,
                },
            ),
        )
        self.front.prim_path = "{ENV_REGEX_NS}/Robot/openarm_link0/front_camera"
        self.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_link0",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/openarm_hand",
                    name=OPENARM_EE_BODY_NAME,
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/openarm_ee_tcp",
                    name=OPENARM_TCP_FRAME_NAME,
                ),
            ],
        )


@configclass
class OpenArmTerminationsCfg(TerminationsCfg):
    """LiftCube success condition with the OpenArm base body mapping."""

    success = DoneTerm(
        func=mdp.cube_height_above_base,
        params={
            "cube_cfg": SceneEntityCfg("cube"),
            "robot_cfg": SceneEntityCfg("robot"),
            "robot_base_name": OPENARM_BASE_BODY_NAME,
            "height_threshold": 0.20,
        },
    )


@configclass
class OpenArmLiftCubeEnvCfg(LiftCubeEnvCfg):
    """VR teleoperation environment for an OpenArm 1.1 unimanual robot."""

    scene: OpenArmLiftCubeSceneCfg = OpenArmLiftCubeSceneCfg(env_spacing=8.0)
    terminations: OpenArmTerminationsCfg = OpenArmTerminationsCfg()
    robot_name: str = "openarm_1_1"
    dynamic_reset_gripper_effort_limit: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        # OpenArm uses its own actuator/gripper limits; the SO-101 mass-based
        # effort rewrite is intentionally disabled for this environment.
        self.dynamic_reset_gripper_effort_limit = False

        # The OpenArm USD also contains two fixed joints for the hand and TCP.
        # Keep those out of policy/dataset joint observations: only the seven
        # actuated arm joints and the two prismatic finger joints are state.
        for term_name in ("joint_pos", "joint_vel", "joint_pos_rel", "joint_vel_rel", "joint_pos_target"):
            term = getattr(self.observations.policy, term_name)
            term.params["asset_cfg"] = SceneEntityCfg(
                "robot", joint_names=list(OPENARM_CONTROLLED_JOINT_PATTERNS)
            )
