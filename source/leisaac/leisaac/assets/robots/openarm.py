"""OpenArm v1.0 hardware configuration used by LeIsaac teleoperation tasks.

The OpenArm USD is the Isaac Lab-maintained Isaac Sim 5.1 Nucleus asset. This
small adapter keeps the compatible ArticulationCfg local so LeIsaac works with
the dependency checkout even when ``isaaclab_assets`` is not installed.

The robot is the OpenArm v1.0 hardware description. The ``DM-J4310-2EC V1.1``
designation applies to the motor used on joints 5--7, not to a separate robot
model.
"""

from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from leisaac.utils.constant import ASSETS_ROOT


OPENARM_ARM_JOINT_NAMES = [f"openarm_joint{index}" for index in range(1, 8)]
OPENARM_GRIPPER_JOINT_PATTERN = "openarm_finger_joint.*"
OPENARM_CONTROLLED_JOINT_PATTERNS = ["openarm_joint[1-7]", OPENARM_GRIPPER_JOINT_PATTERN]
OPENARM_FEATURE_JOINT_NAMES = [
    *(f"{joint_name}.pos" for joint_name in OPENARM_ARM_JOINT_NAMES),
    "openarm_finger_joint1.pos",
    "openarm_finger_joint2.pos",
]
OPENARM_EE_BODY_NAME = "openarm_hand"
OPENARM_TCP_FRAME_NAME = "openarm_ee_tcp"
OPENARM_BASE_BODY_NAME = "openarm_link0"
OPENARM_GRIPPER_OPEN_POSITION = 0.044
OPENARM_GRIPPER_CLOSED_POSITION = 0.0
OPENARM_BIMANUAL_SIDES = ("left", "right")
OPENARM_BIMANUAL_ARM_JOINT_NAMES = {
    side: [f"openarm_{side}_joint{index}" for index in range(1, 8)]
    for side in OPENARM_BIMANUAL_SIDES
}
OPENARM_BIMANUAL_GRIPPER_JOINT_PATTERNS = {
    side: f"openarm_{side}_finger_joint.*" for side in OPENARM_BIMANUAL_SIDES
}
OPENARM_BIMANUAL_CONTROLLED_JOINT_PATTERNS = [
    *(f"openarm_{side}_joint[1-7]" for side in OPENARM_BIMANUAL_SIDES),
    *(OPENARM_BIMANUAL_GRIPPER_JOINT_PATTERNS[side] for side in OPENARM_BIMANUAL_SIDES),
]
OPENARM_BIMANUAL_FEATURE_JOINT_NAMES = [
    *(
        f"{joint_name}.pos"
        for side in OPENARM_BIMANUAL_SIDES
        for joint_name in OPENARM_BIMANUAL_ARM_JOINT_NAMES[side]
    ),
    *(
        f"openarm_{side}_finger_joint{index}.pos"
        for side in OPENARM_BIMANUAL_SIDES
        for index in range(1, 3)
    ),
]
OPENARM_BIMANUAL_EE_BODY_NAMES = {
    side: f"openarm_{side}_hand" for side in OPENARM_BIMANUAL_SIDES
}
OPENARM_BIMANUAL_BASE_BODY_NAME = "openarm_body_link"
# Source-of-truth v1.0/v1.1 mechanical limits from Enactic's
# openarm_description/assets/robot/openarm_v1.0/config/arm/joint_limits.yaml.
# The generated USD authors the same limits; this table is exposed for smoke
# tests and for code that needs to validate teleoperation targets.
OPENARM_V10_JOINT_LIMITS_RAD = {
    "openarm_joint1": (-1.396263, 3.490659),
    "openarm_joint2": (-1.745329, 1.745329),
    "openarm_joint3": (-1.570796, 1.570796),
    "openarm_joint4": (0.0, 2.443461),
    "openarm_joint5": (-1.570796, 1.570796),
    "openarm_joint6": (-0.785398, 0.785398),
    "openarm_joint7": (-1.570796, 1.570796),
}
OPENARM_LOCAL_USD_PATH = Path(ASSETS_ROOT) / "robots" / "openarm_v1.0" / "openarm_unimanual.usd"
OPENARM_BIMANUAL_LOCAL_USD_PATH = Path(ASSETS_ROOT) / "robots" / "openarm_v1.0" / "openarm_bimanual.usd"


def _openarm_usd_path() -> str:
    """Resolve a reproducibly imported local USD, with Nucleus fallback.

    ``LEISAAC_OPENARM_USD_PATH`` is useful on air-gapped machines. If unset,
    the checked-in conversion location is preferred when present; otherwise
    the official Isaac Sim 5.1 Nucleus asset is used.
    """

    configured_path = os.environ.get("LEISAAC_OPENARM_USD_PATH")
    if configured_path:
        return str(Path(configured_path).expanduser().resolve())
    if OPENARM_LOCAL_USD_PATH.is_file():
        return str(OPENARM_LOCAL_USD_PATH)
    return f"{ISAAC_NUCLEUS_DIR}/Robots/OpenArm/openarm_unimanual/openarm_unimanual.usd"


def _openarm_bimanual_usd_path() -> str:
    """Resolve the official bimanual OpenArm USD with a local override."""

    configured_path = os.environ.get("LEISAAC_OPENARM_BIMANUAL_USD_PATH")
    if configured_path:
        return str(Path(configured_path).expanduser().resolve())
    if OPENARM_BIMANUAL_LOCAL_USD_PATH.is_file():
        return str(OPENARM_BIMANUAL_LOCAL_USD_PATH)
    return f"{ISAAC_NUCLEUS_DIR}/Robots/OpenArm/openarm_bimanual/openarm_bimanual.usd"


OPENARM_BI_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_openarm_bimanual_usd_path(),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "openarm_left_joint.*": 0.0,
            "openarm_right_joint.*": 0.0,
            "openarm_left_finger_joint.*": OPENARM_GRIPPER_OPEN_POSITION,
            "openarm_right_finger_joint.*": OPENARM_GRIPPER_OPEN_POSITION,
        },
    ),
    actuators={
        "openarm_arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "openarm_left_joint[1-7]",
                "openarm_right_joint[1-7]",
            ],
            velocity_limit_sim={
                "openarm_left_joint[1-2]": 2.175,
                "openarm_right_joint[1-2]": 2.175,
                "openarm_left_joint[3-4]": 2.175,
                "openarm_right_joint[3-4]": 2.175,
                "openarm_left_joint[5-7]": 2.61,
                "openarm_right_joint[5-7]": 2.61,
            },
            effort_limit_sim={
                "openarm_left_joint[1-2]": 40.0,
                "openarm_right_joint[1-2]": 40.0,
                "openarm_left_joint[3-4]": 27.0,
                "openarm_right_joint[3-4]": 27.0,
                "openarm_left_joint[5-7]": 7.0,
                "openarm_right_joint[5-7]": 7.0,
            },
            stiffness=80.0,
            damping=4.0,
        ),
        "openarm_gripper": ImplicitActuatorCfg(
            joint_names_expr=[
                "openarm_left_finger_joint.*",
                "openarm_right_finger_joint.*",
            ],
            velocity_limit_sim=0.2,
            effort_limit_sim=333.33,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

OPENARM_BI_HIGH_PD_CFG = OPENARM_BI_CFG.copy()
OPENARM_BI_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
OPENARM_BI_HIGH_PD_CFG.actuators["openarm_arm"].stiffness = 400.0
OPENARM_BI_HIGH_PD_CFG.actuators["openarm_arm"].damping = 80.0


OPENARM_UNI_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # Official Isaac Sim 5.1 / Isaac Lab 2.3 OpenArm unimanual asset.
        usd_path=_openarm_usd_path(),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "openarm_joint1": 1.57,
            "openarm_joint2": 0.0,
            "openarm_joint3": -1.57,
            "openarm_joint4": 1.57,
            "openarm_joint5": 0.0,
            "openarm_joint6": 0.0,
            "openarm_joint7": 0.0,
            OPENARM_GRIPPER_JOINT_PATTERN: OPENARM_GRIPPER_OPEN_POSITION,
        },
    ),
    actuators={
        "openarm_arm": ImplicitActuatorCfg(
            joint_names_expr=["openarm_joint[1-7]"],
            velocity_limit_sim={
                "openarm_joint[1-2]": 2.175,
                "openarm_joint[3-4]": 2.175,
                "openarm_joint[5-7]": 2.61,
            },
            effort_limit_sim={
                "openarm_joint[1-2]": 40.0,
                "openarm_joint[3-4]": 27.0,
                "openarm_joint[5-7]": 7.0,
            },
            stiffness=80.0,
            damping=4.0,
        ),
        "openarm_gripper": ImplicitActuatorCfg(
            joint_names_expr=[OPENARM_GRIPPER_JOINT_PATTERN],
            velocity_limit_sim=0.2,
            effort_limit_sim=333.33,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

# High-gain, gravity-free tuning recommended by the official Isaac Lab OpenArm
# task configs for differential IK control.
OPENARM_UNI_HIGH_PD_CFG = OPENARM_UNI_CFG.copy()
OPENARM_UNI_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
OPENARM_UNI_HIGH_PD_CFG.actuators["openarm_arm"].stiffness = 400.0
OPENARM_UNI_HIGH_PD_CFG.actuators["openarm_arm"].damping = 80.0


def get_openarm_unimanual_cfg() -> ArticulationCfg:
    """Return the local adapter for the official OpenArm unimanual asset."""

    return OPENARM_UNI_HIGH_PD_CFG


def get_openarm_bimanual_cfg() -> ArticulationCfg:
    """Return the official high-PD bimanual OpenArm configuration."""

    return OPENARM_BI_HIGH_PD_CFG
