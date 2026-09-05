"""Check USD composition and bimanual OpenArm teleoperation compatibility."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("usd_path", type=Path, help="USD scene to inspect.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

from leisaac.utils.general_assets import get_prim_pos_rot
from pxr import Sdf, Usd, UsdPhysics, UsdUtils

OPENARM_ARM_JOINTS = {f"openarm_{side}_joint{index}" for side in ("left", "right") for index in range(1, 8)}
OPENARM_GRIPPER_JOINTS = {f"openarm_{side}_finger_joint{index}" for side in ("left", "right") for index in range(1, 3)}
OPENARM_EE_BODY_OPTIONS = {
    "left": ("openarm_left_hand", "openarm_left_ee_base_link"),
    "right": ("openarm_right_hand", "openarm_right_ee_base_link"),
}


def main() -> int:
    path = args_cli.usd_path.expanduser().resolve()
    if not path.is_file():
        print(f"FAIL: USD file does not exist: {path}")
        return 2

    layer = Sdf.Layer.FindOrOpen(str(path))
    if layer is None:
        print(f"FAIL: USD root layer cannot be opened: {path}")
        return 2

    _, dependencies, unresolved = UsdUtils.ComputeAllDependencies(str(path))
    missing_layers = [asset for asset in unresolved if Path(str(asset)).suffix.lower() in {".usd", ".usda", ".usdc"}]
    missing_resources = [asset for asset in unresolved if asset not in missing_layers]
    if missing_layers:
        print(f"FAIL: {path} has unresolved USD layers:")
        for asset in missing_layers:
            print(f"  - {asset}")
        print("Request the missing files and preserve their relative directory layout.")
        return 2
    if missing_resources:
        print("WARN: unresolved material/texture resources (geometry and physics can still compose):")
        for asset in missing_resources:
            print(f"  - {asset}")

    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadAll)
    if stage is None:
        print(f"FAIL: USD stage cannot be composed: {path}")
        return 2

    prims = list(stage.Traverse())
    physics_scenes = [str(prim.GetPath()) for prim in prims if prim.IsA(UsdPhysics.Scene)]
    articulation_roots = [prim for prim in prims if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
    joint_names = {prim.GetName() for prim in prims if prim.IsA(UsdPhysics.Joint)}
    body_names = {prim.GetName() for prim in prims if prim.HasAPI(UsdPhysics.RigidBodyAPI)}
    gripper_joint_prims = {prim.GetName(): prim for prim in prims if prim.GetName() in OPENARM_GRIPPER_JOINTS}
    missing_arm_joints = sorted(OPENARM_ARM_JOINTS - joint_names)
    missing_gripper_joints = sorted(OPENARM_GRIPPER_JOINTS - joint_names)
    ee_body_names = {
        side: next((name for name in options if name in body_names), None)
        for side, options in OPENARM_EE_BODY_OPTIONS.items()
    }
    missing_ee_sides = [side for side, name in ee_body_names.items() if name is None]

    print(f"PASS: composed {path}")
    print(f"  resolved dependencies: {len(dependencies)}")
    print(f"  active prims: {len(prims)}")
    print(f"  physics scenes: {physics_scenes or 'none'}")
    print(f"  articulation roots: {[str(prim.GetPath()) for prim in articulation_roots] or 'none'}")
    for name in sorted(gripper_joint_prims):
        joint = UsdPhysics.RevoluteJoint(gripper_joint_prims[name])
        print(f"  {name} limits [deg]: {joint.GetLowerLimitAttr().Get()}, {joint.GetUpperLimitAttr().Get()}")
    for prim in articulation_roots:
        parent = prim.GetParent()
        print(f"  articulation asset pose: {parent.GetPath()} {get_prim_pos_rot(parent)}")

    if len(articulation_roots) == 1 and not missing_arm_joints and not missing_gripper_joints and not missing_ee_sides:
        print(f"PASS: bimanual OpenArm teleoperation schema; end effectors: {ee_body_names}")
        return 0

    print("FAIL: scene is not compatible with the bimanual OpenArm teleoperator:")
    if len(articulation_roots) != 1:
        print(f"  - expected one articulation root, found {len(articulation_roots)}")
    if missing_arm_joints:
        print(f"  - missing arm joints: {missing_arm_joints}")
    if missing_gripper_joints:
        print(f"  - missing gripper joints: {missing_gripper_joints}")
    if missing_ee_sides:
        print(f"  - missing end-effector body for sides: {missing_ee_sides}")
    return 3


if __name__ == "__main__":
    exit_code = main()
    if exit_code:
        # Kit normalizes SystemExit to zero during shutdown; preserve the check's
        # failure status for shell scripts and CI after flushing its diagnostics.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
    simulation_app.close()
