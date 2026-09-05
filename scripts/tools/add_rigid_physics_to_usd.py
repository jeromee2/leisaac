"""Add pickable rigid-body physics to a mesh-based USD asset."""

from __future__ import annotations

import argparse
from pathlib import Path

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("usd_path", type=Path)
    parser.add_argument("--mass", type=float, default=0.1)
    parser.add_argument("--static-friction", type=float, default=1.0)
    parser.add_argument("--dynamic-friction", type=float, default=0.8)
    parser.add_argument("--restitution", type=float, default=0.0)
    parser.add_argument(
        "--approximation",
        choices=("convexHull", "convexDecomposition"),
        default="convexDecomposition",
    )
    args = parser.parse_args()

    stage = Usd.Stage.Open(str(args.usd_path))
    if stage is None:
        raise RuntimeError(f"Unable to open USD: {args.usd_path}")

    root = stage.GetDefaultPrim()
    if not root:
        roots = list(stage.GetPseudoRoot().GetChildren())
        if len(roots) != 1:
            raise RuntimeError("USD must have a default prim or exactly one root prim")
        root = roots[0]
        stage.SetDefaultPrim(root)

    mesh_prims = [prim for prim in Usd.PrimRange(root) if prim.IsA(UsdGeom.Mesh)]
    if not mesh_prims:
        raise RuntimeError(f"No Mesh prim found below {root.GetPath()}")

    rigid_body = UsdPhysics.RigidBodyAPI.Apply(root)
    rigid_body.CreateRigidBodyEnabledAttr(True)
    rigid_body.CreateKinematicEnabledAttr(False)

    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(args.mass)

    material_path = root.GetPath().AppendPath("PhysicsMaterials/Grippy")
    material = UsdShade.Material.Define(stage, material_path)
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material_api.CreateStaticFrictionAttr(args.static_friction)
    material_api.CreateDynamicFrictionAttr(args.dynamic_friction)
    material_api.CreateRestitutionAttr(args.restitution)

    for prim in mesh_prims:
        collision = UsdPhysics.CollisionAPI.Apply(prim)
        collision.CreateCollisionEnabledAttr(True)
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_collision.CreateApproximationAttr(args.approximation)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material,
            bindingStrength=UsdShade.Tokens.weakerThanDescendants,
            materialPurpose="physics",
        )

    stage.GetRootLayer().Save()
    print(f"Updated {args.usd_path}")
    print(f"Rigid body: {root.GetPath()} | mass={args.mass} kg")
    print(f"Colliders: {', '.join(str(prim.GetPath()) for prim in mesh_prims)}")
    print(
        f"Approximation={args.approximation} | static friction={args.static_friction} | "
        f"dynamic friction={args.dynamic_friction} | restitution={args.restitution}"
    )


if __name__ == "__main__":
    main()
