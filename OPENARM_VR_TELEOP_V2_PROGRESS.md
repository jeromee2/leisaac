# OpenArm Quest 3 teleoperation V2 progress

Last updated: 2026-09-04 (Asia/Seoul)

## Outcome

Physics01 is now the main scene and default teleoperation task. The new path is registered as `quest3-controller-v2` and does not replace the legacy
`quest3-controller` implementation. It drives the official bimanual OpenArm with a 16D
absolute joint action: left arm 7 + left gripper 1 + right arm 7 + right gripper 1.

## Root causes confirmed

- The former pose path mixed OpenXR `wxyz` quaternions with SciPy's `xyzw` convention in
  an absolute-target branch. This can rotate a wrist/controller delta about the wrong axis.
- The old bimanual path used per-frame differential IK deltas. Near an elbow singularity,
  small pose/noise changes could select a different joint-space motion.
- A direct cross-engine check found that identical joint values do not preserve end-effector
  orientation between the installed MuJoCo model and Isaac USD. For example, a pure MuJoCo
  left-hand +X rotation of `0.12 rad` became approximately
  `[0.0776, 0.0784, -0.0671] rad` in Isaac. V2 therefore does not use MuJoCo FK/Jacobians at
  runtime; this mismatch is a concrete explanation for cross-engine elbow/wrist disagreement.
- The official bimanual USD reports a false left-arm self-contact at the neutral pose. In
  a fixed-target headless test it produced `0.51448 rad/s` combined joint RMS velocity even
  though position changes were only microradians. Disabling self-collision for the V2 task
  alone reduced both initial joint error and settled RMS velocity to `0.00000`.
- The complete bundle at `assets/scenes/Collected_physics01/physics01.usd` resolves all USD layers. Its
  embedded OpenArm uses `*_ee_base_link` end effectors and mirrored revolute gripper signs, unlike the official asset.
- Relocation preflight passes the schema and 132 USD dependencies; five cosmetic material-resource warnings remain (Map #1461, OmniGlass.mdl, OmniPBR.mdl, OmniSurfacePresets.mdl, gltf/pbr.mdl).

## V2 implementation

- Fixed operator frame captured from HMD yaw; later head motion does not move either arm.
- Independent squeeze clutch and trigger gripper state for each hand, with hysteresis.
- Per-hand tracking timeout, reference-space jump detection, release-before-reclutch, and
  `WAITING`, `READY`, `CLUTCHED`, `HOLD`, and `FAULT` states.
- Timestamp-aware One Euro position filtering, quaternion SLERP, and linear/angular target
  speed caps.
- Separate constrained QP solve for each arm using the live Isaac USD FK and PhysX Jacobian,
  not a second engine's robot model. It includes verified joint order/limits, null-space
  posture stabilization, joint braking, singularity slowdown/damping, and hard joint velocity
  limits. A failure on one side does not stop the other side.
- Direct absolute joint targets with no default-position offset. Command/measurement
  divergence faults only the affected arm and requires a fresh clutch.
- Optional JSONL diagnostics via `--openarm_v2_debug_log`.
- V2-only task registration and V2-only self-collision override; legacy code and task names
  remain available.

## Verification completed

Pure mapping/state tests:

```text
8 passed in 1.03s
```

Isaac Sim 5.1 / RTX 5070 Ti headless integration:

```text
official scene:
  schema: 16D absolute joint action and verified joint limits passed
  hold: max_initial_error=0.00000rad, settled_rms_velocity=0.00000rad/s
  grippers: close/reopen passed
  motion: left=0.0497m, right=0.0497m, max_velocity_ratio=1.0000
Physics01 scene:
  resolved dependencies=132, active prims=1469, articulation roots=1
  relocation preflight: schema and 132 USD dependencies pass; five cosmetic material-resource warnings remain (Map #1461, OmniGlass.mdl, OmniPBR.mdl, OmniSurfacePresets.mdl, gltf/pbr.mdl)
  hold: max_initial_error=0.01228rad, settled_rms_velocity=0.01124rad/s
  grippers: close/reopen passed
  motion: left=0.0499m, right=0.0497m, max_velocity_ratio=1.0000
frame parity: both arms +X/+Y/+Z = 0.0300m, cos=1.0000;
              both arms +roll/+pitch/+yaw = 0.1200rad, cos=1.0000
```

Commands:

```bash
PYTHONPATH=source/leisaac \
  python -m pytest -q \
  source/leisaac/test/test_openarm_vr_v2_core.py

python -u scripts/tools/openarm_v2_teleop_check.py \
  --headless --device cuda:0

python -u scripts/tools/openarm_v2_fk_parity_check.py \
  --headless --device cuda:0

python -u scripts/tools/usd_teleop_preflight.py assets/scenes/Collected_physics01/physics01.usd --device cuda:0

python -u scripts/tools/openarm_v2_teleop_check.py --task LeIsaac-OpenArm-Bimanual-Physics01-QuestV2-v0 --headless --device cuda:0
```

## Install and run V2

```bash
python -m pip install -e 'source/leisaac[openarm-vr]'

export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
NV_DEVICE_PROFILE=auto-webrtc \
XR_RUNTIME_JSON="$CLOUDXR_ROOT/openxr_cloudxr.json" \
LD_LIBRARY_PATH="$CLOUDXR_ROOT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
python -u scripts/environments/teleoperation/teleop_se3_agent.py \
  --task LeIsaac-OpenArm-Bimanual-Physics01-QuestV2-v0 \
  --teleop_device quest3-controller-v2 \
  --num_envs 1 --device cuda:0 --headless \
  --openarm_v2_debug_log logs/openarm_v2_physics01.jsonl
```

Physics01 is bundled at `assets/scenes/Collected_physics01` and is also the default when `--task` is omitted. The former official LiftCube V2 scene remains registered for
regression checks as:

```bash
--task LeIsaac-OpenArm-Bimanual-LiftCube-QuestV2-v0
```

Hold the matching controller squeeze to move an arm. Release it to freeze that arm. The
matching trigger controls its gripper. Left X pauses/resumes and forces recalibration; Left
Y resets the task.

## Remaining hardware validation

- Confirm controller forward/left/up and wrist roll/pitch/yaw signs in the Quest headset.
- Confirm tracking loss and squeeze release/re-engage never jump either arm.
- Self-collision is disabled only for the V2 task to remove the confirmed false contact.
  Cross-arm/body collision prevention is therefore not guaranteed; begin at low speed and
  keep the hands separated until collision geometry is repaired or filtered per link pair.
- The collected lab asset has one unresolved texture (`Map #1461`); geometry and physics compose, but that material
  may render with a fallback. Isaac's built-in MDL identifiers are also reported as dependency warnings.

## Response-speed update (2026-09-04)

The live JSONL trace confirmed that the QP was velocity-saturated while each absolute
joint target stayed only one 60 Hz control step (about `0.033 rad` on joints 1--2)
ahead of the measured joint. With the intentionally high-damping OpenArm drive, that
small position error limited the physical joints to roughly `0.08--0.10 rad/s`.

The V2 controller now:

- projects the velocity-bounded QP target `0.08 s` ahead instead of one `1/60 s` step;
- keeps the existing high damping, joint limits, per-joint velocity limits, and
  per-arm divergence fault;
- raises and synchronizes the mapper/QP Cartesian caps from `0.6/3.0` to
  `1.0 m/s` and `4.5 rad/s`;
- exposes `--openarm_v2_joint_target_lookahead_s` in the safe range `(0, 0.09]`.

Isaac Sim headless comparison for the same bilateral 5 cm target:

| Setting | Frames to 90% | Simulated time | Peak physical velocity-limit ratio | Settled RMS |
|---|---:|---:|---:|---:|
| Previous `0.0167 s` | 82 | 1.37 s | 0.0783 | 0.00001 rad/s |
| New `0.08 s` | 17 | 0.28 s | 0.3844 | 0.00002 rad/s |

The measured 90%-response improvement is `82 / 17 = 4.82x`. The updated headless
integration check, all seven mapping/state tests, fixed-pose hold, bilateral grippers,
joint-limit enforcement, physical velocity-limit enforcement, and independent-side
failure handling pass.
