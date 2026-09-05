# OpenArm Quest 3 teleoperation setup

This branch contains the robot task, Quest controller mapping, tuned reach and wrist values, XR scene orientation,
initial pose, the complete Physics01 scene bundle, and headless reachability checks. It does not contain NVIDIA
binary SDKs or the separately licensed Isaac Teleop web client.

## Known-good baseline

- Ubuntu 24.04
- NVIDIA RTX 5070 Ti, driver 580.173.02
- Isaac Sim 5.1.0, Isaac Lab 2.3.0, Python 3.11, CUDA 12.8
- NVIDIA CloudXR SDK 6.2.1
- Meta Quest 3 on the same LAN as the workstation

Other NVIDIA GPUs can work. Use the driver, codec, and resolution supported by that GPU; do not copy GPU libraries
from another workstation. AV1 can be changed to H.265 or H.264 in the Quest web client.

## 1. Install LeIsaac

```bash
git clone --branch openarm-quest3-teleop --recursive https://github.com/jeromee2/leisaac.git
cd leisaac
conda create -n leisaac python=3.11
conda activate leisaac
conda install -c nvidia/label/cuda-12.8.1 cuda-toolkit
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
sudo apt install cmake build-essential
dependencies/IsaacLab/isaaclab.sh --install
pip install -e "source/leisaac[openarm-vr]"
```

Use the official OpenArm assets documented in
`docs/docs/resources/openarm_v1.0_asset.md`. Run the portable check before connecting a headset:

```bash
python scripts/tools/openarm_bimanual_teleop_check.py --headless --device cuda
```

## 2. Install CloudXR

Download CloudXR SDK 6.2.1 for Linux from NVIDIA on each workstation and extract it locally. Do not copy the
`libcloudxr.so` from a different GPU machine.

```bash
export CLOUDXR_ROOT="${CLOUDXR_ROOT:-$HOME/cloudxr6}"
export CXR_LIB_PATH="$CLOUDXR_ROOT/libcloudxr.so"
export LD_LIBRARY_PATH="$CLOUDXR_ROOT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export XR_RUNTIME_JSON="$CLOUDXR_ROOT/openxr_cloudxr.json"
python scripts/cloudxr/cloudxr_service.py
```

## 3. Serve the Quest client

Obtain the licensed NVIDIA Isaac Teleop Web Client build and place `index.html`, `bundle.js`, its license file, and
`favicon.ico` in `$HOME/isaacteleop-client`. The build used during development had these checksums:

```text
904d3a7054e420af0f83b3e94065bf723d47117848d59ae86fbf13a99b12fbc0  index.html
96e0981b0483e59e9420a1beed4b8883617d4b6929c63164133d42921a3f9499  bundle.js
```

```bash
python -m http.server 8080 --bind 0.0.0.0 --directory "$HOME/isaacteleop-client"
```

Open `http://WORKSTATION_LAN_IP:8080` in the Quest browser and connect to `WORKSTATION_LAN_IP:49100`. Tailscale is
optional for administration; low-latency VR streaming should use the LAN address.

## 4. Start teleoperation

Run this in the same shell that has `XR_RUNTIME_JSON` and `LD_LIBRARY_PATH`:

```bash
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
NV_DEVICE_PROFILE=auto-webrtc \
XR_RUNTIME_JSON="$CLOUDXR_ROOT/openxr_cloudxr.json" \
LD_LIBRARY_PATH="$CLOUDXR_ROOT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
python scripts/environments/teleoperation/teleop_se3_agent.py \
  --task=LeIsaac-OpenArm-Bimanual-Physics01-QuestV2-v0 \
  --teleop_device=quest3-controller-v2 \
  --num_envs=1 --device=cuda --headless
```

Play/Reset in VR sends CloudXR `teleop_command` messages. Left X also pauses/resumes, Left Y resets, squeeze is the
motion clutch, and triggers operate the grippers.

## 5. Legacy V1 teleoperation stability status (2026-09-02)

Completed:

- Kept OpenXR and Isaac poses in `xyz + wxyz` order at every SciPy rotation boundary. The previous absolute-target
  path passed `wxyz` directly to SciPy as `xyzw`, which could map a controller rotation onto the wrong robot axis.
- Enabled the existing One Euro pose filter for absolute bimanual tracking and added per-frame target speed caps.
  Defaults are `1.0 m/s` and `6.0 rad/s`; they are configurable with
  `--openarm_bimanual_max_linear_speed` and `--openarm_bimanual_max_angular_speed`.
- Seeded the filter when the motion clutch engages so its first controlled frame cannot bypass the speed cap.
- Use a bent initial elbow posture and posture-aware differential IK with null-space correction and joint-limit
  protection instead of starting joint 4 at its singular lower limit.
- Added regression checks for quaternion direction/order, target speed limits, translation and rotation response,
  elbow posture, joint limits, collision geometry, and crossing-hand separation.

Validated on the known-good baseline:

```text
pose math: wxyz direction and target speed caps passed
forward/backward/outward/upward translation probes passed
roll/pitch/yaw rotation probes passed
posture/limits: elbows=[1.011, 1.000], minimum_limit_margin=0.158 rad
crossing-hand collision: minimum=0.179 m, final=0.207 m
OpenArm bimanual headless physics checks passed.
```

Remaining hardware check:

- In Quest 3, verify that wrist rotations follow the same physical axis, elbow bend stays continuous near the edge
  of the workspace, and releasing/re-engaging squeeze does not produce a target jump.

## 6. OpenArm Quest V2 controller (Physics01, 2026-09-03)

V2 is isolated under a new device and task name. It uses fixed-HMD-yaw clutch mapping and independent constrained
native Isaac-Jacobian QP controllers, while the legacy `quest3-controller` path remains available.

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

Run these checks before putting on the headset:

```bash
python -u scripts/tools/usd_teleop_preflight.py assets/scenes/Collected_physics01/physics01.usd --device cuda:0
python -u scripts/tools/openarm_v2_teleop_check.py --task LeIsaac-OpenArm-Bimanual-Physics01-QuestV2-v0 --headless --device cuda:0
python -u scripts/tools/openarm_v2_fk_parity_check.py --headless --device cuda:0
```
Physics01 is the main V2 scene and its complete bundle is included at
`assets/scenes/Collected_physics01`. Set `LEISAAC_PHYSICS01_USD_PATH` only when using a relocated compatible bundle. CloudXR and the licensed web client are separate workstation installations.


The collected scene composes and its embedded OpenArm passes the 16D action, fixed-hold, gripper, and bilateral QP
motion checks. To run that scene, use the same CloudXR environment above with the Physics01 task:

```bash
python -u scripts/environments/teleoperation/teleop_se3_agent.py --task LeIsaac-OpenArm-Bimanual-Physics01-QuestV2-v0 --teleop_device quest3-controller-v2 --num_envs 1 --device cuda:0 --headless
```

Detailed design, results, safety behavior, and remaining headset checks are recorded in
`OPENARM_VR_TELEOP_V2_PROGRESS.md`.

## 7. V2 arm response-speed tuning (2026-09-04)

The default V2 response was increased after the live trace showed that its high-damping
joint drive received only a one-frame-ahead target. The default joint-target lookahead is
now `0.08 s`, and the Cartesian target caps are `1.0 m/s` and `4.5 rad/s`. No extra
launch flags are required.

If the arm feels too fast in the headset, add this to the teleop command:

```bash
--openarm_v2_joint_target_lookahead_s 0.05
```

The accepted range is greater than `0` and at most `0.09 s`. Keep the default `0.08`
for the verified faster response. The physical joint velocity limits, joint-position
limits, high damping, and per-arm safety hold remain active.
