# OpenArm Quest 3 teleoperation setup

This branch contains the robot task, Quest controller mapping, tuned reach and wrist values, XR scene orientation,
initial pose, and a headless reachability check. It intentionally does not contain NVIDIA binaries or a generated
Isaac Teleop web bundle.

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
pip install -e source/leisaac
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
export CLOUDXR_ROOT="$HOME/cloudxr6"
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
python scripts/environments/teleoperation/teleop_se3_agent.py \
  --task=LeIsaac-OpenArm-Bimanual-LiftCube-v0 \
  --teleop_device=quest3-controller \
  --num_envs=1 --device=cuda --headless --sensitivity=0.6 --xr_start_paused
```

Play/Reset in VR sends CloudXR `teleop_command` messages. Left X also pauses/resumes, Left Y resets, squeeze is the
motion clutch, and triggers operate the grippers.
