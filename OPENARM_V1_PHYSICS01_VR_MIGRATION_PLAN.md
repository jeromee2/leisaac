# Physics01 OpenArm v2 → v1.0 VR 텔레옵 교체 계획

## 1. 목표

현재 `LeIsaac-OpenArm-Bimanual-Physics01-QuestV2-v0`의 배경·물체·VR 조작 방식은 유지하고, Physics01 USD에 포함된 OpenArm v2 articulation만 공식 OpenArm v1.0 bimanual 모델로 교체한다.

최종 실행 명령과 teleop device 이름은 유지한다.

```text
task:          LeIsaac-OpenArm-Bimanual-Physics01-QuestV2-v0
teleop_device: quest3-controller-v2
action:        left arm 7 + left gripper 1 + right arm 7 + right gripper 1 = 16D
physics:       120 Hz
control:       60 Hz
```

여기서 `V2`는 VR 제어 알고리즘 버전이다. 교체 후 로봇 hardware model은 `OpenArm v1.0`이지만 HMD 기준 좌표, clutch, filtering, QP 및 안전 정지는 그대로 사용한다.

구현 상태 (2026-09-09): canonical task의 V1 전환과 V2Legacy 롤백 등록을 완료했다.
CPU PhysX에서 asset/composition, hold, gripper, bilateral motion, reset, joint limit 및
six-axis parity 검사를 통과했다. CUDA 실행과 Quest 3 실기 검증은 남아 있다.

## 2. 확인된 현재 상태

- Physics01 배경 파일은 `assets/scenes/Collected_physics01/physics01.usd`이고 내부에 `/World/openarm_v20`이 포함되어 있다.
- 현재 Physics01 config는 이 내장 articulation을 `{ENV_REGEX_NS}/Scene/openarm_v20`에서 `spawn=None`으로 연결한다.
- 저장소에는 이미 공식 v1.0용 `get_openarm_bimanual_cfg()`와 일반 v1.0 LiftCube task가 있다.
- v1.0 bimanual asset의 현재 해석 우선순위는 `LEISAAC_OPENARM_BIMANUAL_USD_PATH` → `assets/robots/openarm_v1.0/openarm_bimanual.usd` → Isaac Nucleus asset이다.
- 현재 checkout에는 `assets/robots/openarm_v1.0/openarm_bimanual.usd`가 없다. 구현 시작 전에 실제 사용할 v1.0 asset을 로컬에 고정해야 한다.
- 공식 v1.0은 양팔 각 7 DoF와 parallel-link gripper를 사용한다. v1.0 gripper는 prismatic `0.0–0.044 m`이고, Physics01 내장 v2 gripper는 mirrored revolute joint와 좌우 `+0.75/-0.75 rad` open target을 사용한다.
- 현재 LeIsaac의 Isaac 5.1 asset adapter는 v1.0 runtime body를 `openarm_body_link`, `openarm_left_hand`, `openarm_right_hand`로 기대한다. 반면 현재 공식 v1.0 Xacro 예제에는 `openarm_body_link0`, `openarm_*_link7`, `openarm_*_hand_tcp`가 보인다. source와 import option에 따라 runtime 이름이 달라질 수 있으므로 asset을 고정하기 전에 어느 이름도 최종 schema로 가정하지 않는다.
- Quest V2 QP는 별도 MuJoCo model이 아니라 현재 Isaac articulation의 FK와 Jacobian을 직접 사용한다. 따라서 모델별 schema만 올바르게 주면 pose mapping과 tracking 로직을 재사용할 수 있다.

공식 근거:

- [OpenArm 1.0 문서](https://docs.openarm.dev/1.0/)
- [OpenArm 1.0 robot description](https://docs.openarm.dev/1.0/software/description/)
- [공식 `openarm_description`](https://github.com/enactic/openarm_description)
- [공식 `openarm_isaac_lab`](https://github.com/enactic/openarm_isaac_lab)

## 3. 유지할 것과 바꿀 것

| 영역 | 유지 | v1.0에 맞게 변경 |
|---|---|---|
| Physics01 | 실험실, 테이블, props, 조명, XR anchor | 내장 `/openarm_v20` 비활성화, v1.0 별도 spawn |
| VR mapping | fixed HMD yaw, 양손 독립 clutch, trigger, tracking timeout, jump detection | 없음 |
| filtering | One Euro 위치 filter, quaternion SLERP, Cartesian 속도 제한 | 없음 |
| IK/QP | Isaac FK/Jacobian, 양팔 독립 QP, singularity slowdown, divergence fault | v1.0 EE body와 joint-limit source |
| action | 16D 절대 joint target | gripper processed target을 좌우 모두 `0.044/0.0 m`로 변경 |
| physics | 120 Hz physics, 60 Hz control, gravity-off teleop | v1.0 actuator와 self-collision 설정 |
| 데이터 | raw action 순서와 18D measured state 순서 | `robot_model`과 asset hash 기록, v2 episode와 혼합 금지 |

## 4. 구현 설계

### 4.1 v1.0 asset 고정

대상:

- `assets/robots/openarm_v1.0/openarm_bimanual.usd` 및 모든 참조 asset
- `source/leisaac/leisaac/assets/robots/openarm.py`

절차:

1. 우선 현재 generic v1.0 LiftCube task가 대상으로 삼는 Isaac Sim 5.1 공식 OpenArm bimanual USD를 선택한다. 이 경로가 기존 LeIsaac integration을 가장 많이 재사용한다.
2. 해당 asset을 사용할 수 없을 때만 공식 `openarm_description/assets/robot/openarm_v1.0` Xacro를 pinned IsaacLab converter로 변환한다. 이 경우 Isaac USD와 같은 schema라고 가정하지 않고 별도 artifact로 전체 검증한다.
3. 한 번 선택한 뒤 두 source를 섞지 않는다. source URL/URI, upstream commit, URDF 생성·USD 변환 명령, Isaac/Importer 버전과 fixed-base·fixed-joint merge·mimic·drive·collision option을 기록한다.
4. 완성된 local asset과 모든 USD, mesh, material, texture dependency를 `assets/robots/openarm_v1.0/`에 둔다. `omniverse://` 또는 Nucleus 참조가 하나라도 남거나 상대 참조가 해석되지 않으면 scene 교체를 시작하지 않는다.
5. root만이 아니라 dependency 전체의 path·size·SHA-256을 `OPENARM_V1_ASSET_MANIFEST.json`에 기록한다.
6. production Physics01-v1 config는 `_openarm_bimanual_usd_path()`의 환경변수 우선순위를 사용하지 않고 manifest가 지정한 vendored root USD를 직접 연다. 시작 시 전체 hash를 검사하고 불일치하면 env 생성 전에 실패한다. 환경변수와 Nucleus fallback은 generic 개발 task에만 남긴다.

asset preflight 필수 schema:

```text
active articulation roots: 1
arm joints:     좌우 각 7개, 실제 이름과 USD order 저장
finger joints:  좌우 각 2개 prismatic, 실제 이름과 USD order 저장
base body:      composed USD에서 확정
IK/TCP bodies:  composed USD에서 확정
```

preflight schema도 `OPENARM_V1_ASSET_MANIFEST.json` 안에 함께 저장한다. 최소한 root/default prim, articulation root, joint별 이름·type·axis·limit, rigid body 이름, fixed-joint merge 결과, base 후보, EE/TCP 후보, gripper mimic/tendon 관계를 담는다. 다음 단계의 Python 상수는 이 inventory를 사람이 확인한 뒤 확정하며, 이름 후보가 있다는 이유만으로 자동 추측하지 않는다. manifest에는 schema version을 두고 production config가 지원 version과 일치하는지 검사한다.

### 4.2 Physics01에서 내장 v2 제거

대상:

- `source/leisaac/leisaac/tasks/lift_cube/openarm_collected_physics01_env_cfg.py`
- `assets/scenes/Collected_physics01/physics01_v1_background.usda`

`physics01.usd` 자체와 기존 V2 task는 수정하지 않는다. 작은 composition layer `physics01_v1_background.usda`가 원본 `/World`를 relative reference하고 `/World/openarm_v20`에 `active=false` override를 author한다. 새 V1 candidate scene만 이 layer를 연다.

```text
compose physics01_v1_background.usda
→ /World/openarm_v20 inactive 확인
→ background를 {ENV_REGEX_NS}/Scene에 spawn
→ v1.0을 {ENV_REGEX_NS}/Scene/openarm_v1에 별도 spawn
```

composition preflight에서 내장 v2 prim이 없거나 active이면 실패한다. V1 background의 `UsdFileCfg`에는 scene-level `articulation_props`를 넣지 않는다. 따라서 V2에 solver/self-collision override를 적용한 뒤 뒤늦게 끄는 순서 문제가 없다.

첫 `sim.reset()`과 PhysX tensor view 생성 전에 composed USD traversal로 `/openarm_v20` 아래 active rigid/joint prim이 0개인지 확인한다. 초기화 후에는 runtime articulation view가 V1 하나뿐이고 V2 rigid/joint handle이 0개인지 다시 assert한다.

기존 `spawn_physics01()`과 그 안의 V2 finger frame/mimic repair는 legacy V2 config에만 남긴다. V1 candidate는 이 callback을 호출하지 않는다. v1.0의 solver와 self-collision 설정은 새 `robot` ArticulationCfg에만 둔다.

### 4.3 v1.0 articulation 연결

기존 `OpenArmBimanualPhysics01EnvCfg`는 A/B 기준으로 유지하고, 같은 파일에 `OpenArmBimanualPhysics01V1SceneCfg`와 `OpenArmBimanualPhysics01V1EnvCfg`를 둔다. 공통 observation, props reset, XR 설정과 timing은 상속하고 robot/background/frame만 override한다.

V1 candidate의 `robot`을 다음 원칙으로 구성한다.

```text
cfg source:  get_openarm_bimanual_cfg()
prim path:   {ENV_REGEX_NS}/Scene/openarm_v1
spawn:       v1.0 UsdFileCfg 유지
root override: v2용 /root_joint 제거
```

v2 root pose 숫자를 v1 root에 그대로 복사하지 않는다. 교체 전 baseline script로 Physics01의 v2 torso mounting plane과 좌우 shoulder joint origin의 world transform을 저장한다. v1 asset의 대응 frame을 asset schema에서 선택하고 좌우 shoulder midpoint와 torso 방향을 맞춰 첫 root transform을 계산한다. 이후 table/floor clearance 검사로 미세 조정한다.

다음 세 값을 명시적 calibration 상수로 두고 headless scene-fit 검사 결과로 확정한다.

- `PHYSICS01_V1_ROOT_POS`
- `PHYSICS01_V1_ROOT_ROT`
- `PHYSICS01_V1_INIT_JOINT_POS`

첫 joint posture는 generic v1.0 task에서 이미 사용하는 bent-elbow posture를 시작점으로 사용한다.

```text
left:  joint1=+0.6, joint4=0.6, others=0
right: joint1=-0.6, joint4=0.6, others=0
```

v1.0 joint limit 안에 있어야 하며, table/scene penetration이나 workspace 부족이 확인될 때만 root pose와 초기 posture를 조정한다. VR pose mapping scale로 잘못된 scene placement를 보정하지 않는다.

scene-fit 결과에는 torso forward/up 방향, 좌우 shoulder world pose, 초기 EE world pose, table top 높이, robot collision AABB를 함께 기록한다. 접촉 검사는 의도된 base/support pair만 명시적 allowlist로 허용하고, arm↔torso, arm↔table, gripper↔table 및 robot↔prop의 초기 contact는 금지한다.

### 4.4 model-specific frame과 gripper 변경

Physics01의 model profile은 고정한 asset schema에서 다음 값으로 확정한다.

```text
robot_name: openarm_bimanual_v1_0_physics01
base frame: OPENARM_V1_RESOLVED_BASE_BODY
left EE:    OPENARM_V1_RESOLVED_LEFT_EE_BODY
right EE:   OPENARM_V1_RESOLVED_RIGHT_EE_BODY
gripper open:  left=0.044 m, right=0.044 m
gripper close: left=0.0 m,   right=0.0 m
```

Isaac 5.1 official asset을 선택하고 inventory가 현재 adapter와 일치할 때만 base=`openarm_body_link`, EE=`openarm_left_hand/openarm_right_hand`를 사용한다. Xacro conversion을 선택하면 `*_body_link0`, `*_link7`, `*_hand_tcp` 후보 중 실제 rigid body와 원하는 tool frame을 six-axis parity로 확정한다. `ee_frame`도 이 resolved base/EE 경로를 사용한다. 현재 v2용 `PHYSICS01_EE_BODY_NAMES`와 좌우 부호가 다른 gripper 상수는 제거한다.

`get_openarm_bimanual_cfg()`의 공식 v1.0 actuator 설정을 그대로 사용한다.

- arm high-PD: stiffness `400`, damping `80`
- arm velocity/effort limit: 기존 v1.0 profile
- gripper: prismatic용 velocity/effort/stiffness/damping

현재 Physics01의 revolute gripper용 velocity `2.0`, effort `5.0`, stiffness `100`, damping `10` override는 V1 candidate에 적용하지 않는다. 공식 v1.0 USD에서 확인된 false self-contact 때문에 env spawn 전에 `robot.spawn.articulation_props.enabled_self_collisions=False`를 명시하고, config parse test와 runtime test가 실제 false인지 확인한다. robot↔table/props collision filter는 건드리지 않고 contact probe로 활성 상태를 별도 확인한다.

### 4.5 Quest V2 controller 재사용

그대로 유지할 파일:

- `source/leisaac/leisaac/devices/openarm_vr_v2/core.py`
- `source/leisaac/leisaac/devices/openarm_vr_v2/openxr_source.py`

그대로 유지할 동작:

- HMD yaw로 고정 operator frame 생성
- squeeze clutch와 release-before-reclutch
- trigger gripper hysteresis
- tracking stale/jump 시 해당 팔만 HOLD/FAULT
- One Euro/SLERP smoothing
- Cartesian 및 joint velocity cap
- 양팔 독립 QP와 singularity slowdown
- `0.08 s` joint-target lookahead
- command/measured divergence 감시

대상:

- `source/leisaac/leisaac/devices/openarm_vr_v2/isaac_qp_controller.py`
- `source/leisaac/leisaac/devices/openarm_vr_v2/device.py`
- `source/leisaac/leisaac/assets/robots/openarm.py`

현재 controller 안의 `EXPECTED_JOINT_LIMITS_RAD`를 이름만 V2인 상수로 두지 않는다. `openarm.py`의 단일팔 `OPENARM_V10_JOINT_LIMITS_RAD`는 mirrored bimanual의 좌우 joint1·2 limit을 표현하지 못하므로 재사용하지 않는다. asset manifest로 확인한 공식 v1.0 bimanual per-side 7×2 ordered table을 새 단일 source로 만들고 device schema 검사와 QP bound가 같은 값을 사용하게 한다. runtime에서 읽은 `soft_joint_pos_limits`가 이 값과 `1e-4 rad` 이상 다르면 VR 입력을 받기 전에 실패한다.

QP는 `openarm_ee_body_names`로 전달된 resolved v1.0 EE body의 live Jacobian을 사용하므로 별도 v1 IK solver를 만들지 않는다. v2와 v1의 joint axis/sign 또는 FK가 같다고 가정하지 않는다. 각 joint에 작은 `+0.02 rad` perturbation을 줬을 때 EE pose delta와 live Jacobian prediction이 같은 방향인지 확인하고, 이어서 양팔 six-axis parity를 검증한다.

controller 초기화 시 articulation이 fixed-base인지, EE body가 정확히 하나인지, Jacobian column이 ordered arm joint 7개와 대응하는지 확인한다. 현재 fixed-base articulation에서 쓰는 `jacobian_body_id = body_id - 1`도 finite-difference Jacobian 비교로 검증되기 전에는 단순 index 규칙만으로 신뢰하지 않는다.

### 4.6 기존 action·observation·reset 보존

`action_process.py`의 Quest V2 16D 구성과 joint 순서는 유지한다.

```text
0..6   left arm joint target
7      left gripper +1=open, -1=close
8..14  right arm joint target
15     right gripper +1=open, -1=close
```

`gripper_open_positions`를 v1.0 좌우 `0.044`로 넘겨 expanded gripper joint target과 실제 prismatic target만 바꾼다. importer가 follower를 mimic/tendon으로 만들었는지 독립 actuated joint로 만들었는지는 asset manifest에서 확인한다. `BinaryJointPositionActionCfg`가 실제 command joint ID 전체를 의도한 순서로 선택하고 follower가 중복 구속되지 않는지 runtime에서 검증한다. observation joint ordering은 기존 `OPENARM_BIMANUAL_CONTROLLED_JOINT_PATTERNS`와 `preserve_order=True`를 유지한다.

Physics01 props 등록, scene reset, 120/60 Hz, gravity disable event, XR anchor, viewer pose 및 성공/실패 callback은 변경하지 않는다.

### 4.7 후보 task와 최종 전환

구현·검증 중에는 기존 V2 task를 건드리지 않고 다음 candidate ID를 등록한다.

```text
LeIsaac-OpenArm-Bimanual-Physics01-V1-QuestV2-v0
```

config parse, headless integration, parity와 Quest 실기 gate를 모두 통과한 뒤 현재 사용자가 쓰는 `LeIsaac-OpenArm-Bimanual-Physics01-QuestV2-v0` registration을 V1 config로 전환한다. 기존 embedded V2 config에는 `LeIsaac-OpenArm-Bimanual-Physics01-V2Legacy-QuestV2-v0`를 연결해 한 release 동안 A/B와 rollback에 사용한다. 두 task가 같은 class를 잘못 가리키지 않는지 registration smoke test로 확인한다.

최종 사용자 실행 명령과 `quest3-controller-v2` device 이름은 바뀌지 않는다. candidate ID는 전환 검증용이며, 안정화 후 유지할 필요가 없으면 제거한다.

## 5. 검증 계획

### A. 정적 asset·scene preflight

`scripts/tools/usd_teleop_preflight.py`를 v1/v2 joint type에 공통으로 동작하게 고친다. v1 finger를 `RevoluteJoint`로 강제 cast하지 않고 실제 `PrismaticJoint`인지와 meter 단위 limit을 검사한다.

검사 항목:

- local v1.0 USD와 모든 dependency 해석
- exact joint/body names와 순서
- 좌우 7개 arm limit
- 네 finger joint가 prismatic이고 command/mimic 관계와 range가 asset manifest와 일치함
- Physics01 구성 후 active articulation root가 v1.0 하나뿐임
- `/openarm_v20` 아래 active rigid body/joint가 없음
- scene props와 table이 기존과 동일하게 등록됨

raw `physics01.usd`만 보지 않고 V1 background와 robot이 함께 compose된 runtime stage를 검사한다. `/Scene/openarm_v1`의 dependency closure에 외부 `omniverse://`/Nucleus 참조가 하나라도 남거나 manifest hash가 다르면 candidate task 생성을 막는다. Physics01 background dependency는 기존 preflight 결과와 별도로 비교한다.

### B. Headless integration

새 test script를 만들지 않고 `scripts/tools/openarm_v2_teleop_check.py`를 model path와 EE/gripper type을 config에서 읽도록 일반화한다. `V2`는 controller 세대 이름으로 유지한다.

필수 통과 조건:

1. 기존 pure mapper/safety test `pytest source/leisaac/test/test_openarm_vr_v2_core.py`가 수정 없이 통과한다.
2. cutover 전에는 candidate→V1, canonical→기존 V2가 parse되고 candidate가 local asset path/hash를 선택한다. cutover 후에는 canonical→V1, legacy→기존 V2가 parse되는지 다시 검사한다.
3. action manager가 정확히 16D이고 action term 순서가 기존과 같다.
4. runtime joint 이름·순서·limit·axis가 v1.0 profile과 일치한다.
5. fixed-base, EE body ID와 Jacobian row/column mapping이 finite-difference 검사와 일치한다.
6. reset 후 robot root와 모든 Physics01 prop pose가 재현된다.
7. 고정 target 120 step에서 arm-joint RMS velocity가 `0.02 rad/s` 이하이고 gripper speed는 별도 `m/s` 기준으로 보고한다.
8. 초기 contact는 base/support allowlist에만 있고, 금지 contact의 최소 separation이 `-1 mm` 이상이다.
9. 좌우 gripper command joint와 follower가 함께 움직이고, joint state가 `0.044 → 0.0 → 0.044 m`로 변하며 실제 fingertip gap이 close 때 감소하고 reopen 때 증가한다.
10. 두 팔의 reachable 5 cm target이 120 step 이내에 최소 4.5 cm 움직인다.
11. physical arm-joint velocity가 profile limit의 `105%`를 넘지 않는다.
12. 한쪽 QP 실패가 반대쪽 팔을 멈추지 않는다.
13. 기존 scene reset과 robot↔table/props contact 검사가 통과한다.

기존 checker의 `/Scene/openarm_v20`, `*_ee_base_link`, revolute gripper body 이름은 config/asset schema 값으로 바꾼다. contact view도 composed runtime robot prim path를 사용한다.

`scripts/tools/openarm_v2_fk_parity_check.py`에는 `--task`를 추가해 Physics01 v1 task를 직접 검사한다. 양팔 각각 root-frame `+X/+Y/+Z` 3 cm와 `+roll/+pitch/+yaw` 0.12 rad를 명령하고 translation cosine `>=0.90`, rotation cosine `>=0.80`을 요구한다.

검증 명령:

```bash
PYTHONPATH=source/leisaac python -m pytest -q \
  source/leisaac/test/test_openarm_vr_v2_core.py

python -u scripts/tools/openarm_v2_teleop_check.py \
  --task LeIsaac-OpenArm-Bimanual-Physics01-V1-QuestV2-v0 \
  --headless --device cuda:0

python -u scripts/tools/openarm_v2_fk_parity_check.py \
  --task LeIsaac-OpenArm-Bimanual-Physics01-V1-QuestV2-v0 \
  --headless --device cuda:0
```

현재 자동화 환경에서는 NVIDIA GPU가 노출되지 않았지만 CPU PhysX로 전체 headless integration과 six-axis parity를 통과했다. 실제 GPU가 보이는 호스트에서 같은 검사를 `cuda:0`으로 재실행하고 Quest 3 실기 gate를 통과해야 하드웨어 검증까지 완료된다.

### C. Quest 3 실기 검증

headless 검사를 모두 통과한 뒤 먼저 candidate ID로 실행한다. 실기 gate 통과와 canonical cutover 후에는 `--task`만 기존 `LeIsaac-OpenArm-Bimanual-Physics01-QuestV2-v0`로 되돌리며 나머지 명령은 같다.

```bash
python -u scripts/environments/teleoperation/teleop_se3_agent.py \
  --task LeIsaac-OpenArm-Bimanual-Physics01-V1-QuestV2-v0 \
  --teleop_device quest3-controller-v2 \
  --num_envs 1 --device cuda:0 --headless \
  --openarm_v2_debug_log logs/openarm_v1_physics01_vr.jsonl
```

실기 체크:

- HMD yaw calibration 후 머리를 움직여도 robot target이 떠다니지 않는다.
- 좌우 controller의 전후·좌우·상하가 해당 hand의 같은 축 움직임으로 보인다.
- 양 손목 roll/pitch/yaw 방향이 controller와 일치한다.
- squeeze를 10회 release/re-engage해도 re-clutch 첫 frame의 hand target jump가 `2 cm`, `5 deg`를 넘지 않는다.
- controller tracking을 각 손 3회 가렸다 복구해도 해당 팔만 HOLD되고 release/re-clutch 전에는 움직이지 않는다.
- trigger가 각 v1.0 parallel gripper를 독립적으로 열고 닫는다.
- 양팔을 동시에 5분, 즉 60 Hz 기준 18,000 control frame 이상 조작한다. 정상 tracking 구간의 NaN, joint-limit violation, unexpected QP fault, 관통 및 XR session loss는 각각 0건이어야 한다.

debug JSONL 첫 record에는 task ID, robot model, root USD hash, dependency-manifest hash와 schema version을 기록한다. 실기 보고서에는 의도적으로 만든 tracking-loss 횟수, side-local HOLD 진입 횟수, 정상 복구 횟수와 re-clutch jump 최대값을 남긴다.

## 6. 카메라·데이터 수집 계획과의 연결

v1.0 교체가 먼저 완료되어야 3카메라 수집 계획을 실행한다.

- 가슴 카메라는 v1.0 torso의 실제 장착 link로 parent path를 다시 확정한다.
- wrist 카메라는 v1.0에서 실제 브래킷이 붙은 link를 확인한다. v2의 `*_ee_base_link` 경로를 복사하지 않는다.
- 카메라 config는 asset schema manifest의 resolved torso/distal-link path를 사용한다. startup에서 세 parent prim이 존재하고 실제 rigid body인지, calibrated extrinsic의 기준 parent와 같은지 assert한다.
- sidecar에 `robot_model=openarm_v1.0`, root USD SHA-256, source commit을 필수로 저장한다.
- v2 revolute gripper state/action과 v1.0 prismatic gripper state/action은 같은 dataset으로 합치지 않는다.
- 기존 raw 16D action은 14 arm joint target과 2 binary gripper command로 유지된다. measured joint state는 arm 14 + finger 4인 18D이고, expanded gripper joint target/state 단위가 v2의 rad에서 v1.0의 m로 바뀐다는 사실을 metadata에 분리해 기록한다.

## 7. 구현 순서

1. 공식 v1.0 asset과 모든 dependency를 local path에 고정하고 hash/schema manifest를 만든다.
2. 원본 Physics01을 reference하고 내장 v2를 inactive로 author한 background composition layer를 만든다.
3. V1 candidate config에서 local v1.0을 `/Scene/openarm_v1`에 spawn한다.
4. root pose, resolved EE frame, gripper, actuator 및 self-collision을 v1.0 profile로 바꾼다.
5. controller의 joint-limit source를 ordered bimanual v1.0 table로 통합한다.
6. candidate task ID와 config/registration smoke test를 추가한다.
7. preflight와 기존 두 integration checker를 composed Physics01 v1.0에 맞게 일반화한다.
8. GPU host에서 static, hold, gripper, motion, parity, collision, reset 검사를 순서대로 통과시킨다.
9. Quest 3 실기 검증을 수행하고 debug JSONL을 보존한다.
10. canonical task ID를 V1 config로 전환하고 기존 V2 config에는 legacy ID를 부여한다.
11. setup/progress 문서와 3카메라 수집 metadata schema를 v1.0 기준으로 갱신한다.

각 단계가 실패하면 다음 단계로 진행하지 않는다. pose scale이나 filter를 먼저 튜닝하지 말고 asset schema → root placement → joint/EE mapping → physics → VR 순서로 원인을 좁힌다.

## 8. 변경 대상 파일

| 파일 | 변경 내용 |
|---|---|
| `assets/scenes/Collected_physics01/physics01_v1_background.usda` | 원본 scene reference와 내장 v2 inactive override |
| `assets/robots/openarm_v1.0/` | 검증·수집에 쓸 local v1.0 asset bundle |
| `assets/robots/openarm_v1.0/OPENARM_V1_ASSET_MANIFEST.json` | 전체 dependency hash, import option, resolved schema |
| `source/leisaac/leisaac/tasks/lift_cube/openarm_collected_physics01_env_cfg.py` | 기존 V2 유지, V1 candidate scene/env와 frame·gripper·physics profile |
| `source/leisaac/leisaac/tasks/lift_cube/__init__.py` | candidate 등록, 검증 후 canonical/legacy ID 전환 |
| `source/leisaac/leisaac/assets/robots/openarm.py` | production local v1 asset과 ordered bimanual v1 joint-limit source |
| `source/leisaac/leisaac/devices/openarm_vr_v2/isaac_qp_controller.py` | v1 limit source 사용; QP 알고리즘 유지 |
| `source/leisaac/leisaac/devices/openarm_vr_v2/device.py` | v1 schema fail-fast 문구와 limit 검증 |
| `scripts/tools/usd_teleop_preflight.py` | prismatic v1 gripper 및 단일 active articulation 검사 |
| `scripts/tools/openarm_v2_teleop_check.py` | v1 root·EE·gripper·collision 검사 |
| `scripts/tools/openarm_v2_fk_parity_check.py` | `--task`와 Physics01 v1 parity 검사 |
| `OPENARM_QUEST3_SETUP.md` | v1 Physics01 실행·검증 절차 |
| `OPENARM_VR_TELEOP_V2_PROGRESS.md` | robot model 교체 결과와 측정값 |
| `OPENARM_THREE_CAMERA_DATA_COLLECTION_PLAN.md` | v1 camera parent 및 metadata 단위 |

새 controller와 새 action schema는 만들지 않는다. candidate task ID는 검증 기간에만 추가한다.

## 9. 완료 기준

다음 조건이 모두 충족되어야 교체 완료다.

- Physics01에 active OpenArm v1.0 articulation이 정확히 하나 있고 v2 physics prim은 없다.
- 실행 시 local v1.0 root와 모든 dependency hash가 manifest와 일치하고 외부 Nucleus 참조가 없다.
- root placement와 초기 posture가 table/scene과 관통하지 않는다.
- 16D raw action, 18D processed action 및 18D measured state 순서가 유지된다.
- v1.0 EE body, 좌우 joint limit 및 prismatic gripper가 runtime schema와 일치한다.
- 기존 mapping/filter/QP/safety unit test가 모두 통과한다.
- hold, gripper, bilateral motion, six-axis parity, collision 및 reset headless 검사가 모두 통과한다.
- Quest 3에서 방향, wrist rotation, clutch, tracking-loss recovery, 양팔 동시 조작이 실기 기준을 통과한다.
- candidate 검증 후 canonical task가 V1 config, legacy task가 기존 V2 config로 각각 등록된다.
- setup/data 문서가 v1.0 model과 gripper 단위를 일관되게 설명한다.
