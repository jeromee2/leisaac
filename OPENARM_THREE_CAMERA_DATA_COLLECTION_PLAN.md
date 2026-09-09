# OpenArm 3카메라 텔레옵 데이터 수집 구현 계획

## 1. 목표와 범위

Physics01 Isaac Sim 환경에서 Quest 3로 OpenArm 양팔을 조종하면서 실제 로봇과 같은 세 카메라 입력을 포함한 시연 데이터를 수집한다.

- 가슴 RGB 카메라 1개
- 왼팔 RGB 카메라 1개
- 오른팔 RGB 카메라 1개
- 현재 관절 상태와 양팔 말단 상태
- 같은 시점에 적용할 16차원 텔레옵 action
- 에피소드 성공 여부
- task, 주기, 카메라 calibration 및 feature 순서를 재현할 메타데이터

상세한 실패 이유 분류, depth, segmentation, 촉각 및 도메인 랜덤화는 첫 수집 범위에서 제외한다. 먼저 RGB·state·action의 정렬과 재생 가능성을 검증한다.

## 2. 확정된 현재 구조

| 항목 | 현재 값 |
|---|---|
| Task | `LeIsaac-OpenArm-Bimanual-Physics01-QuestV2-v0` |
| Teleop device | `quest3-controller-v2` |
| Physics | 120 Hz |
| Control | 60 Hz, `decimation=2` |
| Raw action | 16D |
| Processed action | 18D |
| Joint observation | 18D |
| 원본 포맷 | IsaacLab streaming HDF5 |
| 학습 포맷 | LeRobot v3 |

기존 코드에는 상위 single-arm task의 `wrist`, `front` 카메라가 정의되어 있지만 Physics01의 USD 경로와 맞지 않아 현재 의도적으로 삭제된다. 이 삭제는 유지하고 Physics01 전용 세 카메라를 추가한다.

XR 경로는 `remove_camera_configs(env_cfg)`로 모든 추가 카메라를 제거한다. 사용 중인 IsaacLab 버전은 XR과 추가 RTX 카메라의 rendering conflict 및 성능 문제를 명시하므로, 라이브 3카메라 수집은 반드시 go/no-go 성능 관문을 통과해야 한다.

## 3. 최종 데이터의 시간 의미

한 번의 `env.step(action_t)`에서 기존 recorder의 순서는 다음과 같다.

```text
ActionManager.process_action(action_t)
    ↓
pre-step recorder
    ├─ obs_t
    └─ raw action_t
    ↓
physics step
    ↓
post-step recorder
    ├─ states_(t+1)
    └─ processed_actions_t
```

따라서 학습 pair는 `obs_t → raw action_t`이다. HDF5의 같은 배열 index에 들어 있는 `states`는 다음 상태인 `states_(t+1)`이다. 검사기와 변환기는 이 관계를 기준으로 동작해야 하며 `obs[i]`를 `states[i]`와 같은 시점이라고 해석하면 안 된다.

카메라까지 포함해 학습 변환 입력으로 인정하는 HDF5의 필수 key는 다음과 같다.

```text
data/demo_N/actions
data/demo_N/processed_actions
data/demo_N/obs/chest
data/demo_N/obs/left_wrist
data/demo_N/obs/right_wrist
data/demo_N/obs/joint_pos
data/demo_N/obs/joint_vel
data/demo_N/obs/joint_pos_rel
data/demo_N/obs/joint_vel_rel
data/demo_N/obs/actions
data/demo_N/obs/ee_frame_state
data/demo_N/obs/joint_pos_target
data/demo_N/states/...
```

XR 카메라가 no-go인 경우 처음 수집한 `openarm_raw.hdf5`에는 세 RGB key가 없다. 이 파일은 state/action 원본이며, 11절의 offline render가 만든 `*.with_cameras.hdf5`가 위 key를 모두 가진 정식 변환 입력이다.

상위 observation에 이미 있는 `joint_pos_rel`, `joint_vel_rel`, `actions`도 원본 HDF5에 남긴다. LeRobot 변환에서는 필요한 feature만 선택한다.

## 4. Action과 state schema

### 4.1 Raw action 16D

```text
0..6   left_arm_joint1..7 target [rad]
7      left_gripper command: +1=open, -1=close
8..14  right_arm_joint1..7 target [rad]
15     right_gripper command: +1=open, -1=close
```

### 4.2 Processed action 18D

두 1D gripper command가 각 손의 finger joint target 두 개로 펼쳐진다.

```text
left arm target 7
+ left finger target 2
+ right arm target 7
+ right finger target 2
= 18D
```

Physics01의 OpenArm v1.0 open target은 양쪽 `0.044 m`이고 close target은 양쪽 `0.0 m`이다. `processed_actions`는 진단과 replay에 보존하지만, 최초 정책의 출력 target은 텔레옵 장치가 생성한 raw 16D로 고정한다.

### 4.3 Observation state 18D

```text
left arm measured position 7
+ right arm measured position 7
+ left finger measured position 2
+ right finger measured position 2
= 18D
```

LeRobot action 이름이 현재처럼 `dim_0`부터 `dim_15`로 저장되지 않게 한다. Physics01 config에 다음 16개 이름을 명시하고 `build_feature_from_env()`가 이 값이 있을 때 우선 사용하도록 한다.

```text
left_joint1.target ... left_joint7.target
left_gripper.open_close
right_joint1.target ... right_joint7.target
right_gripper.open_close
```

구성한 이름 수와 `env.action_manager.total_action_dim`이 다르면 수집 또는 변환을 즉시 실패시킨다.

## 5. 카메라 부착 위치

실제 카메라가 장착된 링크와 같은 움직임을 갖도록 센서를 부모 link의 자식 prim으로 생성한다.

| Key | 부모 link | Physics01 prim path |
|---|---|---|
| `chest` | `openarm_body_link` | `{ENV_REGEX_NS}/Scene/openarm_v1/openarm_body_link/chest_camera` |
| `left_wrist` | `openarm_left_link7` | `{ENV_REGEX_NS}/Scene/openarm_v1/openarm_left_link7/left_wrist_camera` |
| `right_wrist` | `openarm_right_link7` | `{ENV_REGEX_NS}/Scene/openarm_v1/openarm_right_link7/right_wrist_camera` |

팔 카메라가 마지막 손목 회전과 같이 돌면 `*_ee_base_link`를 사용한다. 손목 회전 전 전완 브래킷에 고정되어 있으면 실제 브래킷이 붙은 전완 link로 변경한다. 이 선택은 장착 사진 또는 CAD로 확인하기 전까지 완료로 처리하지 않는다.

각 카메라에서 측정하고 기록할 값:

- 부모 link에서 optical center까지의 translation, m
- 부모 link에서 optical frame까지의 quaternion, `wxyz`
- 해상도
- 수평·수직 FOV 또는 focal length와 aperture
- 실제 운용 FPS
- 측정 출처와 날짜

보기 좋은 임의 pose를 최종값으로 사용하지 않는다. 초기 smoke test에 임시값을 쓰는 경우 이름과 로그에 `UNCALIBRATED`를 남기고 학습용 수집을 막는다.

## 6. 환경 카메라 및 observation 구현

대상 파일:

- `source/leisaac/leisaac/tasks/lift_cube/openarm_collected_physics01_env_cfg.py`

### 6.1 Scene config

필요한 import를 추가한다.

```python
import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import FrameTransformerCfg, TiledCameraCfg
```

`OpenArmBimanualPhysics01V1SceneCfg`에 `chest`, `left_wrist`, `right_wrist`라는 `TiledCameraCfg`를 각각 추가한다. 첫 라이브 검증에서는 제어 index마다 새 영상을 얻기 위해 세 카메라 모두 다음 설정을 사용한다.

```text
data_types=["rgb"]
width=640
height=480
update_period=1/60 s
```

실제 카메라가 30 FPS여도 원본 정렬 검증은 60 Hz 렌더로 수행하고 LeRobot 변환에서 30 Hz로 선택한다. XR 성능 관문을 통과하지 못하면 11절의 offline render 경로를 사용한다.

다음 기존 삭제는 제거하지 않는다.

```python
delete_attribute(self, "wrist")
delete_attribute(self, "front")
```

이 삭제가 없으면 상위 클래스의 잘못된 경로를 가진 두 카메라까지 남아 총 다섯 카메라가 된다.

### 6.2 Observation config

중첩 config를 명시적으로 재바인딩한다.

```python
@configclass
class OpenArmBimanualPhysics01ObservationsCfg(SingleArmObservationsCfg):
    @configclass
    class PolicyCfg(SingleArmObservationsCfg.PolicyCfg):
        chest = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("chest"), "data_type": "rgb", "normalize": False},
        )
        left_wrist = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("left_wrist"), "data_type": "rgb", "normalize": False},
        )
        right_wrist = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("right_wrist"), "data_type": "rgb", "normalize": False},
        )

    policy: PolicyCfg = PolicyCfg()

    def __post_init__(self):
        super().__post_init__()
        delete_attribute(self.policy, "wrist")
        delete_attribute(self.policy, "front")
```

상위 `wrist`, `front` observation 삭제도 유지한다. 환경 생성 후 정확히 세 camera sensor와 세 camera observation만 존재하는지 검사한다.

## 7. XR 라이브 카메라 관문

대상 파일:

- `scripts/environments/teleoperation/teleop_se3_agent.py`

`--record`와 카메라 활성화를 분리하기 위해 다음 CLI를 추가한다.

```text
--record_cameras none|chest|all
--camera_probe none|chest|all
```

두 인자의 기본값은 `none`이다. 일반 수집에서 `--record_cameras chest|all`은 `--record` 및 `--enable_cameras`가 함께 있을 때만 허용한다. `--camera_probe chest|all`은 `--enable_cameras`를 요구하고 `--record`, `--resume` 및 `--record_cameras chest|all`과 함께 쓸 수 없다. 조건을 만족하지 않으면 env 생성 전에 실패시킨다.

환경에 실제로 남길 camera set은 probe일 때 `camera_probe`, 그 밖에는 `record_cameras` 값이다.

- `none`: `remove_camera_configs(env_cfg)`를 호출한다.
- `chest`: `left_wrist`, `right_wrist` scene/observation config를 삭제하고 가슴 카메라만 유지한다.
- `all`: Physics01의 세 카메라를 모두 유지한다.

```python
active_camera_set = (
    args_cli.camera_probe
    if args_cli.camera_probe != "none"
    else args_cli.record_cameras
)
if args_cli.xr and active_camera_set == "none":
    env_cfg = remove_camera_configs(env_cfg)
elif active_camera_set == "chest":
    delete_attribute(env_cfg.scene, "left_wrist")
    delete_attribute(env_cfg.scene, "right_wrist")
    delete_attribute(env_cfg.observations.policy, "left_wrist")
    delete_attribute(env_cfg.observations.policy, "right_wrist")
```

먼저 `--camera_probe chest`로 60초, 통과하면 `--camera_probe all`로 60초를 측정한다. probe 모드는 HDF5 recorder를 만들지 않고, 0·20·40초에 phase 안내를 출력하며 60초에 자동 종료한다. 각 probe는 처음 20초 정지, 다음 20초 왼팔과 오른팔을 차례로 움직이기, 마지막 20초 양팔 움직이기로 고정한다. 본 수집 loop에는 probe 계측과 자동 종료가 실행되지 않는다.

각 60초 측정의 go 조건:

- crash, 흰 화면, 검은 화면 또는 XR session loss가 없다.
- `env.step()` 처리율이 평균 54 Hz 이상이고 제어 loop 시간 p95가 25 ms 이하이다.
- 공개 `Camera.frame`을 매 control step 읽는다.
- 각 활성 카메라의 `frame` 값이 전체 control step의 90% 이상에서 증가한다.
- RGB의 luminance 표준편차가 2 이상이고, 움직임 구간의 연속 frame 평균 절대 차이가 정지 구간보다 크다.

측정값은 camera set별 `logs/openarm_camera_live_probe_chest.json` 또는 `logs/openarm_camera_live_probe_all.json`에 camera set, 해상도, 평균 Hz, p95 loop ms, camera frame 증가 횟수, 영상 통계 및 결과를 기록한다. 같은 이름이 이미 있으면 덮어쓰지 않고 timestamp suffix를 붙인다.

`chest` 또는 `all`에서 하나라도 실패하면 라이브 카메라 경로는 no-go다. 이후 수집은 `--record --record_cameras none`으로 카메라 없는 state/action HDF5를 만들고 11절에서 영상을 추가한다.

## 8. 성공/실패 입력

Quest V2에서는 키보드 `N` callback이 등록되지 않으므로 보조 키보드 방식을 사용하지 않는다. 현재 사용하지 않는 오른쪽 컨트롤러 A 버튼을 성공 종료에 사용한다.

변경 파일:

- `source/leisaac/leisaac/devices/openarm_vr_v2/openxr_source.py`
- `scripts/environments/teleoperation/teleop_se3_agent.py`

구현:

1. `poll()`에서 left와 right controller를 `_handle_button_edges()`에 전달한다.
2. `_right_a_pressed` edge state를 유지한다.
3. 오른쪽 A rising edge에서 `SUCCESS` callback을 호출한다.
4. Quest V2 callback map에 `"SUCCESS": reset_task_success`를 등록한다.
5. 조작 안내에 `Right A: mark success and reset`을 추가한다.
6. 기존 왼쪽 Y는 실패 종료 및 reset으로 유지한다.

`reset_task_success()` 자체는 success/reset boolean flag만 설정한다. main loop가 다음 순서로 처리하는 현재 구조를 유지한다.

```text
should_reset_task_success
  → manual_terminate(env, True)
should_reset_recording_instance
  env.reset()
    └─ pre-reset recorder가 success=True 저장
  manual_terminate(env, False)
```

테스트는 fake input device로 A 버튼 rising edge 한 번당 callback이 정확히 한 번 호출되고, 누른 상태 유지와 release에서는 추가 호출되지 않는지 확인한다.

MVP에서 저장하는 outcome은 HDF5 episode의 `success: bool`뿐이다. 실패 이유의 자유 텍스트나 분류는 이번 범위에 포함하지 않는다.

## 9. 재현 메타데이터

Streaming HDF5는 episode seed만 보존하므로 동일 stem의 JSON sidecar를 추가한다.

```text
datasets/openarm_three_camera.hdf5
datasets/openarm_three_camera.meta.json
```

HDF5 root의 `data.attrs["env_args"].env_name`에도 task ID가 들어가도록 env 생성 전에 `env_cfg.env_name = args_cli.task`를 설정한다. 검사기는 이 값과 sidecar의 `task`를 비교한다.

필수 JSON field:

```text
schema_version
task
teleop_device
physics_hz
control_hz
raw_dataset_hz
captured_camera_set
camera_render_mode
action_names[16]
state_names[18]
sessions[]
  ├─ created_at
  ├─ seed
  ├─ git_commit
  └─ git_dirty
cameras.chest
cameras.left_wrist
cameras.right_wrist
  ├─ parent_prim
  ├─ translation_m
  ├─ rotation_wxyz
  ├─ convention
  ├─ width
  ├─ height
  ├─ update_hz
  ├─ focal_length
  ├─ horizontal_aperture
  └─ calibration_source
```

`captured_camera_set`은 `none`, `chest`, `all` 중 하나이고, `camera_render_mode`는 카메라가 없으면 `none`, 라이브 수집이면 `live`, offline 증강 결과면 `offline`이다. 검사기는 이 두 값과 실제 HDF5 camera key 구성을 함께 확인한다.

`teleop_se3_agent.py`가 sidecar를 작성한다. 새 수집에서는 env 생성 전 임시 JSON을 만들고, env와 action manager 생성 후 resolved sensor 및 dimension을 검증한 다음 첫 `env.step()` 전에 `os.replace()`로 최종 이름에 원자적으로 반영한다. 검증이나 rename이 실패하면 한 frame도 기록하지 않는다.

`--resume`에서는 env/HDF5를 열기 전에 기존 sidecar의 schema, task, action/state 이름, 주기 및 camera calibration을 비교한다. sidecar가 없거나 하나라도 다르면 이어쓰기를 거부한다. env 생성 후에는 resolved dimension을 한 번 더 검사한다. session별 `created_at`, seed, git 상태는 `sessions[]`에 추가하고 같은 temp+replace 방식으로 갱신한다.

## 10. HDF5 검사기

새 파일:

- `scripts/tools/check_openarm_camera_dataset.py`

입력은 HDF5와 같은 stem의 metadata JSON이다. `--expect-cameras yes|no`를 필수로 받고 episode별로 검사하며, 하나라도 실패하면 exit code 1을 반환한다.

- sidecar `task`와 HDF5 root `env_args.env_name`이 같은가
- `actions.shape[-1] == 16`
- `processed_actions.shape[-1] == 18`
- `obs/joint_pos.shape[-1] == 18`
- `--expect-cameras yes`이면 세 RGB tensor가 `[N, 480, 640, 3]`, `uint8`인가
- `--expect-cameras no`이면 세 RGB key가 없는가
- `actions`, `processed_actions`, 모든 `obs`, `states`의 첫 차원 길이가 같은가
- float tensor에 NaN 또는 Inf가 없는가
- raw gripper 값이 허용값 `{-1,+1}`인가
- episode의 success attr이 존재하는가
- action 이름과 state 이름의 수가 dimension과 맞는가
- `--expect-cameras yes`일 때 첫·중간·마지막 이미지를 contact sheet로 만들 수 있는가

시간 의미 검증은 `obs[i] → actions[i] → states[i]`가 각각 `t`, `t`, `t+1`이라는 schema 검사로 보고한다. 동일 index라는 이유로 obs와 states가 같은 값이어야 한다고 검사하지 않는다.

## 11. XR 카메라 no-go 시 offline render

기존 `scripts/environments/teleoperation/replay.py --replay_mode state`는 robot joint position을 action으로 다시 적용할 뿐 전체 scene state를 frame별로 복원하지 않으므로 카메라 생성에 사용하지 않는다.

### 11.1 수집 전 state coverage preflight

새 파일:

- `scripts/tools/openarm_camera_preflight.py`

라이브 probe보다 먼저 Physics01 USD의 RigidBody/Articulation prim과 `env.scene`에 등록된 articulation/rigid object를 비교한다. 등록된 articulation의 root 아래 link들은 그 articulation state로 포괄되므로 개별 미등록 rigid object로 오판하지 않는다. 그 밖의 task 관련 movable prim이 `scene.get_state()`에 포함되지 않으면 해당 object를 task config에 등록한 뒤 다시 검사한다. 의도적으로 제외하는 prim은 경로와 제외 사유를 명시한 allowlist로만 통과시킨다.

preflight는 임의의 상태를 한 frame 기록해 `initial_state`와 `states`에 robot 및 모든 task-relevant movable object가 존재하는지도 검사한다. 이 검사를 통과하기 전에는 라이브·오프라인 어느 경로로도 본 수집을 시작하지 않는다.

### 11.2 Camera-augmented HDF5 생성

새 파일:

- `scripts/convert/render_hdf5_cameras.py`

입력과 출력은 다음으로 고정한다.

```text
input:  openarm_raw.hdf5
output: openarm_raw.with_cameras.hdf5
```

출력은 원본 HDF5를 임시 작업 디렉터리 안에 복사한 뒤 각 episode의 `obs` 아래에 `chest`, `left_wrist`, `right_wrist` dataset을 같은 60 Hz index와 길이로 추가한다. LeRobot을 직접 생성하지 않는다. 출력 sidecar도 그 디렉터리에 복사·갱신하고 `captured_camera_set=all`, `camera_render_mode=offline`, 원본 SHA-256, 렌더 시각 및 렌더 코드 commit을 추가한다.

최종 두 파일과 별도의 `openarm_raw.with_cameras.ready.json`을 한 묶음으로 다룬다. HDF5와 sidecar를 먼저 최종 이름으로 rename하고, 두 파일의 SHA-256과 schema version을 담은 ready marker를 마지막에 원자적으로 rename한다. 소비자는 ready marker가 존재하고 두 hash가 일치할 때만 결과를 연다. 시작 시 marker 없는 orphan 최종 파일은 `*.orphan.<timestamp>`로 격리한 뒤 재실행할 수 있게 한다. 유효한 ready marker가 이미 있으면 덮어쓰지 않고 실패한다.

절차:

1. XR 없이 Physics01 task와 세 카메라를 `--enable_cameras`로 구성하고, env 생성 전에 `env_cfg.use_teleop_device("quest3-controller-v2")`로 `MISSING` action term 네 개를 초기화한다.
2. env 생성 전에 `env_cfg.recorders={}`, `env_cfg.terminations={}`로 비활성화한다.
3. env는 처음 한 번만 reset한다. frame별 복원에는 side effect가 있는 `env.reset_to()` 대신 `env.scene.reset_to(source_state_i, env_ids, is_relative=True)`를 사용한다.
4. HDF5 episode의 `initial_state`와 `states`를 읽는다.
5. 관측 index `i=0`은 `initial_state`, `i>0`은 `states[i-1]`로 복원한다.
6. 각 index에서 state 복원 후 `env.sim.forward()`를 호출한다. 현재 pinned IsaacLab에는 `render_context`가 없으므로 직접 접근하지 않는다. 향후 버전에 `env.sim.render_context.reset_transform_cadence()`가 둘 다 존재할 때만 capability check 뒤 호출한다.
7. `env.sim.render()`로 USD transform과 RTX/Replicator 출력을 먼저 동기화한다. 그 다음 각 camera sensor에 `camera.update(1 / source_fps, force_recompute=True)`를 호출해 명시적으로 새 buffer를 만들고, 각 카메라의 공개 `Camera.frame`이 index마다 정확히 1 증가했는지 확인한 뒤 RGB를 읽는다. 증가하지 않으면 즉시 실패한다.
8. RGB를 임시 HDF5의 동일 episode와 index에 추가한다.
9. 임시 HDF5와 sidecar가 검사기를 통과하면 두 파일을 최종 이름으로 옮긴 뒤 ready marker를 마지막에 commit한다. 중간 실패나 orphan 복구 과정에서도 입력 원본은 수정하지 않는다.

복원 직후 원본 `obs/joint_pos[i]`와 scene의 measured joint position 최대 오차가 `1e-5 rad` 이하인지 모든 frame에서 검사한다. 대표 frame을 같은 state로 두 번 렌더링한 RGB는 완전 hash 일치를 요구하지 않고 채널당 평균 절대 차이 2.0 이하를 요구한다. 이 허용치를 넘으면 temporal rendering 설정을 고정하거나 원인을 해결하기 전까지 출력 파일을 완료 처리하지 않는다.

생성 후 `check_openarm_camera_dataset.py --expect-cameras yes`를 통과한 파일만 LeRobot 변환 입력으로 사용한다.

## 12. 60 Hz HDF5에서 30 Hz LeRobot 변환

대상 파일:

- `scripts/convert/isaaclab2lerobotv3.py`
- `source/leisaac/leisaac/utils/robot_utils.py`

변환기에 다음 인자를 추가한다.

```text
--source_fps 60
--fps 30
--skip_initial_frames 0
```

첫 구현은 `source_fps % fps == 0`인 정수 비율만 허용한다.

변환기는 입력 HDF5와 같은 stem의 sidecar를 필수로 읽고, 실행 전에 다음 source contract를 강제한다.

- CLI `source_fps`가 sidecar `raw_dataset_hz`와 같은가
- sidecar `task`, CLI `task_name`, HDF5 `env_args.env_name`이 모두 같은가
- `captured_camera_set == all`이고 `camera_render_mode`가 `live` 또는 `offline`인가
- action/state 이름의 수와 순서가 HDF5 dimension과 같은가
- offline 결과이면 원본 HDF5 SHA-256과 ready marker provenance가 존재하고 실제 hash와 맞는가

하나라도 다르면 output repo를 만들기 전에 실패시킨다.

```python
stride = source_fps // fps
selected_indices = range(skip_initial_frames, num_frames, stride)
```

기본값은 `skip_initial_frames=0`이다. 현재 하드코딩된 첫 5 frame 생략은 제거한다. 명시적으로 frame을 건너뛸 때는 `skip_initial_frames`가 stride의 배수인지 검사한다.

60→30 Hz 기본 선택은 `0, 2, 4, ...`이고 출력 길이는 다음과 같다.

```text
ceil((num_frames - skip_initial_frames) / stride)
```

선택된 각 `i`에서 LeRobot pair는 `obs[i]`와 `raw actions[i]`다. 영상도 `obs[i]`에 들어 있는 frame을 사용한다. LeRobot timestamp는 출력 episode index를 30 Hz로 생성하며 원본 HDF5 index는 변환 로그에 남긴다.

변환 결과 feature:

```text
observation.images.chest
observation.images.left_wrist
observation.images.right_wrist
observation.state
action
task
```

변환은 `episode.success == True`인 episode만 사용한다. Parquet frame 수와 각 MP4 frame 수가 같아야 한다.

## 13. 단계별 실행 순서

### A. Calibration 확보

- 실제 세 카메라의 parent link, extrinsic, intrinsic 및 FPS를 측정한다.
- Physics01 상수와 metadata schema에 입력한다.
- 카메라 pose가 `UNCALIBRATED`이면 학습용 수집을 금지한다.

### B. State coverage와 headless 카메라 preflight

- `openarm_camera_preflight.py`로 모든 task-relevant movable object가 `scene.get_state()`에 포함되는지 확인한다.
- XR 없이 세 카메라를 생성한다.
- 각 shape, dtype, prim path를 확인한다.
- 왼팔만 움직였을 때 왼팔 카메라만 해당 link를 따라가는지 확인한다.
- 오른팔과 가슴 카메라도 같은 방식으로 확인한다.

### C. XR 라이브 go/no-go

- 가슴 1개로 60초를 측정한다.
- 통과하면 세 카메라로 60초를 측정한다.
- 통과하면 라이브 HDF5 경로, 실패하면 state/action HDF5 + offline render 경로로 확정한다.

### D. 에피소드 기능 검증

- 오른쪽 A로 성공 3회, 왼쪽 Y로 실패 1회를 기록한다.
- 라이브 경로는 원본에 `--expect-cameras yes` 검사기를 실행한다.
- no-go 경로는 원본에 `--expect-cameras no`를 실행하고, offline render 후 `*.with_cameras.hdf5`에 `--expect-cameras yes`를 실행한다.
- 성공 3개와 실패 1개의 attr 및 contact sheet를 확인한다.

### E. LeRobot 변환 검증

- 성공 episode만 30 Hz로 변환한다.
- 선택 index가 `0,2,4,...`인지 변환 로그로 확인한다.
- 영상 세 개, state, action의 frame 수와 feature 이름을 검사한다.
- 변환된 성공 episode 3개를 viewer로 재생한다.

### F. 소규모 학습

- 고정 초기 조건의 성공 시연 20–50개로 학습하고 같은 고정 조건의 rollout 20회 중 18회 이상 성공하는지 확인한다.
- 이후 물체 위치와 방향을 바꿔 수집한다.
- 학습에 쓰지 않은 초기 조건에서 20회 rollout한다.
- 가슴 카메라만 쓴 정책과 세 카메라 정책의 성공률을 비교한다.

## 14. 실행 명령 초안

XR 라이브 camera probe:

```bash
python -u scripts/environments/teleoperation/teleop_se3_agent.py \
  --task LeIsaac-OpenArm-Bimanual-Physics01-QuestV2-v0 \
  --teleop_device quest3-controller-v2 \
  --num_envs 1 \
  --device cuda:0 \
  --headless \
  --enable_cameras \
  --camera_probe chest

# chest 통과 후 마지막 인자만 all로 바꿔 다시 실행
```

라이브 카메라 경로가 go일 때 새 HDF5 수집:

```bash
python -u scripts/environments/teleoperation/teleop_se3_agent.py \
  --task LeIsaac-OpenArm-Bimanual-Physics01-QuestV2-v0 \
  --teleop_device quest3-controller-v2 \
  --num_envs 1 \
  --device cuda:0 \
  --headless \
  --enable_cameras \
  --record \
  --record_cameras all \
  --dataset_file ./datasets/openarm_three_camera.hdf5
```

XR 카메라 no-go일 때 state/action 원본 수집:

```bash
python -u scripts/environments/teleoperation/teleop_se3_agent.py \
  --task LeIsaac-OpenArm-Bimanual-Physics01-QuestV2-v0 \
  --teleop_device quest3-controller-v2 \
  --num_envs 1 \
  --device cuda:0 \
  --headless \
  --record \
  --record_cameras none \
  --dataset_file ./datasets/openarm_raw.hdf5
```

no-go 원본에 오프라인 카메라 추가:

```bash
python -u scripts/convert/render_hdf5_cameras.py \
  --task LeIsaac-OpenArm-Bimanual-Physics01-QuestV2-v0 \
  --input ./datasets/openarm_raw.hdf5 \
  --output ./datasets/openarm_raw.with_cameras.hdf5 \
  --enable_cameras \
  --headless
```

LeRobot 변환(라이브 go 경로):

```bash
python -u scripts/convert/isaaclab2lerobotv3.py \
  --task_name LeIsaac-OpenArm-Bimanual-Physics01-QuestV2-v0 \
  --hdf5_root ./datasets \
  --hdf5_files openarm_three_camera.hdf5 \
  --repo_id local/openarm-three-camera \
  --source_fps 60 \
  --fps 30 \
  --skip_initial_frames 0
```

no-go 경로에서는 같은 명령의 `--hdf5_files`를 `openarm_raw.with_cameras.hdf5`로 바꾼다. 두 경로 모두 `check_openarm_camera_dataset.py --expect-cameras yes`를 먼저 통과한 파일만 입력할 수 있다.

실제 XR 실행에는 기존 CloudXR 환경 변수도 함께 설정한다.

## 15. 구현 대상 파일

| 파일 | 변경 내용 |
|---|---|
| `source/leisaac/leisaac/tasks/lift_cube/openarm_collected_physics01_env_cfg.py` | Physics01 전용 세 카메라, observation 및 명시적 action feature 이름 |
| `source/leisaac/leisaac/devices/openarm_vr_v2/openxr_source.py` | 오른쪽 A 성공 edge 및 안내 |
| `scripts/environments/teleoperation/teleop_se3_agent.py` | 성공 callback, `--record_cameras`, probe, HDF5 env_name과 sidecar metadata |
| `source/leisaac/leisaac/utils/robot_utils.py` | 명시적 action feature 이름 사용 및 dimension 검증 |
| `scripts/tools/openarm_camera_preflight.py` | dynamic prim과 scene-state coverage 검사 |
| `scripts/tools/check_openarm_camera_dataset.py` | camera 유무별 HDF5·metadata·영상 정렬 검사 |
| `scripts/convert/isaaclab2lerobotv3.py` | 명시적 정수 stride 변환 및 첫 5 frame 하드코딩 제거 |
| `scripts/convert/render_hdf5_cameras.py` | XR 카메라 no-go 시 전체 state 기반 offline RGB 생성 |
| `source/leisaac/test/test_openarm_vr_v2_openxr_source.py` | 오른쪽 A edge callback 회귀 테스트 |

offline render 파일은 XR 라이브 3카메라 관문이 실패한 경우에만 구현한다.

## 16. 완료 기준

다음 조건이 모두 증명되어야 완료다.

- 실제 세 카메라의 부모 link와 calibration 출처가 기록되어 있다.
- 수집 전 state coverage preflight가 모든 task-relevant movable object를 확인한다.
- Physics01에는 정확히 `chest`, `left_wrist`, `right_wrist` 세 camera sensor가 있다.
- raw 16D, processed 18D 및 state 18D 순서가 이름과 일치한다.
- `obs_t → action_t → states_(t+1)` 관계가 검사기와 변환기에 반영되어 있다.
- Quest 오른쪽 A 성공, 왼쪽 Y 실패가 각각 HDF5 success attr로 저장된다.
- 라이브 또는 검증된 offline render 경로 중 하나로 세 RGB 영상이 생성된다.
- 라이브에서는 두 camera probe가 정량 관문을 통과하고, offline에서는 모든 index에서 각 `Camera.frame`이 정확히 1 증가한다.
- metadata sidecar가 task, 주기, schema 및 camera calibration을 보존한다.
- LeRobot 변환 전에 HDF5, sidecar 및 offline ready marker의 task·FPS·schema·hash 계약이 일치한다.
- HDF5를 30 Hz LeRobot v3로 변환했을 때 영상·state·action frame 수가 일치한다.
- 고정 조건 rollout 20회에서 18회 이상 성공해 작은 데이터셋의 학습·rollout 한 사이클이 검증된다.
