# OpenArm Quest 3 V2 설치 핸드오프

이 문서는 다른 컴퓨터의 AI 에이전트가 이 저장소를 설치하고 검증하기 위한 인수인계 문서다.
명령은 저장소 루트에서 실행한다. 사용자의 계정명, 홈 디렉터리, UID를 추측하거나 하드코딩하지 않는다.

## 목표와 기준

- 저장소: `https://github.com/jeromee2/leisaac.git`
- 브랜치: `openarm-quest3-teleop`
- 기준 소스(테스트 기록): `465628a`
- Physics01 번들: Git 일반 파일 192개, 약 237 MB, 최대 약 66.89 MiB
- Git LFS는 필요하지 않다. 큰 파일 경고가 나도 LFS로 바꾸지 않는다.
- 검증 기준은 Ubuntu 24.04, RTX 5070 Ti (driver 580.173.02), Python 3.11, Isaac Sim 5.1.0,
  Isaac Lab 2.3.0, CUDA 12.8에서 실행됐다.
- 다른 GPU/드라이버 조합은 호스트에서 호환성을 확인하며, 확인되지 않은 조합도 안전한 범위에서 설치를 계속할 수 있다.
- 정확히 고정된 IsaacLab 서브모듈은 그대로 사용한다. `--remote`로 서브모듈을 갱신하지 않는다.

저장소의 [OpenArm Quest 3 setup](OPENARM_QUEST3_SETUP.md)을 canonical 설치 문서로 사용한다.
여기에 전체 의존성 설치 절차를 복사하거나 새로운 다운로드 URL을 만들어내지 않는다.
먼저 setup 문서의 `1. Install LeIsaac` 전체 순서를 완료한다. 아래 명령은 그중 V2 추가 의존성 설치 단계다.

## AI가 먼저 확인할 것

AI는 다음 명령을 직접 실행하고 결과를 사용자에게 요약한다.

```bash
uname -a
lsb_release -a
nvidia-smi
df -h .
python3 --version
git --version
```

그 결과로 Ubuntu 버전, NVIDIA 드라이버/GPU, 디스크 여유, Python 버전을 확인한다.
차이를 기록하고 호환성을 확인한다. 실제 누락된 파일·접근 권한·사람의 조작이 필요할 때만 사용자에게 요청하며, 그 밖의 설치·검증은 계속한다.

## 저장소와 격리 환경

```bash
git clone --branch openarm-quest3-teleop --recursive https://github.com/jeromee2/leisaac.git
cd leisaac
git rev-parse --short HEAD
git submodule status
```

브랜치와 현재 HEAD를 기록한다. `465628a`는 테스트 기록의 역사적 기준이며, 브랜치의 최신 문서 커밋으로 갱신된 HEAD를 실패로 처리하지 않는다.
기존 전역 Python에 설치하지 말고 conda/venv 등 격리 환경을 만든다. 예시는 사용자가 선택한 환경을
`$ENV_NAME`으로 가정하지 않고, 활성화한 환경의 `python`과 `pip`만 사용한다.

```bash
python -m pip install -e 'source/leisaac[openarm-vr]'
```

CUDA, PyTorch, Isaac Sim, Isaac Lab 설치는 현재 호스트의 공식 설치 방식과
`OPENARM_QUEST3_SETUP.md`를 따른다. 이미 설치된 IsaacLab 서브모듈을 임의로 교체하지 않는다.

## 별도 사용자 제공 항목

다음은 저장소에 포함되지 않으며 사용자가 NVIDIA에서 합법적으로 확보해야 한다.

- NVIDIA CloudXR SDK 6.2.1 Linux: `libcloudxr.so`, `openxr_cloudxr.json`
- 라이선스된 NVIDIA Isaac Teleop Web Client: `index.html`, `bundle.js`, 라이선스 파일,
  `favicon.ico`

AI는 다운로드 URL, 라이선스 파일, 바이너리, 체크섬을 발명하지 않는다. CloudXR SDK를 다른
컴퓨터에서 복사하거나 GPU 라이브러리를 섞지 않는다. Web Client는 저장소 밖의 별도 디렉터리에 둔다.

## 환경 변수

각 터미널은 환경을 상속하지 않는다. 서비스, 웹 서버, 시뮬레이터 터미널 모두 같은 격리 환경을
활성화하고 저장소 루트로 이동한 뒤 필요한 변수를 다시 export한다.

```bash
export CLOUDXR_ROOT="${CLOUDXR_ROOT:-$HOME/cloudxr6}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export CXR_LIB_PATH="$CLOUDXR_ROOT/libcloudxr.so"
export LD_LIBRARY_PATH="$CLOUDXR_ROOT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export XR_RUNTIME_JSON="$CLOUDXR_ROOT/openxr_cloudxr.json"
```

`CLOUDXR_ROOT`만 실제 사용자가 설치한 SDK 위치로 설정한다. `/home/lab`, `/run/user/1000` 같은
개발 머신 경로를 복사하지 않는다.

## 세 터미널 실행 (검증 통과 후)

터미널 1에서 같은 환경을 활성화하고 CloudXR 서비스를 실행한다.

```bash
python scripts/cloudxr/cloudxr_service.py
```

서비스가 출력하는 포트(정상 기준 49100)를 기록한다. 임의의 포트나 방화벽 변경을 추가하지 않는다.

터미널 2에서 라이선스된 Web Client 디렉터리를 실제 경로로 지정한다.

```bash
python -m http.server 8080 --bind 0.0.0.0 --directory "$HOME/isaacteleop-client"
```

Quest 브라우저에서 `http://WORKSTATION_LAN_IP:8080`을 열고 서비스가 출력한 `WORKSTATION_LAN_IP:49100`에
연결한다. LAN 주소를 사용하며, 주소/포트/방화벽을 추측해 추가하지 않는다.

터미널 3에서 다음을 다시 export하고 Physics01 V2를 실행한다.

```bash
export CLOUDXR_ROOT="${CLOUDXR_ROOT:-$HOME/cloudxr6}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export XR_RUNTIME_JSON="$CLOUDXR_ROOT/openxr_cloudxr.json"
export LD_LIBRARY_PATH="$CLOUDXR_ROOT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
NV_DEVICE_PROFILE=auto-webrtc \
python -u scripts/environments/teleoperation/teleop_se3_agent.py \
  --task LeIsaac-OpenArm-Bimanual-Physics01-QuestV2-v0 \
  --teleop_device quest3-controller-v2 --num_envs 1 --device cuda:0 --headless
```

`--headless`는 검증/원격 실행용이다. 실제 Quest 사용 시에도 CloudXR 변수와 네트워크를 유지한다.
Physics01이 기본 V2 장면이며, 공식 LiftCube V2 회귀 장면은 필요할 때 다음 task를 사용한다.

```text
LeIsaac-OpenArm-Bimanual-LiftCube-QuestV2-v0
```

## 정확한 검증 명령 5개

다음 다섯 명령만 설치 검증 기준으로 기록한다. 모두 저장소 루트에서 실행한다.
CloudXR/VR 실행 전에 같은 GPU에서 한 번에 하나씩 순서대로 실행한다. 실패하면 원인을 해결하고 재실행한다.

```bash
PYTHONPATH=source/leisaac python -m pytest -q source/leisaac/test/test_openarm_vr_v2_core.py
python -u scripts/tools/openarm_v2_teleop_check.py --task LeIsaac-OpenArm-Bimanual-LiftCube-QuestV2-v0 --headless --device cuda:0
python -u scripts/tools/openarm_v2_teleop_check.py --task LeIsaac-OpenArm-Bimanual-Physics01-QuestV2-v0 --headless --device cuda:0
python -u scripts/tools/openarm_v2_fk_parity_check.py --headless --device cuda:0
python -u scripts/tools/usd_teleop_preflight.py assets/scenes/Collected_physics01/physics01.usd --device cuda:0
```

기준 머신에서는 mapper 테스트 8개, 공식/Physics01 V2 통합, FK parity 12개, USD preflight가 통과했다.
이 결과는 소스 머신의 기록이지 새 컴퓨터의 성공을 보장하지 않는다. 각 명령의 전체 출력과 종료 코드를
새 환경에서 다시 기록한다. USD preflight에서 missing USD dependency가 나오면 실패로 보고한다.

FK harness는 `.08 s` joint-target lookahead를 실제 관절의 한 60 Hz control step으로 적분한다.
설치 과정에서 lookahead나 속도 제한을 튜닝하지 않는다. 기본값은 `.08 s`, Cartesian cap은 1 m/s와
4.5 rad/s이며 기존 옵션 `--openarm_v2_joint_target_lookahead_s 0.05`만 이미 제공된다.

Physics01 relocation preflight 기준으로 schema와 132 dependencies는 통과하지만 Map #1461,
`OmniGlass.mdl`, `OmniPBR.mdl`, `OmniSurfacePresets.mdl`, `gltf/pbr.mdl`의 다섯 cosmetic material
warning은 남을 수 있다. geometry/schema가 통과하고 missing USD가 없으면 이는 가능한 fallback appearance
경고로 기록한다.

## 실제 Quest 수동 확인

- Quest 수용 테스트는 자동으로 완료되지 않는다. 양쪽 squeeze로 각 팔이 움직이고 release 후 hold되는지, trigger가 각 gripper를 조작하는지, Left X pause/resume·Left Y reset, forward/left/up·wrist roll/pitch/yaw, tracking loss 후 release/reclutch 점프를 사용자가 확인한다. 현재 소스 머신에서 Quest end-to-end는 검증되지 않았다.
- V2는 self-collision이 비활성화되어 cross-arm/body collision이 보장되지 않는다. 낮은 속도와 손 간격으로 시작한다.

## 결과 보고 양식

```text
환경: OS / GPU / driver / Python / Isaac Sim / Isaac Lab / CUDA
소스: URL / branch / HEAD / submodule status
검증: 다섯 명령별 PASS/FAIL, 종료 코드, 로그 경로
경고: material warnings 및 missing dependencies
미완료: CloudXR 연결, Quest 수동 acceptance, collision safety
```

## 복사해서 다른 AI에게 전달할 프롬프트

```text
이 저장소를 설치하라: https://github.com/jeromee2/leisaac.git, branch openarm-quest3-teleop.
먼저 AI_INSTALL_HANDOFF.md와 OPENARM_QUEST3_SETUP.md를 읽고 OS/GPU/driver/disk/Python을 검사하라.
격리 환경을 사용하고 IsaacLab pinned submodule을 --remote로 바꾸지 말라.
사용자 제공 CloudXR 6.2.1과 라이선스 Web Client가 없으면 발명하거나 다운로드 URL을 만들지 말라.
정확히 다섯 검증 명령을 실행하고 각 PASS/FAIL, 종료 코드, 로그, 경고, 수동 미완료 항목을 보고하라.
기존 성공 기록을 새 PC의 결과로 간주하지 말고 이 PC에서 직접 실행한 결과만 보고해줘. 헤드셋 착용/조작이 필요하면 사용자에게 요청해줘.
소스 push나 commit은 하지 말라.
```
