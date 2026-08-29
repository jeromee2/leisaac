# OpenArm v1.0 simulation asset

LeIsaac uses the official Isaac Sim 5.1 / Isaac Lab 2.3 unimanual OpenArm
asset below `ISAAC_NUCLEUS_DIR`:

```
${ISAAC_NUCLEUS_DIR}/Robots/OpenArm/openarm_unimanual/openarm_unimanual.usd
```

The runtime config falls back to this Nucleus asset unless a converted local
USD exists at `assets/robots/openarm_v1.0/openarm_unimanual.usd` or
`LEISAAC_OPENARM_USD_PATH` is set. This keeps the repository lightweight while
allowing an air-gapped deployment to pin a locally converted USD.

The bimanual environment uses the matching official asset:

```
${ISAAC_NUCLEUS_DIR}/Robots/OpenArm/openarm_bimanual/openarm_bimanual.usd
```

A local bimanual conversion can be selected with
`LEISAAC_OPENARM_BIMANUAL_USD_PATH` or placed at
`assets/robots/openarm_v1.0/openarm_bimanual.usd`.

The corresponding mechanical description is maintained by Enactic in
[`openarm_description`](https://github.com/enactic/openarm_description),
`assets/robot/openarm_v1.0`. The hardware specifications are documented at
[`docs.openarm.dev/1.0/hardware`](https://docs.openarm.dev/1.0/hardware/).
The unimanual, no-prefix model used here exposes:

- arm joints: `openarm_joint1` through `openarm_joint7`
- gripper joints: `openarm_finger_joint1` and `openarm_finger_joint2`
- base link: `openarm_link0`
- task-space body: `openarm_hand`
- TCP frame: `openarm_ee_tcp`

The bimanual USD uses `openarm_left_` and `openarm_right_` prefixes. Its torso
base is `openarm_body_link`; its IK bodies are `openarm_left_hand` and
`openarm_right_hand`; and its TCP bodies are `openarm_left_ee_tcp` and
`openarm_right_ee_tcp`. These runtime names were verified against Isaac Sim
5.1 because they differ slightly from the source Xacro names.

The OpenArm v1.0 parallel gripper range is `0.0`–`0.044` m. The arm actuator
limits used by the Isaac Lab profile are 40 Nm for joints 1–2, 27 Nm for
joints 3–4, and 7 Nm for joints 5–7, with velocity limits 2.175, 2.175, and
2.61 rad/s respectively. Joint position limits remain authored in the USD.
