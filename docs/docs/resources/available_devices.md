# Available Devices

::::tip
We support several teleoperation devices in simulation, but for the best control experience we recommend using the SO101 Leader.
::::

## Single-Arm SO101 Follower

You can control the Single-Arm SO101 Follower in simulation with SO101 Leader or keyboard.

### SO101-Leader

Run with `--teleop_device=so101leader` to enable the SO101 Leader; once the port is configured you can start operating immediately.

When you move the physical SO101 Leader, the simulated manipulator mirrors the same motion.


### Keyboard

Run with `--teleop_device=keyboard` to enable keyboard control.

| target frame                                               | translation control                          | rotation control                         |
| :--------------------------------------------------------: | :------------------------------------------: | :--------------------------------------: |
| ![target frame](/img/devices/teleop_info/target_frame.jpg) | ![trans](/img/devices/teleop_info/trans.jpg) | ![rot](/img/devices/teleop_info/rot.jpg) |

Keyboard teleoperation relies on target-frame-based control. Within the target-frame (gripper link) coordinate system, key signals always command the movement of that frame, so the operator only needs to focus on the desired pose of the link rather than sending joint-level commands. Gripper joints are still handled separately.

We split target-frame control into translation and rotation. Translation covers forward/backward, up/down, and left/right motions of the target frame. Because of the manipulator structure, the left/right action is effectively a rotation around the base rather than a pure translation. Rotation control includes pitch up/down and rotation around the wrist link.

The keyboard mapping is as follows:

| input key | description                                                                   |
| :-------: | :---------------------------------------------------------------------------- |
| `W` / `S` | Forward/backward, aligned with the red arrows in the translation diagram      |
| `A` / `D` | Left/right, aligned with the green arrows in the translation diagram          |
| `Q` / `E` | Up/down, aligned with the blue arrows in the translation diagram              |
| `J` / `L` | Rotate (yaw) left/right, aligned with the blue arrows in the rotation diagram |
| `K` / `I` | Rotate (pitch) up/down, aligned with the green arrows in the rotation diagram |
| `U` / `O` | Gripper open/close                                                            |

### Gamepad

Run with `--teleop_device=gamepad` to enable gamepad control. (Now we support xbox series controller.)

Similar to the keyboard controls, the gamepad also commands the target frame (gripper link) and is divided into translation control and rotation control. The mappings are described below.

| control key                          | description                                                                   |
| :----------------------------------: | :---------------------------------------------------------------------------- |
| move `L` forward / move `L` backward | Forward/backward, aligned with the red arrows in the translation diagram      |
| move `L` left / move `L` right       | Left/right, aligned with the green arrows in the translation diagram          |
| move `R` forward / move `R` backward | Up/down, aligned with the blue arrows in the translation diagram              |
| move `R` left / move `R` right       | Rotate (yaw) left/right, aligned with the blue arrows in the rotation diagram |
| press `LB` / press `LT`              | Rotate (pitch) up/down, aligned with the green arrows in the rotation diagram |
| press `RT` / press `RB`              | Gripper open/close                                                            |

### Quest 3 Controller

Run with `--teleop_device=quest3-controller` to use Meta Quest 3 Touch Plus controllers through OpenXR. The script
enables XR automatically and starts controller teleoperation enabled; the right squeeze remains a motion clutch. Use
`--xr_start_paused` to wait for Left X or an external OpenXR `START` command. The right controller drives relative
end-effector motion, while the left controller handles the session controls.

For `LeIsaac-OpenArm-Bimanual-LiftCube-v0`, both controllers drive their matching arms. Each squeeze is an independent
clutch and each trigger controls the matching gripper; Left X and Y keep their pause/reset functions.

| controller input | description |
| :--------------: | :---------- |
| Left `X` | Pause or resume teleoperation |
| Left `Y` | Reset the environment |
| Right squeeze | Hold as a clutch; the arm moves only while held |
| Right grip pose | Move the end effector relatively |
| Right trigger | Close at 65% pull; reopen below 35% pull |

### Quest 3 Hand Tracking

Run with `--teleop_device=handtracking` for bare-hand control. Quest/OpenXR uses one interaction profile at a time:
when the log reports `meta/touch_plus_controller`, only controller poses are available, not hand joints. Put both Touch
Plus controllers down or turn them off, expose your right hand to the headset, and wait for the script to report that it
captured a zero-motion baseline. The script pauses control until it can see `wrist`, `thumb_tip`, and `index_tip`.
The bimanual OpenArm environment requires those joints on both hands and emits a 14D left-then-right action.

## Bi-Arm SO101 Follower

Because keyboard control becomes complicated for dual arms, the Bi-Arm SO101 Follower in simulation currently supports only the Bi-SO101 Leader.

### Bi-SO101-Leader

Run with `--teleop_device=bi-so101leader` to enable the Bi-SO101 Leader. After configuring both `left_arm_port` and `right_arm_port`, you can operate immediately, and the two simulated arms will reproduce the real bi-arm behavior.

## LeKiwi

LeKiwi的teleoperation可以分为两个部分，分别是单臂SO101 Follower的控制，以及底座的控制。

### lekiwi-leader

Run with `--teleop_device=lekiwi-leader` to enable it. In this configuration, the SO101 Follower is controlled by the SO101-Leader arm, while the mobile base is driven via the keyboard.

The default keyboard mappings are:

| input key                    | description                             |
| :--------------------------: | :-------------------------------------- |
| :arrow_up: / :arrow_down:    | Move forward / backward                 |
| :arrow_left: / :arrow_right: | Move left / right                       |
| `Z` / `X`                    | Rotate left / right                     |
| `1` / `2` / `3`              | Set speed level to slow / medium / fast |

### lekiwi-keyboard

Run with `--teleop_device=lekiwi-keyboard` to enable it. In this configuration, both the SO101 Follower and the mobile base are controlled via the keyboard.

The arm uses the same key mappings as the [`Keyboard`](#keyboard) configuration described above, and the base shares the same keyboard controls as in the [`lekiwi-leader`](#lekiwi-leader) configuration.


### lekiwi-gamepad

Run with `--teleop_device=lekiwi-gamepad` to enable it. In this configuration, both the SO101 Follower and the mobile base are controlled via the gamepad.

The arm uses the same mappings as the [`Gamepad`](#gamepad) configuration described above, while the mobile base uses the following layout:

| control key                              | description                                            |
| :--------------------------------------: | :----------------------------------------------------- |
| D-pad :arrow_up: / D-pad :arrow_down:    | Move forward / backward                                |
| D-pad :arrow_left: / D-pad :arrow_right: | Move left / right                                      |
| `X` / `B`                                | Rotate left / right                                    |
| `Y` / `A`                                | Increase / decrease speed level (slow / medium / fast) |
