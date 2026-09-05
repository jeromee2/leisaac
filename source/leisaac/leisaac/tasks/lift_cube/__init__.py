import gymnasium as gym

gym.register(
    id="LeIsaac-SO101-LiftCube-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lift_cube_env_cfg:LiftCubeEnvCfg",
    },
)

gym.register(
    id="LeIsaac-OpenArm-LiftCube-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.openarm_lift_cube_env_cfg:OpenArmLiftCubeEnvCfg",
    },
)

gym.register(
    id="LeIsaac-OpenArm-Bimanual-LiftCube-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.openarm_bimanual_lift_cube_env_cfg:OpenArmBimanualLiftCubeEnvCfg",
    },
)

gym.register(
    id="LeIsaac-OpenArm-Bimanual-LiftCube-QuestV2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.openarm_bimanual_lift_cube_env_cfg:OpenArmBimanualLiftCubeEnvCfg",
    },
)

gym.register(
    id="LeIsaac-SO101-LiftCube-DigitalTwin-v0",
    entry_point="leisaac.enhance.envs:ManagerBasedRLDigitalTwinEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lift_cube_env_cfg:LiftCubeDigitalTwinEnvCfg",
    },
)

gym.register(
    id="LeIsaac-SO101-LiftCube-Mimic-v0",
    entry_point=f"leisaac.enhance.envs:ManagerBasedRLLeIsaacMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lift_cube_mimic_env_cfg:LiftCubeMimicEnvCfg",
    },
)

gym.register(
    id="LeIsaac-SO101-LiftCube-Direct-v0",
    entry_point=f"{__name__}.direct.lift_cube_env:LiftCubeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct.lift_cube_env:LiftCubeEnvCfg",
    },
)

gym.register(
    id="LeIsaac-OpenArm-Bimanual-Physics01-QuestV2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.openarm_collected_physics01_env_cfg:OpenArmBimanualPhysics01EnvCfg",
    },
)
