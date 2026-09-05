"""OpenArm Quest controller teleoperation V2."""

from importlib import import_module
from typing import TYPE_CHECKING

from .core import (
    ClutchedPoseMapper,
    HandMappingResult,
    HandTeleopState,
    OneEuroPoseSmoother,
    OpenXRInputFrame,
    OperatorFrame,
    TrackedPoseSample,
)
if TYPE_CHECKING:
    from .device import Quest3OpenArmTeleopV2
    from .isaac_qp_controller import OpenArmIsaacQPControllerV2

__all__ = [
    "ClutchedPoseMapper",
    "HandMappingResult",
    "HandTeleopState",
    "OneEuroPoseSmoother",
    "OpenArmIsaacQPControllerV2",
    "OpenXRInputFrame",
    "OperatorFrame",
    "Quest3OpenArmTeleopV2",
    "TrackedPoseSample",
]

_LAZY_IMPORTS = {
    "OpenArmIsaacQPControllerV2": (".isaac_qp_controller", "OpenArmIsaacQPControllerV2"),
    "Quest3OpenArmTeleopV2": (".device", "Quest3OpenArmTeleopV2"),
}


def __getattr__(name):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
