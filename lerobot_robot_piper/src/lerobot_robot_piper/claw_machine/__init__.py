"""Modular claw-machine workflow for Piper + RH56F2."""

__all__ = ["ClawMachineController", "ClawMachineTaskConfig"]


def __getattr__(name):
    if name in __all__:
        from .lerobot_claw import ClawMachineController, ClawMachineTaskConfig

        return {
            "ClawMachineController": ClawMachineController,
            "ClawMachineTaskConfig": ClawMachineTaskConfig,
        }[name]
    raise AttributeError(name)
