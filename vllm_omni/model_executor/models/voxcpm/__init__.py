from .configuration_voxcpm import VoxCPMConfig

__all__ = [
    "VoxCPMConfig",
    "VoxCPMForConditionalGeneration",
    "VoxCPMStage0PagedForConditionalGeneration",
]


def __getattr__(name: str):
    if name == "VoxCPMForConditionalGeneration":
        from .voxcpm import VoxCPMForConditionalGeneration

        return VoxCPMForConditionalGeneration
    if name == "VoxCPMStage0PagedForConditionalGeneration":
        from .voxcpm_stage0_talker import VoxCPMStage0PagedForConditionalGeneration

        return VoxCPMStage0PagedForConditionalGeneration
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
