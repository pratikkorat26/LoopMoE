"""LoopMoE — Long-Context Looped Mixture-of-Experts Transformer in Apple MLX."""

from loopmoe.config import ModelConfig, TrainConfig
from loopmoe.kernels import fused_adaln_rmsnorm, fused_router_topk_sigmoid, fused_swiglu
from loopmoe.model import LoopedMoETransformer
from loopmoe.moe import IterationAwareRouter, MoELayer, SharedExpert

__all__ = [
    "ModelConfig",
    "TrainConfig",
    "LoopedMoETransformer",
    "MoELayer",
    "IterationAwareRouter",
    "SharedExpert",
    "fused_swiglu",
    "fused_adaln_rmsnorm",
    "fused_router_topk_sigmoid",
]
__version__ = "0.1.0"
