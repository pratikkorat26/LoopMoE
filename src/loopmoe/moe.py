"""DeepSeek-style Sparse MoE with Iteration-Aware Routing and Auxiliary-Loss-Free Balancing.

Uses Apple MLX's hardware-accelerated `mx.gather_mm(..., sorted_indices=True)` with
fused Gate+Up SwiGLU projections for high Metal throughput on Apple Silicon.
"""

import math
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn

from loopmoe.config import ModelConfig
from loopmoe.kernels import fused_router_topk_sigmoid, fused_swiglu


class SharedExpert(nn.Module):
    """Always-active SwiGLU shared expert with fused Gate+Up projection."""

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        resid_scale: float = 1.0,
        use_fused_kernels: bool = True,
    ):
        super().__init__()
        self.use_fused_kernels = use_fused_kernels
        self.w_gate_up = nn.Linear(d_model, 2 * hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)
        # Scale down output projection for deep/recurrent residual stability
        std = 0.02 / math.sqrt(max(1.0, resid_scale))
        self.w_down.weight = mx.random.normal(self.w_down.weight.shape) * std

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        gate_up = x @ self.w_gate_up.weight.astype(dtype).T
        if self.use_fused_kernels:
            act = fused_swiglu(gate_up)
        else:
            gate, up = mx.split(gate_up, 2, axis=-1)
            act = nn.silu(gate) * up
        return act @ self.w_down.weight.astype(dtype).T


class IterationAwareRouter(nn.Module):
    """Top-K sigmoid router with loop-step conditioning and dynamic load-balancing bias.

    Routing decisions incorporate both the token hidden state and the current
    recurrent loop-step embedding so experts can specialize in early vs. late
    reasoning passes. Load balancing follows DeepSeek-V3's auxiliary-loss-free
    strategy: a non-gradient `expert_bias` shifts top-k selection without
    perturbing the unbiased combining weights or adding an auxiliary loss term.
    """

    def __init__(self, d_model: int, n_experts: int, top_k: int, use_fused_kernels: bool = True):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.use_fused_kernels = use_fused_kernels

        # Token & iteration-step projections (small std for uniform initial routing)
        scale = 0.02
        self.gate_weight = mx.random.normal((n_experts, d_model)) * scale
        self.step_weight = mx.random.normal((n_experts, d_model)) * scale

        # Non-trainable dynamic load-balancing bias (updated outside gradient descent)
        self.expert_bias = mx.zeros((n_experts,), dtype=mx.float32)
        self.freeze(keys=["expert_bias"])

    def __call__(
        self, x: mx.array, step_emb: mx.array
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Compute top-k expert indices, normalized unbiased weights, and expert load counts.

        Args:
            x: (B, T, D) hidden states.
            step_emb: (D,) embedding for the current recurrent loop iteration.

        Returns:
            top_k_idx: (B*T, K) int32 expert indices.
            top_k_weights: (B*T, K) float32 normalized combining weights.
            expert_counts: (E,) float32 token count per expert (stop-gradient).
        """
        x_flat = x.reshape(-1, x.shape[-1]).astype(mx.float32)
        step_f32 = step_emb.astype(mx.float32)

        # Iteration-aware logits in fp32 for routing stability
        logits = x_flat @ self.gate_weight.T.astype(mx.float32)
        logits = logits + (step_f32 @ self.step_weight.T.astype(mx.float32))

        if self.use_fused_kernels:
            top_k_idx, top_k_weights = fused_router_topk_sigmoid(
                logits, mx.stop_gradient(self.expert_bias), self.top_k
            )
        else:
            raw_scores = mx.sigmoid(logits)
            selection_scores = raw_scores + mx.stop_gradient(self.expert_bias)
            k = self.top_k
            top_k_idx = mx.stop_gradient(
                mx.argpartition(-selection_scores, kth=k - 1, axis=-1)[:, :k]
            )
            top_k_raw = mx.take_along_axis(raw_scores, top_k_idx, axis=-1)
            top_k_weights = top_k_raw / (mx.sum(top_k_raw, axis=-1, keepdims=True) + 1e-6)

        # Compute per-expert token counts for dynamic bias update & telemetry
        # One-hot sum over (N*K, E) is tiny since E=8
        flat_idx = mx.stop_gradient(top_k_idx.reshape(-1))
        eye = mx.eye(self.n_experts, dtype=mx.float32)
        expert_counts = mx.sum(eye[flat_idx], axis=0)

        return top_k_idx, top_k_weights, expert_counts

    def update_bias(self, expert_counts: mx.array, gamma: float) -> None:
        """DeepSeek-V3 dynamic bias update: nudge underused experts up, overused down."""
        total = mx.sum(expert_counts)
        if float(total.item()) <= 0.0:
            return
        avg_load = total / self.n_experts
        load_err = avg_load - expert_counts
        # Deadband of 1% prevents oscillation once balanced
        sign = mx.where(mx.abs(load_err) < 0.01 * avg_load, 0.0, mx.sign(load_err))
        new_bias = mx.clip(self.expert_bias + gamma * sign, -3.0, 3.0)
        self.expert_bias = new_bias


class MoELayer(nn.Module):
    """Fine-grained Sparse MoE + Always-Active Shared Expert using `mx.gather_mm`."""

    def __init__(self, cfg: ModelConfig, resid_scale: float = 1.0):
        super().__init__()
        self.d_model = cfg.d_model
        self.n_experts = cfg.n_routed_experts
        self.top_k = cfg.top_k
        self.expert_dim = cfg.expert_dim
        self.n_shared = cfg.n_shared_experts

        self.use_fused_kernels = cfg.use_fused_kernels
        self.router = IterationAwareRouter(
            cfg.d_model, cfg.n_routed_experts, cfg.top_k, use_fused_kernels=cfg.use_fused_kernels
        )

        # 3D batched routed expert weights for `mx.gather_mm`:
        #   w_gate_up: (E, D, 2 * H)
        #   w_down:    (E, H, D)
        in_std = 0.02
        out_std = 0.02 / math.sqrt(max(1.0, resid_scale))
        self.w_gate_up = (
            mx.random.normal((self.n_experts, self.d_model, 2 * self.expert_dim)) * in_std
        )
        self.w_down = (
            mx.random.normal((self.n_experts, self.expert_dim, self.d_model)) * out_std
        )

        # Always-active shared expert(s)
        if self.n_shared > 0:
            self.shared_expert = SharedExpert(
                cfg.d_model,
                cfg.shared_expert_dim * self.n_shared,
                resid_scale=resid_scale,
                use_fused_kernels=cfg.use_fused_kernels,
            )
        else:
            self.shared_expert = None

    def __call__(
        self, x: mx.array, step_emb: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """Forward pass through routed + shared experts.

        Args:
            x: (B, T, D) input activations.
            step_emb: (D,) loop iteration embedding.

        Returns:
            out: (B, T, D) combined MoE output.
            expert_counts: (E,) float32 routing counts for this pass.
        """
        B, T, D = x.shape
        N = B * T
        K = self.top_k
        orig_dtype = x.dtype

        top_k_idx, top_k_weights, expert_counts = self.router(x, step_emb)

        # Direct hardware-accelerated gather_mm:
        #   x_in: (N, 1, 1, D), w_gate_up: (E, D, 2*H), rhs_indices: (N, K)
        #   -> h: (N, K, 1, 2*H)  (zero sorting/unsorting overhead, exact at any N >= 1)
        x_flat = x.reshape(N, 1, 1, D)
        h = mx.gather_mm(
            x_flat,
            self.w_gate_up.astype(orig_dtype),
            rhs_indices=top_k_idx,
            sorted_indices=False,
        )  # (N, K, 1, 2*H)

        if self.use_fused_kernels:
            h = fused_swiglu(h)
        else:
            gate, up = mx.split(h, 2, axis=-1)
            h = nn.silu(gate) * up

        # Down projection via gather_mm -> (N, K, 1, D) -> (N, K, D)
        y_routed = mx.gather_mm(
            h,
            self.w_down.astype(orig_dtype),
            rhs_indices=top_k_idx,
            sorted_indices=False,
        ).squeeze(2)

        # Combine with unbiased router weights -> (B, T, D)
        y_routed = mx.sum(
            y_routed * top_k_weights.astype(orig_dtype)[..., None], axis=1
        ).reshape(B, T, D)

        if self.shared_expert is not None:
            y_routed = y_routed + self.shared_expert(x)

        return y_routed, expert_counts
