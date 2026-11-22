"""Configuration and parameter accounting for the Looped MoE Transformer."""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ModelConfig:
    """Architecture hyperparameters sized for Apple M4 (16 GB Unified Memory)."""

    # Vocabulary & sequence
    vocab_size: int = 50304  # GPT-2 50257 padded to multiple of 64 for Metal alignment
    max_seq_len: int = 16384

    # Core dimensions
    d_model: int = 512
    n_heads: int = 8
    n_kv_heads: int = 2  # 4:1 Grouped-Query Attention (GQA)
    head_dim: int = 64
    rope_theta: float = 500000.0  # Llama-3 / Qwen-2.5 high-base RoPE for long context
    rms_norm_eps: float = 1e-6

    # Long-context extensions (YaRN RoPE + Attention Sinks + Hybrid SWA + Memory Chunking)
    yarn_orig_ctx: int = 4096      # Base context length before YaRN interpolation ramp
    yarn_scale: float = 4.0        # 4x YaRN extension (4k -> 16k native, extrapolates to 32k-64k)
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    use_attention_sinks: bool = True  # Learned per-head attention sinks in fused SDPA
    sliding_window: Optional[int] = 2048  # Hybrid SWA on alternating core layers; None = all global
    sdpa_chunk_size: int = 512     # Exact Q-chunked checkpointed SDPA threshold/chunk size
    ce_chunk_size: int = 1024      # Exact chunked checkpointed cross-entropy chunk size

    # Looped / Recurrent-Depth topology
    n_prelude_layers: int = 1  # Non-looped dense prelude block(s)
    n_core_layers: int = 2     # Shared MoE block(s) executed per loop iteration
    n_coda_layers: int = 1     # Non-looped dense coda block(s)
    num_loops: int = 4         # Default recurrent iterations R (effective depth = 1 + 2*4 + 1 = 10)
    max_loops: int = 16        # Maximum supported loop iterations for step embeddings

    # Dense FFN (Prelude & Coda)
    dense_ffn_dim: int = 1408  # ~2.75x d_model, multiple of 128

    # DeepSeek-style Fine-Grained MoE + Shared Expert (Recurrent Core)
    n_routed_experts: int = 8
    top_k: int = 2
    expert_dim: int = 512       # Fine-grained routed expert SwiGLU hidden dim
    n_shared_experts: int = 1
    shared_expert_dim: int = 512
    router_bias_update_rate: float = 1e-3  # Auxiliary-loss-free load balancing gamma

    # Loop-step conditioning & stability
    step_cond_dim: int = 64     # Bottleneck dim for per-step AdaLN modulation
    tie_embeddings: bool = True
    use_qk_norm: bool = True
    use_input_injection: bool = True
    use_fused_kernels: bool = True  # Custom Metal fused SwiGLU, AdaLN+RMSNorm, Router Top-K
    use_checkpointing: bool = False # Per-loop activation checkpointing (enable for long ctx / large R)
    bptt_steps: Optional[int] = None  # If set (< num_loops), truncate BPTT to last K loops

    @property
    def effective_depth(self) -> int:
        return self.n_prelude_layers + self.n_core_layers * self.num_loops + self.n_coda_layers

    def param_breakdown(self, num_loops: Optional[int] = None) -> Dict[str, int]:
        """Exact analytical parameter counts: unique, effective-depth, and active-per-token."""
        R = num_loops if num_loops is not None else self.num_loops
        D = self.d_model
        H = self.n_heads * self.head_dim
        KV = self.n_kv_heads * self.head_dim

        # Embeddings (tied input/output)
        emb_params = self.vocab_size * D
        if not self.tie_embeddings:
            emb_params *= 2
        final_norm_params = D

        # Attention block (GQA + QK-Norm + optional Attention Sinks)
        attn_params = D * H + 2 * (D * KV) + H * D
        if self.use_qk_norm:
            attn_params += 2 * self.head_dim
        if self.use_attention_sinks:
            attn_params += self.n_heads
        block_norms = 2 * D

        # Dense SwiGLU FFN (fused gate+up + down)
        dense_ffn_params = D * (2 * self.dense_ffn_dim) + self.dense_ffn_dim * D
        dense_block_params = attn_params + dense_ffn_params + block_norms

        # MoE FFN (routed + shared + iteration-aware router)
        single_expert_params = D * (2 * self.expert_dim) + self.expert_dim * D
        routed_params_total = self.n_routed_experts * single_expert_params
        routed_params_active = self.top_k * single_expert_params

        shared_params = self.n_shared_experts * (
            D * (2 * self.shared_expert_dim) + self.shared_expert_dim * D
        )
        # Router: token gate (D -> E) + step conditioning gate (D -> E)
        router_params = 2 * (D * self.n_routed_experts)

        # Core block extras: AdaLN step modulation (D -> step_cond_dim -> 4*D) + 2 LayerScales (2*D)
        adaln_params = D * self.step_cond_dim + self.step_cond_dim * (4 * D) + 4 * D
        layerscale_params = 2 * D

        moe_block_unique = (
            attn_params + routed_params_total + shared_params + router_params
            + block_norms + adaln_params + layerscale_params
        )
        moe_block_active = (
            attn_params + routed_params_active + shared_params + router_params
            + block_norms + adaln_params + layerscale_params
        )

        # Recurrent core shared machinery: input injection (2*D -> D + norm) + step embeddings
        inject_params = (2 * D * D + D) if self.use_input_injection else 0
        step_emb_params = self.max_loops * D

        prelude_total = self.n_prelude_layers * dense_block_params
        coda_total = self.n_coda_layers * dense_block_params
        core_unique = self.n_core_layers * moe_block_unique + inject_params + step_emb_params
        core_active_per_iter = self.n_core_layers * moe_block_active + inject_params

        unique_total = emb_params + prelude_total + core_unique + coda_total + final_norm_params
        # Effective parameters if the R recurrent loops were unrolled into distinct weights
        effective_depth_total = (
            emb_params + prelude_total + R * (self.n_core_layers * moe_block_unique + inject_params)
            + step_emb_params + coda_total + final_norm_params
        )
        # Active parameters per token (non-embedding compute path across all R iterations)
        active_non_emb_per_token = prelude_total + R * core_active_per_iter + coda_total + final_norm_params
        # Unique active parameters touched per token (counting shared core weights once + lm_head)
        active_unique_with_emb = emb_params + prelude_total + core_active_per_iter + coda_total + final_norm_params

        return {
            "unique_total": unique_total,
            "embeddings": emb_params,
            "non_embedding_unique": unique_total - emb_params,
            "effective_depth_total": effective_depth_total,
            "active_non_emb_compute_params": active_non_emb_per_token,
            "active_unique_total": active_unique_with_emb,
            "effective_layers": self.n_prelude_layers + R * self.n_core_layers + self.n_coda_layers,
        }


@dataclass
class TrainConfig:
    """Training, optimizer, and data configuration."""

    # Data
    data_dir: str = "data"
    dataset_name: str = "roneneldan/TinyStories"
    num_train_stories: int = 25000
    num_val_stories: int = 1000

    # Batching & sequence
    batch_size: int = 8
    seq_len: int = 512
    grad_accum_steps: int = 4  # Effective batch = 8 * 4 * 512 = 16,384 tokens/step

    # Optimization (Muon for 2D hidden weights, AdamW for embeddings/1D/routers)
    max_steps: int = 300
    warmup_steps: int = 30
    muon_lr: float = 0.015
    adamw_lr: float = 1.5e-3
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    dtype: str = "bfloat16"

    # Recurrent-depth training options
    random_loops: bool = False  # Optionally sample R in [min_train_loops, max_train_loops]
    min_train_loops: int = 2
    max_train_loops: int = 6

    # Logging & checkpointing
    log_interval: int = 10
    eval_interval: int = 50
    eval_batches: int = 10
    save_dir: str = "checkpoints"
    seed: int = 42
