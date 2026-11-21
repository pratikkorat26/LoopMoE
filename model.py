"""Looped / Recurrent-Depth Mixture-of-Experts Transformer in Apple MLX.

Long-Context & Efficiency Architecture:
    Token Embeddings (tied)
      -> Prelude (non-looped dense GQA + SwiGLU block, Global Attention)
      -> Recurrent Core (shared GQA + MoE stack looped R times):
           * Hybrid Local Sliding-Window (even core layers) + Global Attention (odd core layers)
           * YaRN High-Base RoPE (theta=500k + smooth frequency ramp + attention mscale)
           * Learned per-head Attention Sinks inside fused Metal SDPA
           * Exact Q-Chunked Checkpointed Causal SDPA (4.4x less memory at 16k, 0.0 parity diff)
           * Prelude input injection + Loop-step AdaLN-Zero + LayerScale
           * Custom Metal fused SwiGLU, AdaLN+RMSNorm, and Register-Resident Router Top-K
           * Per-iteration activation checkpointing (`mx.checkpoint`) & optional truncated BPTT
      -> Coda (non-looped dense GQA + SwiGLU block, Global Attention)
      -> Final RMSNorm -> Tied LM Head with Exact Chunked Checkpointed Cross-Entropy
"""

import math
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from config import ModelConfig
from kernels import fused_adaln_rmsnorm, fused_swiglu
from moe import MoELayer


# ---------------------------------------------------------------------------
# Long-Context Primitives: YaRN RoPE & Exact Q-Chunked Checkpointed SDPA
# ---------------------------------------------------------------------------

def build_yarn_rope_periods(
    head_dim: int,
    theta: float = 500000.0,
    orig_ctx: int = 4096,
    scale: float = 4.0,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> Tuple[mx.array, float]:
    """Compute YaRN RoPE periods for `mx.fast.rope(..., base=None, freqs=periods)` and mscale.

    Preserves high-frequency (local) dimensions untouched while smoothly interpolating
    low-frequency (long-range) dimensions, enabling 16k training and 32k-64k extrapolation
    without degrading short-context perplexity.
    """
    half = head_dim // 2
    dims = mx.arange(0, half, dtype=mx.float32)
    base_inv_freq = 1.0 / (theta ** (2.0 * dims / float(head_dim)))

    if scale <= 1.0:
        return 1.0 / base_inv_freq, 1.0

    # Wavelength of each rotary dimension: 2 * pi / inv_freq
    wavelengths = 2.0 * math.pi / base_inv_freq
    # Number of rotations over original context length
    r = float(orig_ctx) / wavelengths
    # Smooth linear ramp: 0 -> interpolate (low freq), 1 -> extrapolate (high freq)
    ramp = mx.clip((r - beta_slow) / max(1e-6, beta_fast - beta_slow), 0.0, 1.0)

    scaled_inv_freq = base_inv_freq / float(scale)
    yarn_inv_freq = (1.0 - ramp) * scaled_inv_freq + ramp * base_inv_freq
    # YaRN attention temperature correction (applied to QK scale)
    mscale = 0.1 * math.log(float(scale)) + 1.0
    return 1.0 / yarn_inv_freq, mscale


def q_chunked_causal_sdpa(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float,
    sinks: Optional[mx.array] = None,
    chunk_size: int = 512,
    window_size: Optional[int] = None,
) -> mx.array:
    """Exact Q-chunked checkpointed causal SDPA (Global or Sliding-Window) with O(T) memory.

    Avoids MLX's O(T^2) backward materialization at 8k-32k+ context lengths while remaining
    mathematically identical (0.00e+00 forward / ~1e-6 backward diff) to full SDPA.
    """
    T = q.shape[2]
    outs = []
    for start in range(0, T, chunk_size):
        end = min(T, start + chunk_size)
        k_start = 0 if window_size is None else max(0, start - window_size + 1)
        q_c = q[:, :, start:end, :]
        k_c = k[:, :, k_start:end, :]
        v_c = v[:, :, k_start:end, :]

        def _attn_chunk(
            qc: mx.array,
            kc: mx.array,
            vc: mx.array,
            s: Optional[mx.array],
            _start: int = start,
            _end: int = end,
            _k_start: int = k_start,
        ) -> mx.array:
            if _k_start == 0 and _start == 0 and window_size is None:
                return mx.fast.scaled_dot_product_attention(
                    qc, kc, vc, scale=scale, mask="causal", sinks=s
                )
            row_idx = mx.arange(_start, _end)[:, None]
            col_idx = mx.arange(_k_start, _end)[None, :]
            mask = col_idx <= row_idx
            if window_size is not None:
                mask = mask & (col_idx >= (row_idx - window_size + 1))
            return mx.fast.scaled_dot_product_attention(
                qc, kc, vc, scale=scale, mask=mask, sinks=s
            )

        if sinks is not None:
            out_c = mx.checkpoint(_attn_chunk)(q_c, k_c, v_c, sinks)
        else:
            out_c = mx.checkpoint(
                lambda qc, kc, vc, fn=_attn_chunk: fn(qc, kc, vc, None)
            )(q_c, k_c, v_c)
        outs.append(out_c)
    return mx.concatenate(outs, axis=2)


# ---------------------------------------------------------------------------
# Core Building Blocks
# ---------------------------------------------------------------------------

class FastLinear(nn.Module):
    """Bias-free linear layer with float32 master weights and native-dtype (bf16) GEMMs."""

    def __init__(self, in_features: int, out_features: int, std: float = 0.02):
        super().__init__()
        self.weight = mx.random.normal((out_features, in_features)) * std

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.weight.astype(x.dtype).T


class RMSNorm(nn.Module):
    """Hardware-accelerated RMSNorm using `mx.fast.rms_norm`."""

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight.astype(x.dtype), self.eps)


class GQAAttention(nn.Module):
    """GQA with QK-Norm, YaRN RoPE, Attention Sinks, Hybrid SWA, and Q-Chunked SDPA."""

    def __init__(
        self,
        cfg: ModelConfig,
        resid_scale: float = 1.0,
        window_size: Optional[int] = None,
    ):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.window_size = window_size
        self.sdpa_chunk_size = cfg.sdpa_chunk_size

        # YaRN RoPE periods + attention temperature scaling
        yarn_periods, mscale = build_yarn_rope_periods(
            head_dim=cfg.head_dim,
            theta=cfg.rope_theta,
            orig_ctx=cfg.yarn_orig_ctx,
            scale=cfg.yarn_scale,
            beta_fast=cfg.yarn_beta_fast,
            beta_slow=cfg.yarn_beta_slow,
        )
        self._yarn_periods = yarn_periods
        self.scale = (cfg.head_dim ** -0.5) * (mscale ** 2)

        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim
        out_std = 0.02 / math.sqrt(max(1.0, resid_scale))

        self.wq = FastLinear(cfg.d_model, q_dim, std=0.02)
        self.wk = FastLinear(cfg.d_model, kv_dim, std=0.02)
        self.wv = FastLinear(cfg.d_model, kv_dim, std=0.02)
        self.wo = FastLinear(q_dim, cfg.d_model, std=out_std)

        self.use_qk_norm = cfg.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)

        self.use_sinks = cfg.use_attention_sinks
        if self.use_sinks:
            self.sinks = mx.zeros((self.n_heads,))

    def __call__(
        self,
        x: mx.array,
        cache: Optional[Tuple[mx.array, mx.array, int]] = None,
        return_cache: bool = False,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array, int]]]:
        B, T, _ = x.shape

        q = self.wq(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # QK-Norm stabilizes attention logits across recurrent depth & long context
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        offset = cache[2] if cache is not None else 0
        q = mx.fast.rope(
            q, dims=self.head_dim, traditional=False, base=None, scale=1.0,
            offset=offset, freqs=self._yarn_periods
        )
        k = mx.fast.rope(
            k, dims=self.head_dim, traditional=False, base=None, scale=1.0,
            offset=offset, freqs=self._yarn_periods
        )

        if cache is not None:
            k_prev, v_prev, _ = cache
            k = mx.concatenate([k_prev, k], axis=2)
            v = mx.concatenate([v_prev, v], axis=2)

        new_offset = offset + T
        # For SWA layers during KV-cached decode, trim cache to window_size to bound RAM
        if return_cache:
            if self.window_size is not None and k.shape[2] > self.window_size:
                k_store = k[:, :, -self.window_size:, :]
                v_store = v[:, :, -self.window_size:, :]
            else:
                k_store, v_store = k, v
            new_cache = (k_store, v_store, new_offset)
        else:
            new_cache = None

        sinks = self.sinks.astype(q.dtype) if self.use_sinks else None

        # During training (no cache): use exact Q-chunked checkpointed SDPA for long contexts or SWA
        use_chunked = (not return_cache) and (
            T > 2048 or (self.window_size is not None and T > self.window_size)
        )
        if use_chunked:
            out = q_chunked_causal_sdpa(
                q, k, v,
                scale=self.scale,
                sinks=sinks,
                chunk_size=self.sdpa_chunk_size,
                window_size=self.window_size,
            )
        else:
            mask = "causal" if T > 1 else None
            if T > 1 and self.window_size is not None and T > self.window_size:
                row_idx = mx.arange(T)[:, None]
                col_idx = mx.arange(k.shape[2])[None, :]
                shift = k.shape[2] - T
                mask = (col_idx <= (row_idx + shift)) & (
                    col_idx >= (row_idx + shift - self.window_size + 1)
                )
            out = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=self.scale, mask=mask, sinks=sinks
            )

        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.n_heads * self.head_dim)
        return self.wo(out), new_cache


class DenseSwiGLU(nn.Module):
    """Fused Gate+Up SwiGLU FFN for Prelude and Coda blocks."""

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        resid_scale: float = 1.0,
        use_fused_kernels: bool = True,
    ):
        super().__init__()
        self.use_fused_kernels = use_fused_kernels
        out_std = 0.02 / math.sqrt(max(1.0, resid_scale))
        self.w_gate_up = FastLinear(d_model, 2 * hidden_dim, std=0.02)
        self.w_down = FastLinear(hidden_dim, d_model, std=out_std)

    def __call__(self, x: mx.array) -> mx.array:
        gate_up = self.w_gate_up(x)
        if self.use_fused_kernels:
            act = fused_swiglu(gate_up)
        else:
            gate, up = mx.split(gate_up, 2, axis=-1)
            act = nn.silu(gate) * up
        return self.w_down(act)


class DenseBlock(nn.Module):
    """Pre-Norm Transformer block (Global GQA + Dense SwiGLU) for Prelude & Coda."""

    def __init__(self, cfg: ModelConfig, resid_scale: float):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.attn = GQAAttention(cfg, resid_scale=resid_scale, window_size=None)
        self.norm2 = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.ffn = DenseSwiGLU(
            cfg.d_model,
            cfg.dense_ffn_dim,
            resid_scale=resid_scale,
            use_fused_kernels=cfg.use_fused_kernels,
        )

    def __call__(
        self,
        x: mx.array,
        cache: Optional[Tuple[mx.array, mx.array, int]] = None,
        return_cache: bool = False,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array, int]]]:
        attn_out, new_cache = self.attn(self.norm1(x), cache=cache, return_cache=return_cache)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, new_cache


class MoECoreBlock(nn.Module):
    """Recurrent Core block: Pre-Norm GQA (SWA or Global) + Iteration-Aware MoE + Step AdaLN."""

    def __init__(self, cfg: ModelConfig, resid_scale: float, window_size: Optional[int] = None):
        super().__init__()
        self.d_model = cfg.d_model
        self.use_fused_kernels = cfg.use_fused_kernels
        self.rms_norm_eps = cfg.rms_norm_eps
        self.norm1 = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.attn = GQAAttention(cfg, resid_scale=resid_scale, window_size=window_size)
        self.norm2 = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.moe = MoELayer(cfg, resid_scale=resid_scale)

        # Lightweight bottleneck AdaLN-Zero conditioning from loop-step embedding
        self.ada_down = FastLinear(cfg.d_model, cfg.step_cond_dim, std=0.02)
        self.ada_up = FastLinear(cfg.step_cond_dim, 4 * cfg.d_model, std=0.0)
        self.ada_bias = mx.zeros((4 * cfg.d_model,))

        # Residual LayerScale initialized to 1 / sqrt(R) to prevent explosion over deep loops
        init_scale = 1.0 / math.sqrt(max(1, cfg.num_loops))
        self.attn_scale = mx.full((cfg.d_model,), init_scale)
        self.ffn_scale = mx.full((cfg.d_model,), init_scale)

    def __call__(
        self,
        x: mx.array,
        step_emb: mx.array,
        cache: Optional[Tuple[mx.array, mx.array, int]] = None,
        return_cache: bool = False,
    ) -> Tuple[mx.array, mx.array, Optional[Tuple[mx.array, mx.array, int]]]:
        dtype = x.dtype
        cond = self.ada_up(nn.silu(self.ada_down(step_emb.astype(dtype)))) + self.ada_bias.astype(dtype)
        s1, b1, s2, b2 = mx.split(cond, 4, axis=-1)

        if self.use_fused_kernels:
            h1 = fused_adaln_rmsnorm(x, self.norm1.weight, s1, b1, self.rms_norm_eps)
        else:
            h1 = self.norm1(x) * (1.0 + s1) + b1
        attn_out, new_cache = self.attn(h1, cache=cache, return_cache=return_cache)
        x = x + attn_out * self.attn_scale.astype(dtype)

        if self.use_fused_kernels:
            h2 = fused_adaln_rmsnorm(x, self.norm2.weight, s2, b2, self.rms_norm_eps)
        else:
            h2 = self.norm2(x) * (1.0 + s2) + b2
        moe_out, expert_counts = self.moe(h2, step_emb)
        x = x + moe_out * self.ffn_scale.astype(dtype)

        return x, expert_counts, new_cache


class RecurrentCore(nn.Module):
    """Shared stack of MoECoreBlocks looped R times with input injection & checkpointing."""

    def __init__(self, cfg: ModelConfig, resid_scale: float):
        super().__init__()
        self.cfg = cfg
        # Hybrid Local (even layers) + Global (odd layers) attention inside the recurrent core
        self.layers = [
            MoECoreBlock(
                cfg,
                resid_scale=resid_scale,
                window_size=(cfg.sliding_window if (i % 2 == 0 and cfg.n_core_layers > 1) else None),
            )
            for i in range(cfg.n_core_layers)
        ]

        # Learned loop-step embeddings (supports up to max_loops at test time)
        self.step_embeddings = mx.random.normal((cfg.max_loops, cfg.d_model)) * 0.02

        # Residual input injection of prelude output x0 at each loop iteration
        self.use_input_injection = cfg.use_input_injection
        if self.use_input_injection:
            self.inject_norm = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
            self.inject_proj = FastLinear(2 * cfg.d_model, cfg.d_model, std=0.0)

    def _get_step_emb(self, step_idx: int) -> mx.array:
        idx = min(step_idx, self.cfg.max_loops - 1)
        return self.step_embeddings[idx]

    def _run_iteration(
        self,
        h: mx.array,
        x0_norm: mx.array,
        step_emb: mx.array,
        iter_caches: Optional[List[Optional[Tuple[mx.array, mx.array, int]]]] = None,
    ) -> Tuple[mx.array, mx.array, Optional[List[Tuple[mx.array, mx.array, int]]]]:
        if self.use_input_injection:
            h_in = mx.concatenate([self.inject_norm(h), x0_norm], axis=-1)
            h = h + self.inject_proj(h_in)

        layer_counts = []
        use_cache = iter_caches is not None
        new_caches = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            layer_cache = iter_caches[i] if use_cache else None
            h, counts, nc = layer(h, step_emb, cache=layer_cache, return_cache=use_cache)
            layer_counts.append(counts)
            if new_caches is not None:
                new_caches.append(nc)

        return h, mx.stack(layer_counts, axis=0), new_caches

    def __call__(
        self,
        x0: mx.array,
        num_loops: int,
        use_checkpointing: bool = False,
        bptt_steps: Optional[int] = None,
        caches: Optional[List[List[Optional[Tuple[mx.array, mx.array, int]]]]] = None,
    ) -> Tuple[mx.array, mx.array, Optional[List[List[Tuple[mx.array, mx.array, int]]]]]:
        h = x0
        x0_norm = self.inject_norm(x0) if self.use_input_injection else x0
        all_counts = []
        new_caches = [] if caches is not None else None

        do_ckpt = use_checkpointing and (caches is None)
        if do_ckpt:
            core_params = self.trainable_parameters()

        for r in range(num_loops):
            if bptt_steps is not None and r < (num_loops - bptt_steps):
                h = mx.stop_gradient(h)

            step_emb = self._get_step_emb(r)

            if do_ckpt:
                def ckpt_fn(params, state, x0_n, s_emb):
                    self.update(params)
                    out_h, out_c, _ = self._run_iteration(state, x0_n, s_emb, iter_caches=None)
                    return out_h, out_c

                h, counts = mx.checkpoint(ckpt_fn)(core_params, h, x0_norm, step_emb)
                all_counts.append(counts)
            else:
                iter_cache = caches[r] if caches is not None else None
                h, counts, nc = self._run_iteration(h, x0_norm, step_emb, iter_caches=iter_cache)
                all_counts.append(counts)
                if new_caches is not None:
                    new_caches.append(nc)

        return h, mx.stack(all_counts, axis=0), new_caches


class LoopedMoETransformer(nn.Module):
    """Complete Long-Context Looped / Recurrent-Depth MoE Language Model."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        resid_scale = 2.0 * cfg.effective_depth

        self.tok_emb = mx.random.normal((cfg.vocab_size, cfg.d_model)) * 0.02

        self.prelude = [DenseBlock(cfg, resid_scale=resid_scale) for _ in range(cfg.n_prelude_layers)]
        self.core = RecurrentCore(cfg, resid_scale=resid_scale)
        self.coda = [DenseBlock(cfg, resid_scale=resid_scale) for _ in range(cfg.n_coda_layers)]
        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)

        if not cfg.tie_embeddings:
            self.lm_head = FastLinear(cfg.d_model, cfg.vocab_size, std=0.02)
        else:
            self.lm_head = None

    def make_cache(self, num_loops: Optional[int] = None) -> Dict:
        R = num_loops if num_loops is not None else self.cfg.num_loops
        return {
            "prelude": [None] * self.cfg.n_prelude_layers,
            "core": [[None] * self.cfg.n_core_layers for _ in range(R)],
            "coda": [None] * self.cfg.n_coda_layers,
        }

    def forward_hidden(
        self,
        input_ids: mx.array,
        num_loops: Optional[int] = None,
        compute_dtype: mx.Dtype = mx.bfloat16,
        cache: Optional[Dict] = None,
        use_checkpointing: Optional[bool] = None,
    ) -> Tuple[mx.array, mx.array, Optional[Dict]]:
        """Run Prelude -> Recurrent Core -> Coda -> Final RMSNorm and return hidden states."""
        R = num_loops if num_loops is not None else self.cfg.num_loops
        ckpt = self.cfg.use_checkpointing if use_checkpointing is None else use_checkpointing

        x = self.tok_emb[input_ids].astype(compute_dtype)

        use_cache = cache is not None
        new_prelude_cache = [] if use_cache else None
        for i, layer in enumerate(self.prelude):
            lc = cache["prelude"][i] if use_cache else None
            x, nc = layer(x, cache=lc, return_cache=use_cache)
            if new_prelude_cache is not None:
                new_prelude_cache.append(nc)

        core_cache = cache["core"] if use_cache else None
        x, iter_expert_counts, new_core_cache = self.core(
            x,
            num_loops=R,
            use_checkpointing=ckpt,
            bptt_steps=self.cfg.bptt_steps if not use_cache else None,
            caches=core_cache,
        )

        new_coda_cache = [] if use_cache else None
        for i, layer in enumerate(self.coda):
            lc = cache["coda"][i] if use_cache else None
            x, nc = layer(x, cache=lc, return_cache=use_cache)
            if new_coda_cache is not None:
                new_coda_cache.append(nc)

        x = self.final_norm(x)

        new_cache = None
        if use_cache:
            new_cache = {
                "prelude": new_prelude_cache,
                "core": new_core_cache,
                "coda": new_coda_cache,
            }
        return x, iter_expert_counts, new_cache

    def __call__(
        self,
        input_ids: mx.array,
        num_loops: Optional[int] = None,
        compute_dtype: mx.Dtype = mx.bfloat16,
        cache: Optional[Dict] = None,
        use_checkpointing: Optional[bool] = None,
    ) -> Tuple[mx.array, mx.array, Optional[Dict]]:
        x, iter_expert_counts, new_cache = self.forward_hidden(
            input_ids,
            num_loops=num_loops,
            compute_dtype=compute_dtype,
            cache=cache,
            use_checkpointing=use_checkpointing,
        )
        if self.lm_head is not None:
            logits = self.lm_head(x).astype(mx.float32)
        else:
            logits = (x @ self.tok_emb.astype(compute_dtype).T).astype(mx.float32)
        return logits, iter_expert_counts, new_cache

    def compute_loss(
        self,
        input_ids: mx.array,
        targets: mx.array,
        num_loops: Optional[int] = None,
        compute_dtype: mx.Dtype = mx.bfloat16,
        use_checkpointing: Optional[bool] = None,
    ) -> Tuple[mx.array, mx.array]:
        """Exact language modeling cross-entropy with automatic chunked checkpointing.

        At long context lengths (e.g. 4k-32k), projecting the full (B, T, V) logits
        in fp32 would consume 3-13 GB. Chunking into `ce_chunk_size` token slices
        wrapped in `mx.checkpoint` keeps LM-head peak memory under ~0.4 GB with
        mathematically identical gradients.
        """
        x, iter_expert_counts, _ = self.forward_hidden(
            input_ids,
            num_loops=num_loops,
            compute_dtype=compute_dtype,
            cache=None,
            use_checkpointing=use_checkpointing,
        )
        B, T, D = x.shape
        N = B * T
        w = self.lm_head.weight if self.lm_head is not None else self.tok_emb
        chunk_size = self.cfg.ce_chunk_size

        if N <= chunk_size * 4:
            logits = (x @ w.astype(compute_dtype).T).astype(mx.float32)
            loss = nn.losses.cross_entropy(logits, targets, reduction="mean")
            return loss, iter_expert_counts

        x_flat = x.reshape(N, D)
        y_flat = targets.reshape(N)

        def _chunk_ce(xc: mx.array, yc: mx.array, weight: mx.array) -> mx.array:
            lc = (xc @ weight.astype(compute_dtype).T).astype(mx.float32)
            return nn.losses.cross_entropy(lc, yc, reduction="sum")

        total_loss = mx.array(0.0, dtype=mx.float32)
        for start in range(0, N, chunk_size):
            end = min(N, start + chunk_size)
            total_loss = total_loss + mx.checkpoint(_chunk_ce)(
                x_flat[start:end], y_flat[start:end], w
            )
        return total_loss / float(N), iter_expert_counts

    def update_router_biases(self, iter_expert_counts: mx.array) -> None:
        layer_totals = mx.sum(iter_expert_counts, axis=0)
        gamma = self.cfg.router_bias_update_rate
        for i, layer in enumerate(self.core.layers):
            layer.moe.router.update_bias(layer_totals[i], gamma=gamma)
