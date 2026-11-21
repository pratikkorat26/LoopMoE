"""Custom Apple Metal Fused Kernels (`mx.fast.metal_kernel`) + Parity Verification.

Fuses the highest-overhead non-GEMM hotspots in the Looped MoE Transformer:
  1. `fused_swiglu`:        Single-pass split + SiLU + multiply on concatenated (..., 2*H)
                            with custom Metal backward VJP (eliminates split copies & 3 dispatches).
  2. `fused_adaln_rmsnorm`: Single-pass SIMD-reduced RMSNorm + AdaLN-Zero (1+scale)+shift
                            with custom Metal backward VJP (eliminates 6 dispatches/block/loop).
  3. `fused_router_topk`:   Register-resident sigmoid + dynamic bias + top-K + unbiased
                            weight normalization in 1 dispatch (replaces 7 dispatches/layer/loop),
                            with custom Metal backward VJP into router logits.

Run `python kernels.py` to execute the full forward + backward numerical parity suite
and M4 micro-benchmarks against pure-MLX reference implementations.
"""

import time
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn


# ===========================================================================
# 1. Fused Split-SwiGLU Metal Kernel (Forward + Backward)
# ===========================================================================

_SWIGLU_FWD_SRC = """
    uint elem = thread_position_in_grid.x;
    uint H = gate_up_shape[gate_up_ndim - 1] / 2;
    uint total = 1;
    for (int i = 0; i < gate_up_ndim - 1; ++i) total *= gate_up_shape[i];
    total *= H;
    if (elem >= total) return;

    uint row = elem / H;
    uint col = elem - row * H;
    uint base = row * (2 * H);

    float g = float(gate_up[base + col]);
    float u = float(gate_up[base + H + col]);
    float sig = 1.0f / (1.0f + metal::exp(-g));
    out[elem] = T((g * sig) * u);
"""

_SWIGLU_BWD_SRC = """
    uint elem = thread_position_in_grid.x;
    uint H = cotangent_shape[cotangent_ndim - 1];
    uint total = 1;
    for (int i = 0; i < cotangent_ndim; ++i) total *= cotangent_shape[i];
    if (elem >= total) return;

    uint row = elem / H;
    uint col = elem - row * H;
    uint base = row * (2 * H);

    float g = float(gate_up[base + col]);
    float u = float(gate_up[base + H + col]);
    float dy = float(cotangent[elem]);

    float sig = 1.0f / (1.0f + metal::exp(-g));
    float silu_g = g * sig;
    float dsilu_dg = sig * (1.0f + g * (1.0f - sig));

    d_gate_up[base + col] = T(dy * u * dsilu_dg);
    d_gate_up[base + H + col] = T(dy * silu_g);
"""

_swiglu_fwd_kernel = mx.fast.metal_kernel(
    name="fused_swiglu_fwd",
    input_names=["gate_up"],
    output_names=["out"],
    source=_SWIGLU_FWD_SRC,
)

_swiglu_bwd_kernel = mx.fast.metal_kernel(
    name="fused_swiglu_bwd",
    input_names=["gate_up", "cotangent"],
    output_names=["d_gate_up"],
    source=_SWIGLU_BWD_SRC,
)


@mx.custom_function
def fused_swiglu(gate_up: mx.array) -> mx.array:
    """Compute silu(gate) * up directly from concatenated (..., 2*H) tensor in 1 Metal kernel."""
    out_shape = gate_up.shape[:-1] + (gate_up.shape[-1] // 2,)
    n_elems = 1
    for s in out_shape:
        n_elems *= s
    tg = min(256, max(32, n_elems))
    grid_x = ((n_elems + tg - 1) // tg) * tg
    return _swiglu_fwd_kernel(
        inputs=[gate_up],
        template=[("T", gate_up.dtype)],
        grid=(grid_x, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[out_shape],
        output_dtypes=[gate_up.dtype],
    )[0]


@fused_swiglu.vjp
def _fused_swiglu_vjp(primals, cotangents, outputs):
    gate_up = primals
    dy = cotangents
    n_elems = dy.size
    tg = min(256, max(32, n_elems))
    grid_x = ((n_elems + tg - 1) // tg) * tg
    (d_gate_up,) = _swiglu_bwd_kernel(
        inputs=[gate_up, dy],
        template=[("T", gate_up.dtype)],
        grid=(grid_x, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[gate_up.shape],
        output_dtypes=[gate_up.dtype],
    )
    return d_gate_up


# ===========================================================================
# 2. Fused AdaLN-Zero + RMSNorm Metal Kernel (Forward + Backward)
# ===========================================================================

_ADALN_RMS_FWD_SRC = """
    // One threadgroup (256 threads) per token row of length D
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint tg_size = threads_per_threadgroup.x;
    uint D = x_shape[x_ndim - 1];
    uint base = row * D;

    threadgroup float shared_sum[32];
    float local_sq = 0.0f;
    for (uint j = tid; j < D; j += tg_size) {
        float v = float(x[base + j]);
        local_sq += v * v;
    }

    // Two-level SIMD + threadgroup reduction
    float simd_sq = metal::simd_sum(local_sq);
    uint simd_lane = tid & 31;
    uint simd_id = tid >> 5;
    if (simd_lane == 0) {
        shared_sum[simd_id] = simd_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_sq = 0.0f;
    if (simd_id == 0) {
        uint n_simds = (tg_size + 31) >> 5;
        float val = (simd_lane < n_simds) ? shared_sum[simd_lane] : 0.0f;
        total_sq = metal::simd_sum(val);
        if (simd_lane == 0) {
            shared_sum[0] = metal::rsqrt(total_sq / float(D) + eps[0]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_rms = shared_sum[0];
    if (tid == 0) {
        inv_rms_out[row] = inv_rms;
    }
    for (uint j = tid; j < D; j += tg_size) {
        float x_norm = float(x[base + j]) * inv_rms;
        float eff_w = float(weight[j]) * (1.0f + float(scale[j]));
        out[base + j] = T(x_norm * eff_w + float(shift[j]));
    }
"""

_ADALN_RMS_DX_SRC = """
    // Exact dx for y_j = (x_j * inv_rms) * eff_w_j + shift_j
    // dx_j = inv_rms * (g_j - x_j * inv_rms^2 * mean_k(g_k * x_k))  where g_j = dy_j * eff_w_j
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint tg_size = threads_per_threadgroup.x;
    uint D = x_shape[x_ndim - 1];
    uint base = row * D;

    float inv_rms = inv_rms_in[row];
    threadgroup float shared_dot[32];

    float local_dot = 0.0f;
    for (uint j = tid; j < D; j += tg_size) {
        float xj = float(x[base + j]);
        float eff_w = float(weight[j]) * (1.0f + float(scale[j]));
        float gj = float(dy[base + j]) * eff_w;
        local_dot += gj * xj;
    }

    float simd_d = metal::simd_sum(local_dot);
    uint simd_lane = tid & 31;
    uint simd_id = tid >> 5;
    if (simd_lane == 0) {
        shared_dot[simd_id] = simd_d;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        uint n_simds = (tg_size + 31) >> 5;
        float val = (simd_lane < n_simds) ? shared_dot[simd_lane] : 0.0f;
        float total_dot = metal::simd_sum(val);
        if (simd_lane == 0) {
            shared_dot[0] = (total_dot / float(D)) * inv_rms * inv_rms;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float coeff = shared_dot[0];
    for (uint j = tid; j < D; j += tg_size) {
        float xj = float(x[base + j]);
        float eff_w = float(weight[j]) * (1.0f + float(scale[j]));
        float gj = float(dy[base + j]) * eff_w;
        dx[base + j] = T(inv_rms * (gj - xj * coeff));
    }
"""

_adaln_rms_fwd_kernel = mx.fast.metal_kernel(
    name="fused_adaln_rmsnorm_fwd",
    input_names=["x", "weight", "scale", "shift", "eps"],
    output_names=["out", "inv_rms_out"],
    source=_ADALN_RMS_FWD_SRC,
)

_adaln_rms_dx_kernel = mx.fast.metal_kernel(
    name="fused_adaln_rmsnorm_dx",
    input_names=["x", "weight", "scale", "dy", "inv_rms_in"],
    output_names=["dx"],
    source=_ADALN_RMS_DX_SRC,
)


@mx.custom_function
def _fused_adaln_rmsnorm_impl(
    x: mx.array,
    weight: mx.array,
    scale: mx.array,
    shift: mx.array,
    eps_arr: mx.array,
) -> mx.array:
    orig_shape = x.shape
    D = orig_shape[-1]
    N = x.size // D
    x_2d = x.reshape(N, D)
    tg = min(256, max(32, ((D + 31) // 32) * 32))
    out, _ = _adaln_rms_fwd_kernel(
        inputs=[x_2d, weight.astype(mx.float32), scale.astype(mx.float32), shift.astype(mx.float32), eps_arr],
        template=[("T", x.dtype)],
        grid=(N * tg, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(N, D), (N,)],
        output_dtypes=[x.dtype, mx.float32],
    )
    return out.reshape(orig_shape)


@_fused_adaln_rmsnorm_impl.vjp
def _fused_adaln_rmsnorm_vjp(primals, cotangents, outputs):
    x, weight, scale, shift, eps_arr = primals
    dy = cotangents
    orig_shape = x.shape
    D = orig_shape[-1]
    N = x.size // D
    x_2d = x.reshape(N, D)
    dy_2d = dy.reshape(N, D)
    w_f32 = weight.astype(mx.float32)
    s_f32 = scale.astype(mx.float32)
    sh_f32 = shift.astype(mx.float32)

    # Recompute inv_rms cheaply (single SIMD pass) or via fast mean
    x_f32 = x_2d.astype(mx.float32)
    inv_rms = mx.rsqrt(mx.mean(x_f32 * x_f32, axis=-1) + eps_arr)  # (N,)

    # dx via fused SIMD Metal kernel
    tg = min(256, max(32, ((D + 31) // 32) * 32))
    (dx_2d,) = _adaln_rms_dx_kernel(
        inputs=[x_2d, w_f32, s_f32, dy_2d, inv_rms],
        template=[("T", x.dtype)],
        grid=(N * tg, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(N, D)],
        output_dtypes=[x.dtype],
    )

    # Parameter gradients in float32
    dy_f32 = dy_2d.astype(mx.float32)
    x_norm = x_f32 * inv_rms[:, None]
    dot_dy_xnorm = mx.sum(dy_f32 * x_norm, axis=0)  # (D,)
    dweight = (dot_dy_xnorm * (1.0 + s_f32)).astype(weight.dtype)
    dscale = (dot_dy_xnorm * w_f32).astype(scale.dtype)
    dshift = mx.sum(dy_f32, axis=0).astype(shift.dtype)
    deps = mx.zeros_like(eps_arr)

    return (dx_2d.reshape(orig_shape), dweight, dscale, dshift, deps)


def fused_adaln_rmsnorm(
    x: mx.array,
    weight: mx.array,
    scale: mx.array,
    shift: mx.array,
    eps: float = 1e-6,
) -> mx.array:
    """Single-pass Metal RMSNorm + AdaLN modulation: rms_norm(x, weight, eps) * (1 + scale) + shift."""
    return _fused_adaln_rmsnorm_impl(x, weight, scale, shift, mx.array([eps], dtype=mx.float32))


# ===========================================================================
# 3. Fused Register-Resident MoE Router Top-K Sigmoid Kernel (Fwd + Bwd)
# ===========================================================================

_ROUTER_TOPK_FWD_SRC = """
    // One thread per token; E <= 32, K <= 8 fit entirely in registers!
    uint row = thread_position_in_grid.x;
    uint N = logits_shape[0];
    if (row >= N) return;

    uint E = logits_shape[1];
    uint base = row * E;

    float raw[32];
    float sel[32];
    for (uint e = 0; e < E; ++e) {
        float l = logits[base + e];
        float r = 1.0f / (1.0f + metal::exp(-l));
        raw[e] = r;
        sel[e] = r + expert_bias[e];
    }

    uint out_base = row * K;
    float w_sum = 0.0f;
    float chosen_raw[8];

    for (uint k = 0; k < K; ++k) {
        uint best_e = 0;
        float best_v = -1e30f;
        for (uint e = 0; e < E; ++e) {
            if (sel[e] > best_v) {
                best_v = sel[e];
                best_e = e;
            }
        }
        sel[best_e] = -1e30f;
        top_k_idx[out_base + k] = int(best_e);
        chosen_raw[k] = raw[best_e];
        w_sum += raw[best_e];
    }

    float inv_sum = 1.0f / (w_sum + 1e-6f);
    for (uint k = 0; k < K; ++k) {
        top_k_weights[out_base + k] = chosen_raw[k] * inv_sum;
    }
"""

_ROUTER_TOPK_BWD_SRC = """
    // Exact VJP of normalized sigmoid top-K weights w.r.t. logits
    // w_k = r_k / (S + 1e-6),  dw_sum = sum_j(dw_j * w_j)
    // d(r_k) = (dw_k - dw_sum) / (S + 1e-6)
    // d(logit_e) = d(r_k) * r_k * (1 - r_k)  for chosen expert e = idx[k], else 0
    uint row = thread_position_in_grid.x;
    uint N = logits_shape[0];
    if (row >= N) return;

    uint E = logits_shape[1];
    uint K = top_k_idx_shape[1];
    uint base = row * E;
    uint out_base = row * K;

    for (uint e = 0; e < E; ++e) {
        d_logits[base + e] = 0.0f;
    }

    float r_k[8];
    int idx_k[8];
    float S = 0.0f;
    for (uint k = 0; k < K; ++k) {
        int e = top_k_idx[out_base + k];
        idx_k[k] = e;
        float l = logits[base + e];
        float r = 1.0f / (1.0f + metal::exp(-l));
        r_k[k] = r;
        S += r;
    }

    float inv_S = 1.0f / (S + 1e-6f);
    float dot = 0.0f;
    for (uint k = 0; k < K; ++k) {
        float w = r_k[k] * inv_S;
        dot += d_weights[out_base + k] * w;
    }

    for (uint k = 0; k < K; ++k) {
        float dr = (d_weights[out_base + k] - dot) * inv_S;
        float r = r_k[k];
        d_logits[base + idx_k[k]] = dr * r * (1.0f - r);
    }
"""

_router_topk_fwd_kernel = mx.fast.metal_kernel(
    name="fused_router_topk_fwd",
    input_names=["logits", "expert_bias"],
    output_names=["top_k_idx", "top_k_weights"],
    source=_ROUTER_TOPK_FWD_SRC,
)

_router_topk_bwd_kernel = mx.fast.metal_kernel(
    name="fused_router_topk_bwd",
    input_names=["logits", "top_k_idx", "d_weights"],
    output_names=["d_logits"],
    source=_ROUTER_TOPK_BWD_SRC,
)


def _make_fused_router_topk(top_k: int):
    @mx.custom_function
    def _fn(logits: mx.array, expert_bias: mx.array) -> Tuple[mx.array, mx.array]:
        N = logits.shape[0]
        tg = min(256, max(32, N))
        grid_x = ((N + tg - 1) // tg) * tg
        idx, weights = _router_topk_fwd_kernel(
            inputs=[logits.astype(mx.float32), expert_bias.astype(mx.float32)],
            template=[("K", top_k)],
            grid=(grid_x, 1, 1),
            threadgroup=(tg, 1, 1),
            output_shapes=[(N, top_k), (N, top_k)],
            output_dtypes=[mx.int32, mx.float32],
        )
        return idx, weights

    @_fn.vjp
    def _vjp(primals, cotangents, outputs):
        logits, expert_bias = primals
        _, d_weights = cotangents
        top_k_idx, _ = outputs
        N = logits.shape[0]
        tg = min(256, max(32, N))
        grid_x = ((N + tg - 1) // tg) * tg
        (d_logits,) = _router_topk_bwd_kernel(
            inputs=[logits.astype(mx.float32), top_k_idx, d_weights.astype(mx.float32)],
            grid=(grid_x, 1, 1),
            threadgroup=(tg, 1, 1),
            output_shapes=[logits.shape],
            output_dtypes=[mx.float32],
        )
        return d_logits.astype(logits.dtype), mx.zeros_like(expert_bias)

    return _fn


_ROUTER_TOPK_CACHE = {}


def fused_router_topk_sigmoid(
    logits: mx.array, expert_bias: mx.array, top_k: int
) -> Tuple[mx.array, mx.array]:
    """Single-dispatch Metal Top-K sigmoid router with unbiased weight normalization."""
    if top_k not in _ROUTER_TOPK_CACHE:
        _ROUTER_TOPK_CACHE[top_k] = _make_fused_router_topk(top_k)
    idx, weights = _ROUTER_TOPK_CACHE[top_k](logits, expert_bias)
    return mx.stop_gradient(idx), weights


# ===========================================================================
# Strict Numerical Parity & Micro-Benchmark Suite
# ===========================================================================

def _ref_swiglu(gate_up: mx.array) -> mx.array:
    g, u = mx.split(gate_up, 2, axis=-1)
    return nn.silu(g) * u


def _ref_adaln_rmsnorm(x, weight, scale, shift, eps=1e-6):
    return mx.fast.rms_norm(x, weight.astype(x.dtype), eps) * (1.0 + scale.astype(x.dtype)) + shift.astype(x.dtype)


def _ref_router_topk(logits, expert_bias, top_k):
    raw = mx.sigmoid(logits)
    sel = raw + expert_bias
    idx = mx.argsort(-sel, axis=-1)[:, :top_k]
    chosen = mx.take_along_axis(raw, idx, axis=-1)
    weights = chosen / (mx.sum(chosen, axis=-1, keepdims=True) + 1e-6)
    return idx, weights


def run_parity_checks() -> None:
    mx.random.seed(0)
    print("=" * 72)
    print("CUSTOM METAL FUSED KERNELS — FORWARD & BACKWARD PARITY CHECK")
    print("=" * 72)

    # 1. Fused SwiGLU (fp32 + bf16)
    for dtype, rel_tol in [(mx.float32, 1e-5), (mx.bfloat16, 1.5e-2)]:
        x = mx.random.normal((4096, 1024)).astype(dtype)
        cot = mx.random.normal((4096, 512)).astype(dtype)

        y_ref = _ref_swiglu(x)
        y_ker = fused_swiglu(x)
        fwd_err = float(mx.max(mx.abs(y_ref.astype(mx.float32) - y_ker.astype(mx.float32))).item())
        fwd_rel = fwd_err / max(1.0, float(mx.max(mx.abs(y_ref.astype(mx.float32))).item()))

        g_ref = mx.grad(lambda z: (_ref_swiglu(z) * cot).astype(mx.float32).sum())(x)
        g_ker = mx.grad(lambda z: (fused_swiglu(z) * cot).astype(mx.float32).sum())(x)
        bwd_err = float(mx.max(mx.abs(g_ref.astype(mx.float32) - g_ker.astype(mx.float32))).item())
        bwd_rel = bwd_err / max(1.0, float(mx.max(mx.abs(g_ref.astype(mx.float32))).item()))

        status = "PASS" if (fwd_rel <= rel_tol and bwd_rel <= rel_tol) else "FAIL"
        print(
            f"[{status}] fused_swiglu        ({str(dtype).split('.')[-1]:>8s}) | "
            f"fwd_rel={fwd_rel:.2e} (abs={fwd_err:.2e}) | bwd_rel={bwd_rel:.2e} (abs={bwd_err:.2e})"
        )
        assert status == "PASS", f"SwiGLU parity failed: fwd_rel={fwd_rel}, bwd_rel={bwd_rel}"

    # 2. Fused AdaLN + RMSNorm (fp32 + bf16)
    for dtype, rel_tol in [(mx.float32, 1e-4), (mx.bfloat16, 2e-2)]:
        x = mx.random.normal((8, 512, 512)).astype(dtype)
        w = mx.ones((512,)) + mx.random.normal((512,)) * 0.1
        s = mx.random.normal((512,)) * 0.1
        b = mx.random.normal((512,)) * 0.1
        cot = mx.random.normal((8, 512, 512)).astype(dtype)

        y_ref = _ref_adaln_rmsnorm(x, w, s, b)
        y_ker = fused_adaln_rmsnorm(x, w, s, b)
        fwd_err = float(mx.max(mx.abs(y_ref.astype(mx.float32) - y_ker.astype(mx.float32))).item())
        fwd_rel = fwd_err / max(1.0, float(mx.max(mx.abs(y_ref.astype(mx.float32))).item()))

        grads_ref = mx.grad(lambda a, b_, c, d: (_ref_adaln_rmsnorm(a, b_, c, d) * cot).astype(mx.float32).sum(), argnums=(0, 1, 2, 3))(x, w, s, b)
        grads_ker = mx.grad(lambda a, b_, c, d: (fused_adaln_rmsnorm(a, b_, c, d) * cot).astype(mx.float32).sum(), argnums=(0, 1, 2, 3))(x, w, s, b)
        dx_err = float(mx.max(mx.abs(grads_ref[0].astype(mx.float32) - grads_ker[0].astype(mx.float32))).item())
        dx_rel = dx_err / max(1.0, float(mx.max(mx.abs(grads_ref[0].astype(mx.float32))).item()))
        param_rel = max(
            float(mx.max(mx.abs(gr.astype(mx.float32) - gk.astype(mx.float32))).item())
            / max(1.0, float(mx.max(mx.abs(gr.astype(mx.float32))).item()))
            for gr, gk in zip(grads_ref[1:], grads_ker[1:])
        )
        status = "PASS" if (fwd_rel <= rel_tol and dx_rel <= rel_tol and param_rel <= rel_tol) else "FAIL"
        print(
            f"[{status}] fused_adaln_rmsnorm ({str(dtype).split('.')[-1]:>8s}) | "
            f"fwd_rel={fwd_rel:.2e} | dx_rel={dx_rel:.2e} | param_rel={param_rel:.2e}"
        )
        assert status == "PASS", f"AdaLN RMSNorm parity failed: {fwd_rel}, {dx_rel}, {param_rel}"

    # 3. Fused Router Top-K Sigmoid (fp32)
    logits = mx.random.normal((4096, 8), dtype=mx.float32)
    bias = mx.random.normal((8,), dtype=mx.float32) * 0.05
    cot_w = mx.random.normal((4096, 2), dtype=mx.float32)

    idx_ref, w_ref = _ref_router_topk(logits, bias, 2)
    idx_ker, w_ker = fused_router_topk_sigmoid(logits, bias, 2)
    idx_match = bool(mx.all(idx_ref == idx_ker).item())
    w_err = float(mx.max(mx.abs(w_ref - w_ker)).item())

    g_ref = mx.grad(lambda l: (_ref_router_topk(l, bias, 2)[1] * cot_w).sum())(logits)
    g_ker = mx.grad(lambda l: (fused_router_topk_sigmoid(l, bias, 2)[1] * cot_w).sum())(logits)
    g_err = float(mx.max(mx.abs(g_ref - g_ker)).item())

    status = "PASS" if (idx_match and w_err <= 1e-6 and g_err <= 1e-6) else "FAIL"
    print(f"[{status}] fused_router_topk  ( float32) | idx_exact={idx_match} | w_err={w_err:.2e} | dlogits={g_err:.2e}")
    assert status == "PASS", f"Router parity failed: idx={idx_match}, w={w_err}, g={g_err}"

    # Quick micro-benchmarks (50 iters)
    print("-" * 72)
    print("KERNEL MICRO-BENCHMARKS (Apple M4, 4096 tokens, bfloat16/fp32):")

    def bench(fn, iters=100):
        for _ in range(10):
            mx.eval(fn())
        t0 = time.perf_counter()
        for _ in range(iters):
            mx.eval(fn())
        return (time.perf_counter() - t0) * 1000.0 / iters

    gu = mx.random.normal((8192, 1024)).astype(mx.bfloat16)
    t_ref = bench(lambda: _ref_swiglu(gu))
    t_ker = bench(lambda: fused_swiglu(gu))
    print(f"  SwiGLU (8192x1024 bf16):      ref={t_ref:.3f} ms | fused={t_ker:.3f} ms | speedup={t_ref/t_ker:.2f}x")

    x_bf = mx.random.normal((8, 512, 512)).astype(mx.bfloat16)
    w_f = mx.ones((512,))
    s_f = mx.zeros((512,))
    b_f = mx.zeros((512,))
    t_ref = bench(lambda: _ref_adaln_rmsnorm(x_bf, w_f, s_f, b_f))
    t_ker = bench(lambda: fused_adaln_rmsnorm(x_bf, w_f, s_f, b_f))
    print(f"  AdaLN+RMSNorm (8x512x512):    ref={t_ref:.3f} ms | fused={t_ker:.3f} ms | speedup={t_ref/t_ker:.2f}x")

    t_ref = bench(lambda: _ref_router_topk(logits, bias, 2))
    t_ker = bench(lambda: fused_router_topk_sigmoid(logits, bias, 2))
    print(f"  Router Top-K (4096x8 fp32):   ref={t_ref:.3f} ms | fused={t_ker:.3f} ms | speedup={t_ref/t_ker:.2f}x")
    print("=" * 72)


if __name__ == "__main__":
    run_parity_checks()
