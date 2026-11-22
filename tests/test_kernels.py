"""Forward & backward numerical parity tests for custom Apple Metal fused kernels."""

import mlx.core as mx
import mlx.nn as nn

from loopmoe.kernels import (
    _ref_adaln_rmsnorm,
    _ref_router_topk,
    _ref_swiglu,
    fused_adaln_rmsnorm,
    fused_router_topk_sigmoid,
    fused_swiglu,
)


def test_fused_swiglu_parity():
    mx.random.seed(0)
    for dtype, rel_tol in [(mx.float32, 1e-5), (mx.bfloat16, 1.5e-2)]:
        x = mx.random.normal((512, 1024)).astype(dtype)
        cot = mx.random.normal((512, 512)).astype(dtype)

        y_ref = _ref_swiglu(x)
        y_ker = fused_swiglu(x)
        fwd_rel = float(mx.max(mx.abs(y_ref.astype(mx.float32) - y_ker.astype(mx.float32))).item()) / max(
            1.0, float(mx.max(mx.abs(y_ref.astype(mx.float32))).item())
        )

        g_ref = mx.grad(lambda z: (_ref_swiglu(z) * cot).astype(mx.float32).sum())(x)
        g_ker = mx.grad(lambda z: (fused_swiglu(z) * cot).astype(mx.float32).sum())(x)
        bwd_rel = float(mx.max(mx.abs(g_ref.astype(mx.float32) - g_ker.astype(mx.float32))).item()) / max(
            1.0, float(mx.max(mx.abs(g_ref.astype(mx.float32))).item())
        )
        assert fwd_rel <= rel_tol
        assert bwd_rel <= rel_tol


def test_fused_adaln_rmsnorm_parity():
    mx.random.seed(1)
    for dtype, rel_tol in [(mx.float32, 1e-4), (mx.bfloat16, 2e-2)]:
        x = mx.random.normal((2, 128, 512)).astype(dtype)
        w = mx.ones((512,)) + mx.random.normal((512,)) * 0.1
        s = mx.random.normal((512,)) * 0.1
        b = mx.random.normal((512,)) * 0.1
        cot = mx.random.normal((2, 128, 512)).astype(dtype)

        y_ref = _ref_adaln_rmsnorm(x, w, s, b)
        y_ker = fused_adaln_rmsnorm(x, w, s, b)
        fwd_rel = float(mx.max(mx.abs(y_ref.astype(mx.float32) - y_ker.astype(mx.float32))).item()) / max(
            1.0, float(mx.max(mx.abs(y_ref.astype(mx.float32))).item())
        )

        grads_ref = mx.grad(
            lambda a, b_, c, d: (_ref_adaln_rmsnorm(a, b_, c, d) * cot).astype(mx.float32).sum(),
            argnums=(0, 1, 2, 3),
        )(x, w, s, b)
        grads_ker = mx.grad(
            lambda a, b_, c, d: (fused_adaln_rmsnorm(a, b_, c, d) * cot).astype(mx.float32).sum(),
            argnums=(0, 1, 2, 3),
        )(x, w, s, b)
        dx_rel = float(mx.max(mx.abs(grads_ref[0].astype(mx.float32) - grads_ker[0].astype(mx.float32))).item()) / max(
            1.0, float(mx.max(mx.abs(grads_ref[0].astype(mx.float32))).item())
        )
        assert fwd_rel <= rel_tol
        assert dx_rel <= rel_tol


def test_fused_router_topk_parity():
    mx.random.seed(2)
    logits = mx.random.normal((1024, 8), dtype=mx.float32)
    bias = mx.random.normal((8,), dtype=mx.float32) * 0.05
    cot_w = mx.random.normal((1024, 2), dtype=mx.float32)

    idx_ref, w_ref = _ref_router_topk(logits, bias, 2)
    idx_ker, w_ker = fused_router_topk_sigmoid(logits, bias, 2)
    assert bool(mx.all(idx_ref == idx_ker).item())
    assert float(mx.max(mx.abs(w_ref - w_ker)).item()) <= 1e-6

    g_ref = mx.grad(lambda l: (_ref_router_topk(l, bias, 2)[1] * cot_w).sum())(logits)
    g_ker = mx.grad(lambda l: (fused_router_topk_sigmoid(l, bias, 2)[1] * cot_w).sum())(logits)
    assert float(mx.max(mx.abs(g_ref - g_ker)).item()) <= 1e-6


if __name__ == "__main__":
    test_fused_swiglu_parity()
    test_fused_adaln_rmsnorm_parity()
    test_fused_router_topk_parity()
    print("All kernel parity tests PASSED!")
