"""End-to-end model, Q-chunked SDPA, chunked CE, and KV-cache tests for LoopMoE."""

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from loopmoe.config import ModelConfig
from loopmoe.model import LoopedMoETransformer, q_chunked_causal_sdpa


def test_param_accounting_exact():
    cfg = ModelConfig()
    model = LoopedMoETransformer(cfg)
    actual_trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    assert actual_trainable == cfg.param_breakdown()["unique_total"]


def test_end_to_end_fused_vs_unfused_parity():
    mx.random.seed(42)
    cfg_ref = ModelConfig(
        vocab_size=256, d_model=128, n_heads=4, n_kv_heads=2, head_dim=32,
        dense_ffn_dim=256, expert_dim=128, shared_expert_dim=128,
        num_loops=3, use_fused_kernels=False, use_checkpointing=False,
    )
    cfg_fus = ModelConfig(
        vocab_size=256, d_model=128, n_heads=4, n_kv_heads=2, head_dim=32,
        dense_ffn_dim=256, expert_dim=128, shared_expert_dim=128,
        num_loops=3, use_fused_kernels=True, use_checkpointing=False,
    )
    m_ref = LoopedMoETransformer(cfg_ref)
    m_fus = LoopedMoETransformer(cfg_fus)
    m_fus.update(m_ref.parameters())

    x = mx.random.randint(0, 256, (2, 32))
    y = mx.random.randint(0, 256, (2, 32))

    def loss_fn(m, x, y):
        loss, counts = m.compute_loss(x, y, compute_dtype=mx.float32, use_checkpointing=False)
        return loss, mx.stop_gradient(counts)

    (l_ref, c_ref), g_ref = nn.value_and_grad(m_ref, loss_fn)(m_ref, x, y)
    (l_fus, c_fus), g_fus = nn.value_and_grad(m_fus, loss_fn)(m_fus, x, y)
    mx.eval(l_ref, l_fus, c_ref, c_fus, g_ref, g_fus)

    assert abs(l_ref.item() - l_fus.item()) < 1e-5
    assert bool(mx.all(c_ref == c_fus).item())
    max_gdiff = max(
        float(mx.max(mx.abs(gr - gf)).item())
        for (_, gr), (_, gf) in zip(tree_flatten(g_ref), tree_flatten(g_fus))
    )
    assert max_gdiff < 1e-4


def test_q_chunked_sdpa_exact_parity():
    mx.random.seed(7)
    q = mx.random.normal((1, 4, 256, 32)).astype(mx.float32)
    k = mx.random.normal((1, 2, 256, 32)).astype(mx.float32)
    v = mx.random.normal((1, 2, 256, 32)).astype(mx.float32)
    sinks = mx.random.normal((4,)).astype(mx.float32) * 0.1

    rv, rg = mx.value_and_grad(
        lambda q, k, v, s: mx.fast.scaled_dot_product_attention(q, k, v, scale=32**-0.5, mask="causal", sinks=s).sum(),
        argnums=(0, 1, 2, 3),
    )(q, k, v, sinks)
    cv, cg = mx.value_and_grad(
        lambda q, k, v, s: q_chunked_causal_sdpa(q, k, v, scale=32**-0.5, sinks=s, chunk_size=64).sum(),
        argnums=(0, 1, 2, 3),
    )(q, k, v, sinks)
    mx.eval(rv, rg, cv, cg)

    assert abs(rv.item() - cv.item()) < 1e-4
    for gr, gc in zip(rg, cg):
        assert float(mx.max(mx.abs(gr - gc)).item()) < 1e-4


def test_chunked_ce_and_kv_cache_parity():
    mx.random.seed(9)
    cfg = ModelConfig(
        vocab_size=256, d_model=128, n_heads=4, n_kv_heads=2, head_dim=32,
        dense_ffn_dim=256, expert_dim=128, shared_expert_dim=128,
        num_loops=2, ce_chunk_size=8, sliding_window=None, use_checkpointing=False,
    )
    model = LoopedMoETransformer(cfg)
    x = mx.random.randint(0, 256, (1, 16))

    # Full forward vs prefill + 1-token KV-cached decode at R=4 (test-time scaling)
    logits_full, _, _ = model(x, num_loops=4, compute_dtype=mx.float32, use_checkpointing=False)
    cache = model.make_cache(num_loops=4)
    _, _, cache = model(x[:, :-1], num_loops=4, compute_dtype=mx.float32, cache=cache, use_checkpointing=False)
    logits_step, _, _ = model(x[:, -1:], num_loops=4, compute_dtype=mx.float32, cache=cache, use_checkpointing=False)
    mx.eval(logits_full, logits_step)

    diff = float(mx.max(mx.abs(logits_full[:, -1:, :] - logits_step)).item())
    assert diff < 1e-4


if __name__ == "__main__":
    test_param_accounting_exact()
    test_end_to_end_fused_vs_unfused_parity()
    test_q_chunked_sdpa_exact_parity()
    test_chunked_ce_and_kv_cache_parity()
    print("All model, SDPA, chunked-CE, and KV-cache tests PASSED!")
