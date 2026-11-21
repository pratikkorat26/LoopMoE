# LoopMoE

**LoopMoE** is a custom **Looped (Recurrent-Depth) Mixture-of-Experts Transformer** built from scratch in **Apple MLX** with custom Metal fused kernels, designed for high-throughput training and long-context web search & crawling on Apple Silicon (16 GB Unified Memory).

## Highlights

- **Looped / Recurrent-Depth + Sparse MoE Topology:** `Prelude (1 dense block)` → `Recurrent Core (2 shared MoE blocks looped R times)` → `Coda (1 dense block)` → `Tied LM Head`. Achieves **96.8M effective-depth parameters** (10 layers at `R=4`) from only **47.7M unique trainable parameters**, with **33.3M active non-embedding compute parameters/token**.
- **Iteration-Aware MoE Routing:** DeepSeek-V3 style fine-grained routed experts (`8` experts, `top_k=2`) + always-active shared expert, conditioned on both token hidden state and loop-step embedding (`x @ W_gate + step_emb @ W_step`), balanced with **auxiliary-loss-free dynamic bias updates**.
- **Test-Time Compute Scaling:** Adjust loop count `R` at inference time (`R=2, 4, 6, 8...`) on the same checkpoint with KV caching.
- **Custom Apple Metal Fused Kernels (`mx.fast.metal_kernel`):**
  - `fused_swiglu` (forward + custom Metal backward VJP) — **1.57× faster**
  - `fused_adaln_rmsnorm` (single-pass SIMD-reduced RMSNorm + AdaLN-Zero + VJP) — **1.28× faster**
  - `fused_router_topk_sigmoid` (register-resident Top-K + unbiased normalization + VJP) — **2.04× faster**
  - Verified end-to-end full-model parity: `loss_diff = 0.00e+00`, `routing_exact = True`, `max_grad_diff = 3.13e-07`.
- **Long-Context Web-Crawling Stack (up to 16k+ training, 32k–64k YaRN extrapolation):**
  - **YaRN High-Base RoPE** (`rope_theta = 500,000`, smooth frequency ramp + attention `mscale`)
  - **Learned Attention Sinks** inside MLX's fused Metal SDPA
  - **Hybrid Sliding-Window (`W=2048`) + Global Attention** (with SWA KV-cache trimming at decode)
  - **Exact Q-Chunked Checkpointed Causal SDPA** (`0.00e+00` forward / `1.19e-07` backward parity; 4.4× less memory & 5.4× faster at 16k)
  - **Exact Chunked Checkpointed Cross-Entropy** (reduces 16k LM-head memory from ~9.9 GB → ~0.4 GB)
- **Modern Optimization:** Hybrid **Muon** (2D hidden projections) + **AdamW** (embeddings, 3D routed experts, routers, norms) via `MultiOptimizer`, `bfloat16` compute / `float32` master weights, cosine decay with linear warmup, and zero-bottleneck `uint16` `np.memmap` dataloaders for **TinyStories** and **FineWeb-Edu**.

## Repository Structure

| File | Description |
|---|---|
| `config.py` | `ModelConfig`, `TrainConfig`, and exact analytical parameter accounting |
| `kernels.py` | Custom Metal fused kernels (`SwiGLU`, `AdaLN+RMSNorm`, `Router Top-K`) + forward/backward parity suite |
| `moe.py` | `IterationAwareRouter`, `SharedExpert`, and `mx.gather_mm` sparse `MoELayer` |
| `model.py` | `GQAAttention` (QK-Norm, YaRN RoPE, Sinks, Hybrid SWA, Q-Chunked SDPA), `RecurrentCore`, `LoopedMoETransformer` |
| `train.py` | Memmap dataset preparation (`tinystories` / `fineweb`), hybrid `Muon`+`AdamW` training loop, router telemetry |
| `generate.py` | KV-cached autoregressive text generation with test-time loop scaling (`--loops 2 4 6`) |

## Benchmarks (Apple M4 — 8-core GPU, 16 GB Unified Memory)

| Benchmark | Config | Throughput | Peak Unified RAM | Validation Loss |
|---|---|---|---|---|
| **TinyStories (200 steps)** | `seq_len=512`, `B=8×2`, `R=4` | **4,261 tok/s** (peak 4,689) | **5.26 GB** | `10.90` → **`2.94`** (`ppl 18.9`) |
| **FineWeb-Edu 16k (5 steps)** | `seq_len=16384`, `B=1`, `R=4` | **306–353 tok/s** | `19.38 GB` | `10.94` → **`9.37`** |
| **KV-Cached Decode** | `R=2 / R=4 / R=6` | **102.3 / 82.1 / 61.3 tok/s** | `< 1.0 GB` | — |

## Quickstart

```bash
# 1. Verify custom Metal kernel forward & backward parity + micro-benchmarks
python kernels.py

# 2. Train on TinyStories (fast sanity check)
python train.py --dataset tinystories --steps 200 --batch_size 8 --grad_accum 2 --seq_len 512 --loops 4

# 3. Train on FineWeb-Edu (crawled web pages) with long context
python train.py --dataset fineweb --train_docs 25000 --seq_len 2048 --batch_size 2 --grad_accum 4 --steps 2000

# 4. Generate text comparing multiple test-time loop counts
python generate.py --ckpt_dir checkpoints --loops 2 4 6 --prompt "Once upon a time, a little dog named Max"
```
