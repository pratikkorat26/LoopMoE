"""Training & Benchmarking script for the Looped MoE Transformer (Apple MLX).

Features:
- Zero-bottleneck pre-tokenized uint16 memory-mapped dataset (TinyStories slice)
- Mixed-precision bfloat16 Metal execution with float32 master weights
- Hybrid Muon (2D hidden matrices) + AdamW (embeddings, 3D experts, routers, norms)
- Linear warmup + Cosine decay LR schedule + gradient clipping + grad accumulation
- DeepSeek-V3 auxiliary-loss-free dynamic router bias updates
- Per-loop-iteration expert utilization diagnostics, peak unified memory, & tokens/sec
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
import numpy as np

from loopmoe.config import ModelConfig, TrainConfig
from loopmoe.data import MemmapDataset, prepare_dataset
from loopmoe.model import LoopedMoETransformer


# ---------------------------------------------------------------------------
# Optimizer & LR Schedule
# ---------------------------------------------------------------------------

def build_optimizer(tcfg: TrainConfig) -> optim.MultiOptimizer:
    """Hybrid Muon (2D hidden projections) + AdamW (embeddings, 3D experts, routers, 1D)."""
    decay_steps = max(1, tcfg.max_steps - tcfg.warmup_steps)

    def make_schedule(peak_lr: float):
        warmup = optim.linear_schedule(1e-6, peak_lr, tcfg.warmup_steps)
        cosine = optim.cosine_decay(peak_lr, decay_steps, end=peak_lr * tcfg.min_lr_ratio)
        return optim.join_schedules([warmup, cosine], [tcfg.warmup_steps])

    muon_opt = optim.Muon(
        learning_rate=make_schedule(tcfg.muon_lr),
        momentum=0.95,
        weight_decay=tcfg.weight_decay,
        nesterov=True,
        ns_steps=5,
    )
    adamw_opt = optim.AdamW(
        learning_rate=make_schedule(tcfg.adamw_lr),
        betas=[0.9, 0.95],
        eps=1e-8,
        weight_decay=tcfg.weight_decay,
        bias_correction=True,
    )

    def is_muon_weight(path: str, param: mx.array) -> bool:
        if param.ndim != 2:
            return False
        excluded = ("tok_emb", "lm_head", "step_embeddings", "router")
        return not any(tok in path for tok in excluded)

    return optim.MultiOptimizer(optimizers=[muon_opt, adamw_opt], filters=[is_muon_weight])


# ---------------------------------------------------------------------------
# Diagnostics & Evaluation
# ---------------------------------------------------------------------------

def format_expert_utilization(iter_counts: np.ndarray) -> str:
    """Format per-iteration expert utilization (%) across all core layers."""
    # iter_counts: (R, n_core_layers, E)
    R, L, E = iter_counts.shape
    lines = []
    # Aggregate across all loops & layers
    total_per_expert = iter_counts.sum(axis=(0, 1))
    total_all = max(1.0, total_per_expert.sum())
    pct_all = 100.0 * total_per_expert / total_all
    dead = int((total_per_expert == 0).sum())
    ideal = 100.0 / E
    cv = float(np.std(pct_all) / (np.mean(pct_all) + 1e-9))
    lines.append(
        f"  Overall ({E} experts, ideal {ideal:.1f}%, CV={cv:.3f}, dead={dead}): "
        + " ".join(f"E{e}:{pct_all[e]:4.1f}%" for e in range(E))
    )
    # Per loop iteration (averaged across core layers)
    for r in range(R):
        r_counts = iter_counts[r].sum(axis=0)
        r_pct = 100.0 * r_counts / max(1.0, r_counts.sum())
        lines.append(
            f"  Loop r={r}: " + " ".join(f"E{e}:{r_pct[e]:4.1f}%" for e in range(E))
        )
    return "\n".join(lines)


def evaluate(
    model: LoopedMoETransformer,
    val_ds: MemmapDataset,
    batch_size: int,
    seq_len: int,
    num_batches: int,
    compute_dtype: mx.Dtype,
    num_loops: Optional[int] = None,
) -> float:
    losses = []
    for _ in range(num_batches):
        x, y = val_ds.get_batch(batch_size, seq_len)
        loss, _ = model.compute_loss(
            x, y, num_loops=num_loops, compute_dtype=compute_dtype, use_checkpointing=False
        )
        mx.eval(loss)
        losses.append(float(loss.item()))
    return float(np.mean(losses))


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------

def train(mcfg: ModelConfig, tcfg: TrainConfig) -> None:
    mx.random.seed(tcfg.seed)
    np.random.seed(tcfg.seed)

    compute_dtype = mx.bfloat16 if tcfg.dtype == "bfloat16" else mx.float16

    # 1. Prepare memmap data
    train_path, val_path = prepare_dataset(
        tcfg.data_dir,
        dataset_key=tcfg.dataset_name,
        num_train_docs=tcfg.num_train_stories,
        num_val_docs=tcfg.num_val_stories,
    )
    train_ds = MemmapDataset(train_path, seed=tcfg.seed)
    val_ds = MemmapDataset(val_path, seed=tcfg.seed + 1)

    # 2. Build model & print exact parameter accounting
    model = LoopedMoETransformer(mcfg)
    mx.eval(model.parameters())

    actual_unique = sum(p.size for _, p in tree_flatten(model.parameters()))
    trainable_unique = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    breakdown = mcfg.param_breakdown()

    dev_info = mx.metal.device_info()
    print("=" * 72)
    print(f"Device: {dev_info['device_name']} | "
          f"Unified RAM: {dev_info['memory_size'] / 1e9:.1f} GB | "
          f"Max Working Set: {dev_info['max_recommended_working_set_size'] / 1e9:.2f} GB")
    print(f"Looped MoE Transformer (R={mcfg.num_loops}, effective depth={mcfg.effective_depth} layers)")
    print(f"  Unique params (actual / analytical):   {actual_unique:,} / {breakdown['unique_total']:,}")
    print(f"  Trainable params:                      {trainable_unique:,}")
    print(f"  Effective-depth params (unrolled R={mcfg.num_loops}): {breakdown['effective_depth_total']:,}")
    print(f"  Active non-emb compute params / token: {breakdown['active_non_emb_compute_params']:,}")
    print(f"  Precision: {tcfg.dtype} compute / float32 master weights | "
          f"Fused Metal: {mcfg.use_fused_kernels} | Checkpointing: {mcfg.use_checkpointing}")
    print(f"  Batch: {tcfg.batch_size} x {tcfg.grad_accum_steps} accum x {tcfg.seq_len} seq = "
          f"{tcfg.batch_size * tcfg.grad_accum_steps * tcfg.seq_len:,} tokens/step")
    print("=" * 72)

    # 3. Optimizer
    optimizer = build_optimizer(tcfg)
    optimizer.init(model.trainable_parameters())

    # 4. Loss + value_and_grad (exact chunked cross-entropy + mx.compile when static)
    def loss_fn(model: LoopedMoETransformer, x: mx.array, y: mx.array, num_loops: int):
        loss, iter_counts = model.compute_loss(
            x,
            y,
            num_loops=num_loops,
            compute_dtype=compute_dtype,
            use_checkpointing=mcfg.use_checkpointing,
        )
        return loss, mx.stop_gradient(iter_counts)

    val_and_grad = nn.value_and_grad(model, loss_fn)

    # Note: disable mx.compile when seq_len > 2048 so Q-chunked SDPA + chunked CE
    # checkpoint boundaries stay memory-optimal without graph-level buffer hoisting.
    if not tcfg.random_loops and tcfg.seq_len <= 2048:
        static_loops = mcfg.num_loops

        def _micro_step(x: mx.array, y: mx.array):
            return val_and_grad(model, x, y, static_loops)

        micro_step = mx.compile(_micro_step, inputs=model.state, outputs=model.state)
    else:
        def micro_step(x: mx.array, y: mx.array):
            return val_and_grad(model, x, y, cur_loops)

    # Reset peak memory before training benchmark
    mx.reset_peak_memory()

    tokens_per_step = tcfg.batch_size * tcfg.grad_accum_steps * tcfg.seq_len
    total_tokens = 0
    step_times = []
    window_counts = np.zeros(
        (mcfg.num_loops, mcfg.n_core_layers, mcfg.n_routed_experts), dtype=np.float64
    )
    cumulative_counts = np.zeros_like(window_counts)

    # Initial val loss
    val_loss_0 = evaluate(
        model, val_ds, tcfg.batch_size, tcfg.seq_len, tcfg.eval_batches, compute_dtype
    )
    print(f"[step   0] val_loss={val_loss_0:.4f} (ppl={np.exp(min(val_loss_0, 20)):.1f})")

    train_start = time.perf_counter()

    for step in range(1, tcfg.max_steps + 1):
        t_step = time.perf_counter()

        # Optionally randomize loop depth during training for test-time scaling robustness
        if tcfg.random_loops:
            cur_loops = int(np.random.randint(tcfg.min_train_loops, tcfg.max_train_loops + 1))
        else:
            cur_loops = mcfg.num_loops

        accum_grads = None
        step_loss = 0.0
        step_counts = None

        for _ in range(tcfg.grad_accum_steps):
            x, y = train_ds.get_batch(tcfg.batch_size, tcfg.seq_len)
            (loss, iter_counts), grads = micro_step(x, y)

            if accum_grads is None:
                accum_grads = grads
                step_counts = iter_counts
            else:
                accum_grads = tree_map(lambda a, b: a + b, accum_grads, grads)
                step_counts = step_counts + iter_counts

            # Materialize micro-batch graph so peak memory stays bounded by 1 micro-batch
            mx.eval(loss, accum_grads, step_counts)
            step_loss += float(loss.item())

        # Average gradients across accumulation steps
        inv_accum = 1.0 / tcfg.grad_accum_steps
        accum_grads = tree_map(lambda g: g * inv_accum, accum_grads)
        step_loss *= inv_accum

        # Clip gradient norm & apply Muon + AdamW update
        accum_grads, grad_norm = optim.clip_grad_norm(accum_grads, max_norm=tcfg.grad_clip)
        optimizer.update(model, accum_grads)

        # DeepSeek-V3 auxiliary-loss-free dynamic router bias update
        model.update_router_biases(step_counts)

        # Evaluate updated model & optimizer state
        mx.eval(model.parameters(), optimizer.state, grad_norm)

        dt = time.perf_counter() - t_step
        total_tokens += tokens_per_step
        # Skip step 1 warmup from steady-state throughput stats
        if step > 1:
            step_times.append(dt)

        if cur_loops == mcfg.num_loops:
            c_np = np.array(step_counts)
            window_counts += c_np
            cumulative_counts += c_np

        if step % tcfg.log_interval == 0 or step == 1 or step == tcfg.max_steps:
            tok_s = tokens_per_step / dt
            peak_gb = mx.get_peak_memory() / (1024 ** 3)
            active_gb = mx.get_active_memory() / (1024 ** 3)
            print(
                f"[step {step:3d}/{tcfg.max_steps}] "
                f"loss={step_loss:.4f} | gnorm={float(grad_norm.item()):.2f} | "
                f"R={cur_loops} | {tok_s:7.0f} tok/s ({dt*1000:5.0f} ms) | "
                f"mem={active_gb:.2f}/{peak_gb:.2f} GB"
            )

        if step % tcfg.eval_interval == 0 or step == tcfg.max_steps:
            val_loss = evaluate(
                model, val_ds, tcfg.batch_size, tcfg.seq_len, tcfg.eval_batches, compute_dtype
            )
            print(f"  --> [eval @ step {step}] val_loss={val_loss:.4f} (ppl={np.exp(min(val_loss, 20)):.1f})")
            if window_counts.sum() > 0:
                print(format_expert_utilization(window_counts))
                window_counts[:] = 0.0

    total_time = time.perf_counter() - train_start
    steady_tok_s = tokens_per_step / float(np.mean(step_times)) if step_times else 0.0
    peak_gb = mx.get_peak_memory() / (1024 ** 3)

    print("=" * 72)
    print("BENCHMARK SUMMARY")
    print(f"  Total steps:          {tcfg.max_steps} ({total_tokens:,} tokens in {total_time:.1f}s)")
    print(f"  Steady throughput:    {steady_tok_s:,.0f} tokens/sec")
    print(f"  Peak unified memory:  {peak_gb:.2f} GB")
    print("  Cumulative Expert Routing Utilization:")
    print(format_expert_utilization(cumulative_counts))
    print("=" * 72)

    # Save checkpoint & config for generation
    save_dir = Path(tcfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "looped_moe.safetensors"
    model.save_weights(str(ckpt_path))
    with open(save_dir / "model_config.json", "w") as f:
        json.dump(mcfg.__dict__, f, indent=2)
    print(f"[save] Saved weights to {ckpt_path}")


def parse_args() -> Tuple[ModelConfig, TrainConfig]:
    parser = argparse.ArgumentParser(description="Train Long-Context Looped MoE Transformer (MLX)")
    parser.add_argument("--dataset", type=str, default="tinystories", choices=["tinystories", "fineweb"])
    parser.add_argument("--train_docs", type=int, default=25000)
    parser.add_argument("--val_docs", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--loops", type=int, default=4)
    parser.add_argument("--random_loops", action="store_true")
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--no_fused", action="store_true")
    parser.add_argument("--no_swa", action="store_true", help="Disable hybrid SWA (full global everywhere)")
    parser.add_argument("--window_size", type=int, default=2048)
    parser.add_argument("--sdpa_chunk", type=int, default=512)
    parser.add_argument("--ce_chunk", type=int, default=1024)
    parser.add_argument("--bptt_steps", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--eval_batches", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    # Auto-enable per-loop activation checkpointing for long contexts (>= 2048)
    auto_ckpt = args.checkpoint or (args.seq_len >= 2048)
    mcfg = ModelConfig(
        num_loops=args.loops,
        max_seq_len=max(16384, args.seq_len),
        use_fused_kernels=not args.no_fused,
        use_checkpointing=auto_ckpt,
        sliding_window=None if args.no_swa else args.window_size,
        sdpa_chunk_size=args.sdpa_chunk,
        ce_chunk_size=args.ce_chunk,
        bptt_steps=args.bptt_steps,
    )
    tcfg = TrainConfig(
        dataset_name=args.dataset,
        num_train_stories=args.train_docs,
        num_val_stories=args.val_docs,
        max_steps=args.steps,
        warmup_steps=min(25, max(5, args.steps // 10)),
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        seq_len=args.seq_len,
        random_loops=args.random_loops,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        save_dir=args.save_dir,
    )
    return mcfg, tcfg


def main() -> None:
    mcfg, tcfg = parse_args()
    train(mcfg, tcfg)


if __name__ == "__main__":
    main()
