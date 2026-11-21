"""KV-cached autoregressive text generation with test-time loop-depth scaling."""

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional

import mlx.core as mx
import tiktoken

from config import ModelConfig
from model import LoopedMoETransformer


def sample_next_token(
    logits: mx.array,
    temperature: float = 0.8,
    top_k: int = 40,
) -> mx.array:
    """Temperature + Top-K sampling from next-token logits (1, V)."""
    if temperature <= 0.0:
        return mx.argmax(logits, axis=-1, keepdims=True)

    logits = logits / temperature
    if top_k > 0 and top_k < logits.shape[-1]:
        # mx.topk is not guaranteed sorted, so take min(top_vals) as the threshold
        top_vals = mx.topk(logits, k=top_k, axis=-1)
        min_val = mx.min(top_vals, axis=-1, keepdims=True)
        logits = mx.where(logits < min_val, -1e9, logits)

    return mx.random.categorical(logits)[:, None]


def generate(
    model: LoopedMoETransformer,
    enc: tiktoken.Encoding,
    prompt: str,
    max_new_tokens: int = 80,
    num_loops: Optional[int] = None,
    temperature: float = 0.8,
    top_k: int = 40,
    seed: int = 42,
) -> tuple[str, float]:
    """Generate text with KV caching at a chosen recurrent loop count R."""
    mx.random.seed(seed)
    R = num_loops if num_loops is not None else model.cfg.num_loops
    prompt_ids = enc.encode_ordinary(prompt)
    if not prompt_ids:
        prompt_ids = [enc.eot_token]

    cache = model.make_cache(num_loops=R)

    # Prefill prompt
    x = mx.array([prompt_ids], dtype=mx.int32)
    logits, _, cache = model(
        x, num_loops=R, compute_dtype=mx.bfloat16, cache=cache, use_checkpointing=False
    )
    next_tok = sample_next_token(logits[:, -1, :], temperature=temperature, top_k=top_k)
    mx.eval(next_tok, cache)

    generated = [int(next_tok.item())]
    t0 = time.perf_counter()

    for _ in range(max_new_tokens - 1):
        logits, _, cache = model(
            next_tok, num_loops=R, compute_dtype=mx.bfloat16, cache=cache, use_checkpointing=False
        )
        next_tok = sample_next_token(logits[:, -1, :], temperature=temperature, top_k=top_k)
        mx.eval(next_tok)
        tok_id = int(next_tok.item())
        if tok_id == enc.eot_token:
            break
        generated.append(tok_id)

    dt = time.perf_counter() - t0
    decode_tok_s = max(1, len(generated) - 1) / max(dt, 1e-6)
    text = enc.decode(prompt_ids + generated)
    return text, decode_tok_s


def main():
    parser = argparse.ArgumentParser(description="Generate with Looped MoE Transformer (MLX)")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--prompt", type=str, default="Once upon a time, a little dog named Max")
    parser.add_argument("--max_tokens", type=int, default=80)
    parser.add_argument("--loops", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    with open(ckpt_dir / "model_config.json") as f:
        mcfg = ModelConfig(**json.load(f))

    model = LoopedMoETransformer(mcfg)
    model.load_weights(str(ckpt_dir / "looped_moe.safetensors"))
    mx.eval(model.parameters())

    enc = tiktoken.get_encoding("gpt2")

    print(f"Prompt: {args.prompt!r}")
    print("=" * 72)
    for R in args.loops:
        text, tok_s = generate(
            model,
            enc,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            num_loops=R,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed,
        )
        eff_depth = mcfg.n_prelude_layers + R * mcfg.n_core_layers + mcfg.n_coda_layers
        print(f"[R={R} loops | effective depth={eff_depth} layers | {tok_s:.1f} decode tok/s]")
        print(text)
        print("-" * 72)


if __name__ == "__main__":
    main()
