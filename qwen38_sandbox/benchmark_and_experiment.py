"""
benchmark_and_experiment.py — BF16 vs. Local 8-Bit & Edit-Refeed Harness for Qwen3.8-27B

Security & Supply-Chain Constraints (STRICT):
  1. NEVER pass `trust_remote_code=True` to any Hugging Face / `transformers` / `mlx_lm` call.
  2. Only pull `.safetensors`, `.gguf`, and `.json` files via `huggingface_hub.snapshot_download`
     with explicit `allow_patterns`.
  3. Support loading the base model (`mlx-community/Qwen3.8-27B-bf16` or `Qwen/Qwen3.8-27B`)
     and generating a local 8-bit quantized variant (`./models/Qwen3.8-27B-8bit-local`)
     converted directly from our local BF16 checkpoint via
     `mlx_lm.convert -q --q-bits 8 --q-group-size 64`.

Hardware & MLX Tuning Constraints:
  - Do NOT set or increase `MLX_METAL_COMMAND_BUFFER_MB` (keep stock defaults).
  - Set prefill chunk size (`prefill_step_size`) to `512` or `1024`.
  - Explicitly call `mx.eval(cache.state)` on the full Gated DeltaNet `ArraysCache` state tuple.
  - Force a 2-token dummy warmup pass (`mx.eval(logits)` + `mx.synchronize()`) before timing
    so SSD page-in and Metal/GatedDeltaNet shader compilation are separated from prefill throughput.
"""

from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download
from mlx_lm.convert import convert as mlx_lm_convert
from mlx_lm.utils import load as mlx_lm_load

from attention_probe import (
    FULL_ATTENTION_LAYERS_DEFAULT,
    ROUTING_HIGHLIGHT_LAYERS,
    HeadShiftMetric,
    HeadTokenRoutingAttribution,
    attribute_top_heads_token_routing,
    capture_attention_maps,
    compute_attention_shift,
)
from hybrid_cache import (
    DEFAULT_PREFILL_STEP_SIZE,
    HybridTimelineCache,
    eval_hybrid_cache_state,
    fork_and_edit,
)
from subspace_projector import (
    HIDDEN_SIZE_DEFAULT,
    NUM_LAYERS_DEFAULT,
    SubspaceProjector,
    WeightSpaceProjector,
    load_subspace_vectors,
    resolve_decoder_layers,
    unwrap_model,
)


SAFE_ALLOW_PATTERNS: List[str] = [
    "*.safetensors",
    "*.gguf",
    "*.json",
]

DEFAULT_BASE_MODEL = "mlx-community/Qwen3.8-27B-bf16"
DEFAULT_8BIT_LOCAL_PATH = "./models/Qwen3.8-27B-8bit-local"
DEFAULT_HEATMAP_PATH = "./outputs/attention_jsd_heatmap.png"
DEFAULT_TOKEN_ROUTING_PLOT_PATH = "./outputs/top_heads_token_routing.png"
DEFAULT_THINK_2X2_PLOT_PATH = "./outputs/think_2x2_lookback_mass.png"
DEFAULT_ACTIVE_LAYERS: Tuple[int, ...] = tuple(range(10, 59))  # Layers 10..58 inclusive
DEFAULT_BENCH_PREFILL_LENGTHS: Tuple[int, ...] = (64, 512, 2048)

LOOKBACK_HEADS_5: Tuple[Tuple[int, int], ...] = (
    (35, 21),
    (39, 1),
    (43, 7),
    (43, 1),
    (39, 0),
)

LAYER_BAND_ABLATION_CONFIGS: List[Tuple[str, List[int]]] = [
    ("Full GLP-49 Band (10..58)", list(range(10, 59))),
    ("Early-Mid Band Only (10..29)", list(range(10, 30))),
    ("Mid-Late Lookback Band Only (30..47)", list(range(30, 48))),
    ("Sparse Full-Attention Only ({31,35,39,43,47})", [31, 35, 39, 43, 47]),
]

COMPONENT_WRITER_ABLATION_CONFIGS: List[Tuple[str, Set[str]]] = [
    ("Weight-Space All Writers (10..58)", {"deltanet", "full_attn", "mlp"}),
    ("Weight-Space MLP Writers Only (10..58)", {"mlp"}),
    ("Weight-Space Gated DeltaNet + MLP Only (10..58)", {"deltanet", "mlp"}),
    ("Weight-Space Full-Attention Writers Only (10..58)", {"full_attn"}),
]


def enforce_metal_command_buffer_invariant() -> None:
    """
    Hardware constraint: Do NOT set or increase MLX_METAL_COMMAND_BUFFER_MB
    (keep stock defaults for maximum M4 Max prefill throughput).
    """
    if "MLX_METAL_COMMAND_BUFFER_MB" in os.environ:
        os.environ.pop("MLX_METAL_COMMAND_BUFFER_MB", None)


def warmup_model_and_kernels(model: nn.Module) -> float:
    """
    Run a 2-token dummy warmup pass with a throwaway `HybridTimelineCache` and call
    `mx.eval(logits)` + `mx.synchronize()` to force all model weights (e.g. 55.6 GB BF16
    or 29.5 GB 8-bit) into wired unified memory and JIT-compile all Metal / GatedDeltaNet
    and `@mx.compile` kernels BEFORE starting any `time.perf_counter()` timers.

    Returns:
        Warmup / page-in latency in milliseconds.
    """
    enforce_metal_command_buffer_invariant()
    if hasattr(model, "eval"):
        model.eval()
    resolve_decoder_layers(model)

    t0 = time.perf_counter()
    dummy_timeline = HybridTimelineCache(model=model, prefill_step_size=512)
    dummy_tokens = mx.array([[1, 2]], dtype=mx.int32)
    logits = model(dummy_tokens, cache=dummy_timeline.cache)
    mx.eval(logits)
    eval_hybrid_cache_state(dummy_timeline.cache)
    mx.synchronize()
    warmup_ms = (time.perf_counter() - t0) * 1000.0

    del dummy_timeline
    del logits
    return warmup_ms


def safe_download_repository(
    repo_id_or_path: Union[str, Path],
    local_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Resolve a local directory or safely pull ONLY `.safetensors`, `.gguf`, and `.json`
    artifacts from Hugging Face Hub using `snapshot_download` with strict `allow_patterns`.
    Never executes remote code.
    """
    candidate = Path(repo_id_or_path)
    if candidate.exists():
        return candidate.resolve()

    kwargs: Dict[str, Any] = {
        "repo_id": str(repo_id_or_path),
        "allow_patterns": list(SAFE_ALLOW_PATTERNS),
    }
    if local_dir is not None:
        kwargs["local_dir"] = str(local_dir)

    downloaded_path = snapshot_download(**kwargs)
    return Path(downloaded_path).resolve()


def safe_load_model_and_tokenizer(
    model_path_or_repo: Union[str, Path],
    lazy: bool = False,
) -> Tuple[nn.Module, Any, Dict[str, Any]]:
    """
    Safely load an MLX model and tokenizer from a local directory or HF repo
    after enforcing `snapshot_download(..., allow_patterns=["*.safetensors", "*.gguf", "*.json"])`
    and `trust_remote_code=False`.
    """
    enforce_metal_command_buffer_invariant()
    local_model_dir = safe_download_repository(model_path_or_repo)

    model, tokenizer, config = mlx_lm_load(
        str(local_model_dir),
        tokenizer_config={"trust_remote_code": False},
        lazy=lazy,
        return_config=True,
        trust_remote_code=False,
    )
    if hasattr(model, "eval"):
        model.eval()
    resolve_decoder_layers(model)
    return model, tokenizer, config


def convert_bf16_to_local_8bit(
    source_model_path: Union[str, Path] = DEFAULT_BASE_MODEL,
    output_8bit_path: Union[str, Path] = DEFAULT_8BIT_LOCAL_PATH,
    q_bits: int = 8,
    q_group_size: int = 64,
    use_cli: bool = False,
) -> Path:
    """
    Convert a local BF16 Qwen3.8-27B checkpoint directly into a local 8-bit quantized
    MLX checkpoint (`./models/Qwen3.8-27B-8bit-local`) with `q_bits=8, q_group_size=64`
    and `trust_remote_code=False`.
    """
    enforce_metal_command_buffer_invariant()
    local_src = safe_download_repository(source_model_path)
    target_dir = Path(output_8bit_path)
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    if use_cli:
        cmd = [
            sys.executable,
            "-m",
            "mlx_lm.convert",
            "--hf-path",
            str(local_src),
            "--mlx-path",
            str(target_dir),
            "-q",
            "--q-bits",
            str(q_bits),
            "--q-group-size",
            str(q_group_size),
        ]
        subprocess.run(cmd, check=True)
    else:
        mlx_lm_convert(
            hf_path=str(local_src),
            mlx_path=str(target_dir),
            quantize=True,
            q_bits=q_bits,
            q_group_size=q_group_size,
            trust_remote_code=False,
        )

    return target_dir.resolve()


def compute_logit_kl_divergence(
    logits_ref: mx.array,
    logits_cand: mx.array,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Compute token-wise KL(P_ref || P_cand) and symmetric JSD between two logit tensors
    of shape `(B, S, V)` in FP32.
    """
    r = logits_ref.astype(mx.float32)
    c = logits_cand.astype(mx.float32)

    s_len = min(int(r.shape[1]), int(c.shape[1]))
    r = r[:, -s_len:, :]
    c = c[:, -s_len:, :]

    p = mx.softmax(r, axis=-1, precise=True)
    q = mx.softmax(c, axis=-1, precise=True)
    m = 0.5 * (p + q)

    p_safe = mx.maximum(p, eps)
    q_safe = mx.maximum(q, eps)
    m_safe = mx.maximum(m, eps)

    kl_pq = mx.sum(p * (mx.log(p_safe) - mx.log(q_safe)), axis=-1)
    kl_pm = mx.sum(p * (mx.log(p_safe) - mx.log(m_safe)), axis=-1)
    kl_qm = mx.sum(q * (mx.log(q_safe) - mx.log(m_safe)), axis=-1)
    jsd = mx.maximum(0.5 * (kl_pm + kl_qm), 0.0)

    kl_mean = mx.mean(kl_pq)
    kl_max = mx.max(kl_pq)
    jsd_mean = mx.mean(jsd)
    top1_agree = mx.mean(
        (mx.argmax(r, axis=-1) == mx.argmax(c, axis=-1)).astype(mx.float32)
    )
    mx.eval(kl_mean, kl_max, jsd_mean, top1_agree)

    return {
        "kl_mean_nats": float(kl_mean.item()),
        "kl_max_nats": float(kl_max.item()),
        "jsd_mean_nats": float(jsd_mean.item()),
        "top1_agreement": float(top1_agree.item()),
    }


def _build_synthetic_benchmark_tokens(
    base_tokens: Sequence[int],
    target_length: int,
) -> List[int]:
    """Tile `base_tokens` deterministically to construct a prompt of exact `target_length`."""
    seed = list(base_tokens) if len(base_tokens) > 0 else [10, 20, 30, 40]
    reps = (target_length + len(seed) - 1) // len(seed)
    return (seed * reps)[:target_length]


def _measure_cold_prefill_across_lengths(
    model: nn.Module,
    base_tokens: Sequence[int],
    prefill_lengths: Sequence[int] = DEFAULT_BENCH_PREFILL_LENGTHS,
    prefill_step_size: int = 2048,
) -> Dict[int, Dict[str, float]]:
    """
    Measure pure M4 Max compute prefill throughput (tok/s and ms) across `prefill_lengths`
    (e.g., 64, 512, 2048 tokens) using a fresh `HybridTimelineCache` for each length
    with `prefill_step_size=2048` so a 512 or 2048 token prefill executes in a single
    dense Metal GEMM sweep without intermediate `eval_all()` stalls, calling `mx.eval(logits)`
    once at the end of the prefill.
    """
    length_stats: Dict[int, Dict[str, float]] = {}
    for seq_len in prefill_lengths:
        synth_tokens = _build_synthetic_benchmark_tokens(base_tokens, int(seq_len))
        fresh_timeline = HybridTimelineCache(
            model=model, prefill_step_size=2048
        )
        tokens_arr = mx.array(synth_tokens, dtype=mx.int32)[None, :]
        mx.eval(tokens_arr)
        mx.synchronize()
        t0 = time.perf_counter()
        logits = model(tokens_arr, cache=fresh_timeline.cache)
        mx.eval(logits)
        mx.synchronize()
        elapsed_s = max(time.perf_counter() - t0, 1e-9)
        length_stats[int(seq_len)] = {
            "tok_s": float(seq_len) / elapsed_s,
            "ms": elapsed_s * 1000.0,
        }
        del fresh_timeline
        del logits
    return length_stats


def _measure_single_stream_decode(
    model: nn.Module,
    timeline: HybridTimelineCache,
    seed_token: mx.array,
    num_decode_steps: int = 32,
) -> Tuple[float, List[int]]:
    """
    Measure single-stream autoregressive decode throughput (tok/s) using async pipelining:
      - Calls `mx.async_eval(next_tok)` on each step so the CPU queues token `t+1` while
        the GPU executes token `t`.
      - Calls `timeline.eval_all()` only once every 32 steps and at the very end before
        stopping the timer.
    """
    generated_tokens: List[mx.array] = []
    cur = seed_token.reshape(1, 1)

    mx.synchronize()
    t0 = time.perf_counter()
    for step_idx in range(1, num_decode_steps + 1):
        logits = model(cur, cache=timeline.cache)
        next_tok = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        mx.async_eval(next_tok)
        generated_tokens.append(next_tok)
        cur = next_tok
        if step_idx % 32 == 0 and step_idx < num_decode_steps:
            timeline.eval_all()

    if generated_tokens:
        mx.eval(*generated_tokens)
    timeline.eval_all()
    mx.synchronize()
    elapsed = max(time.perf_counter() - t0, 1e-9)
    generated = [int(t.item()) for t in generated_tokens]
    return num_decode_steps / elapsed, generated


def _measure_mlx_lm_generate_step_decode(
    model: nn.Module,
    prompt_tokens: Sequence[int],
    num_decode_steps: int = 16,
) -> Optional[float]:
    """
    Measure native `mlx_lm.generate.generate_step` decode throughput (tok/s) on an
    un-wrapped model after prompt prefill to benchmark the stock `mlx_lm` bandwidth ceiling.
    """
    try:
        from mlx_lm.generate import generate_step
    except ImportError:
        return None

    prompt_arr = mx.array(list(prompt_tokens), dtype=mx.int32)
    gen = generate_step(prompt_arr, model, max_tokens=num_decode_steps + 1)
    try:
        # Advance first token (which includes prompt prefill) before starting decode timer
        next(gen)
        mx.synchronize()
        t0 = time.perf_counter()
        count = 0
        for _ in range(num_decode_steps):
            next(gen)
            count += 1
        mx.synchronize()
        elapsed = max(time.perf_counter() - t0, 1e-9)
        return count / elapsed if count > 0 else None
    except Exception:
        return None


def _benchmark_weight_space_pass(
    model: nn.Module,
    vectors: Dict[int, mx.array],
    active_layers: Sequence[int],
    prompt_tokens: Sequence[int],
    edit_token_idx: int,
    edited_suffix_tokens: Sequence[int],
    prefill_lengths: Sequence[int],
    prefill_step_size: int,
    decode_steps: int,
) -> Dict[str, Any]:
    """
    Unwrap `model` to 100% stock `mlx_lm` layers, apply `WeightSpaceProjector.apply_in_place`
    across `active_layers`, and measure:
      - `apply_in_place_ms` and `restore_baseline_ms`
      - Cold prefill tok/s at `prefill_lengths` (`64`, `512`, `2048`)
      - Warm edit-refeed latency (`fork_and_edit`, ms)
      - Single-stream pipelined decode tok/s and native `mlx_lm.generate_step` decode tok/s
    """
    unwrap_model(model)
    ws_proj = WeightSpaceProjector(model=model, vectors=vectors)

    mx.synchronize()
    t_apply0 = time.perf_counter()
    num_matrices = ws_proj.apply_in_place(
        alpha=1.0,
        active_layers=active_layers,
        target_components=("deltanet", "full_attn", "mlp"),
    )
    mx.synchronize()
    apply_ms = (time.perf_counter() - t_apply0) * 1000.0

    # Warmup un-wrapped model kernels
    warmup_model_and_kernels(model)

    prefill_by_len = _measure_cold_prefill_across_lengths(
        model=model,
        base_tokens=prompt_tokens,
        prefill_lengths=prefill_lengths,
        prefill_step_size=2048,
    )

    timeline = HybridTimelineCache(model=model, prefill_step_size=prefill_step_size)
    mx.synchronize()
    t0 = time.perf_counter()
    logits = timeline.prefill_with_snapshots(
        prompt_tokens,
        snapshot_indices=[edit_token_idx],
    )
    mx.eval(logits)
    mx.synchronize()
    cold_prefill_s = max(time.perf_counter() - t0, 1e-9)

    mx.synchronize()
    t1 = time.perf_counter()
    warm_logits = fork_and_edit(
        cache=timeline,
        edit_token_idx=edit_token_idx,
        new_token_ids=edited_suffix_tokens,
        model=model,
        prefill_step_size=prefill_step_size,
    )
    mx.eval(warm_logits)
    mx.synchronize()
    warm_edit_ms = (time.perf_counter() - t1) * 1000.0

    first_next = mx.argmax(warm_logits[:, -1, :], axis=-1)
    mx.eval(first_next)
    decode_tok_s, _ = _measure_single_stream_decode(
        model=model,
        timeline=timeline,
        seed_token=first_next,
        num_decode_steps=decode_steps,
    )
    del timeline

    gen_step_tok_s = _measure_mlx_lm_generate_step_decode(
        model=model,
        prompt_tokens=prompt_tokens,
        num_decode_steps=decode_steps,
    )

    mx.synchronize()
    t_rest0 = time.perf_counter()
    ws_proj.restore_baseline()
    mx.synchronize()
    restore_ms = (time.perf_counter() - t_rest0) * 1000.0

    return {
        "matrices_orthogonalized": num_matrices,
        "cached_v_w_mb": ws_proj.cached_v_w_nbytes / (1024 * 1024),
        "apply_in_place_ms": apply_ms,
        "restore_baseline_ms": restore_ms,
        "cold_prefill_tok_s": len(prompt_tokens) / cold_prefill_s,
        "cold_prefill_by_length": {
            f"len_{k}": v["tok_s"] for k, v in prefill_by_len.items()
        },
        "cold_prefill_ms_by_length": {
            f"len_{k}": v["ms"] for k, v in prefill_by_len.items()
        },
        "warm_edit_refeed_ms": warm_edit_ms,
        "decode_tok_s": decode_tok_s,
        "mlx_lm_generate_step_tok_s": gen_step_tok_s,
    }


def run_benchmark_mode(
    model: nn.Module,
    prompt_tokens: Sequence[int],
    edit_token_idx: int,
    edited_suffix_tokens: Sequence[int],
    prefill_step_size: int = DEFAULT_PREFILL_STEP_SIZE,
    decode_steps: int = 16,
    model_8bit: Optional[nn.Module] = None,
    model_8bit_path: Optional[Union[str, Path]] = None,
    projector: Optional[SubspaceProjector] = None,
    prefill_lengths: Sequence[int] = DEFAULT_BENCH_PREFILL_LENGTHS,
    release_bf16_before_8bit: bool = False,
) -> Dict[str, Any]:
    """
    Benchmark M4 Max:
      1. 2-token dummy warmup pass (`mx.eval(logits)` + `mx.synchronize()`) to separate
         SSD weight page-in and Metal shader compilation from prefill timing.
      2. Multi-length cold prefill throughput (`64`, `512`, `2048` tokens) using a fresh
         `HybridTimelineCache` per length.
      3. Prompt prefill + snapshot at `edit_token_idx` + warm edit-refeed latency (ms) via `fork_and_edit`.
      4. Single-stream autoregressive decode throughput (tok/s).
      5. BF16 vs. 8-bit-local comparison: saves small BF16 reference logits/attention arrays
         to CPU/NumPy, explicitly releases the 55.6 GB BF16 model & cache (`del model`, `del cache`,
         `gc.collect()`, `mx.clear_cache()`) when loading from `model_8bit_path`, and measures
         8-bit cold prefill (`64`, `512`, `2048`), warm edit-refeed (ms), decode (tok/s),
         logit KL, and full-attention JSD.
    """
    enforce_metal_command_buffer_invariant()
    if projector is not None:
        projector.attach(model)

    # 1. Warmup pass to page in weights & compile Metal / @mx.compile kernels
    warmup_ms = warmup_model_and_kernels(model)

    # 2. Multi-length cold prefill benchmark (e.g., 64, 512, 2048 tokens)
    prefill_by_len = _measure_cold_prefill_across_lengths(
        model=model,
        base_tokens=prompt_tokens,
        prefill_lengths=prefill_lengths,
        prefill_step_size=prefill_step_size,
    )

    # 3. Prompt prefill + snapshot at edit_token_idx (without unfused probe overhead for timing)
    timeline = HybridTimelineCache(model=model, prefill_step_size=prefill_step_size)
    mx.synchronize()
    t0 = time.perf_counter()
    logits_bf16 = timeline.prefill_with_snapshots(
        prompt_tokens,
        snapshot_indices=[edit_token_idx],
    )
    mx.eval(logits_bf16)
    mx.synchronize()
    cold_prefill_sec = max(time.perf_counter() - t0, 1e-9)
    cold_prefill_tok_s = len(prompt_tokens) / cold_prefill_sec

    # Capture BF16 reference attention maps for 8-bit fidelity comparison if needed
    need_8bit = (model_8bit is not None) or (
        model_8bit_path is not None and Path(model_8bit_path).exists()
    )
    bf16_attn_np: Dict[int, np.ndarray] = {}
    bf16_logits_np: Optional[np.ndarray] = None
    if need_8bit:
        probe_timeline = HybridTimelineCache(model=model, prefill_step_size=prefill_step_size)
        with capture_attention_maps(model) as probe_bf16:
            ref_logits = probe_timeline.prefill_with_snapshots(
                prompt_tokens,
                snapshot_indices=[edit_token_idx],
            )
            mx.eval(ref_logits)
        bf16_logits_np = np.array(ref_logits.astype(mx.float32), copy=True)
        for l_idx, arr in probe_bf16.attention_maps.items():
            bf16_attn_np[l_idx] = np.array(arr.astype(mx.float32), copy=True)
        del probe_timeline
        del ref_logits

    # 4. Warm edit-refeed latency (ms)
    mx.synchronize()
    t1 = time.perf_counter()
    warm_logits = fork_and_edit(
        cache=timeline,
        edit_token_idx=edit_token_idx,
        new_token_ids=edited_suffix_tokens,
        model=model,
        prefill_step_size=prefill_step_size,
    )
    mx.eval(warm_logits)
    mx.synchronize()
    warm_edit_ms = (time.perf_counter() - t1) * 1000.0

    # 5. Single-stream decode tok/s
    first_next = mx.argmax(warm_logits[:, -1, :], axis=-1)
    mx.eval(first_next)
    decode_tok_s, decoded_ids = _measure_single_stream_decode(
        model=model,
        timeline=timeline,
        seed_token=first_next,
        num_decode_steps=decode_steps,
    )

    cache_summary = timeline.inspect()
    results: Dict[str, Any] = {
        "warmup_page_in_ms": warmup_ms,
        "prompt_tokens": len(prompt_tokens),
        "edit_token_idx": edit_token_idx,
        "edited_suffix_tokens": len(edited_suffix_tokens),
        "prefill_step_size": prefill_step_size,
        "cold_prefill_tok_s": cold_prefill_tok_s,
        "cold_prefill_ms": cold_prefill_sec * 1000.0,
        "cold_prefill_by_length": {
            f"len_{k}": v["tok_s"] for k, v in prefill_by_len.items()
        },
        "cold_prefill_ms_by_length": {
            f"len_{k}": v["ms"] for k, v in prefill_by_len.items()
        },
        "warm_edit_refeed_ms": warm_edit_ms,
        "decode_tok_s": decode_tok_s,
        "decoded_token_ids": decoded_ids,
        "kv_cache_mb": cache_summary.kv_nbytes / (1024 * 1024),
        "deltanet_cache_mb": cache_summary.deltanet_nbytes / (1024 * 1024),
    }

    # 5b. Un-Wrapped Weight-Space Orthogonalization Benchmark on BF16 (if projector provided)
    if projector is not None:
        del timeline
        gc.collect()
        results["weight_space_bf16"] = _benchmark_weight_space_pass(
            model=model,
            vectors=projector.vectors,
            active_layers=sorted(projector.active_layers),
            prompt_tokens=prompt_tokens,
            edit_token_idx=edit_token_idx,
            edited_suffix_tokens=edited_suffix_tokens,
            prefill_lengths=prefill_lengths,
            prefill_step_size=prefill_step_size,
            decode_steps=decode_steps,
        )

    # 6. Optional BF16 vs 8bit-local fidelity & performance comparison
    if need_8bit:
        # Release BF16 caches and optionally BF16 model weights before loading 8-bit
        if "timeline" in locals():
            del timeline
        del logits_bf16
        del warm_logits
        if release_bf16_before_8bit and model_8bit is None:
            if projector is not None:
                projector.detach(model)
            del model
        gc.collect()
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()

        loaded_8bit_internally = False
        if model_8bit is None and model_8bit_path is not None:
            model_8bit, _, _ = safe_load_model_and_tokenizer(model_8bit_path)
            loaded_8bit_internally = True

        assert model_8bit is not None
        if projector is not None:
            projector.attach(model_8bit)

        warmup_8bit_ms = warmup_model_and_kernels(model_8bit)
        prefill_8bit_by_len = _measure_cold_prefill_across_lengths(
            model=model_8bit,
            base_tokens=prompt_tokens,
            prefill_lengths=prefill_lengths,
            prefill_step_size=prefill_step_size,
        )

        # Prompt cold prefill on 8-bit
        timeline_8bit = HybridTimelineCache(
            model=model_8bit, prefill_step_size=prefill_step_size
        )
        mx.synchronize()
        t_q0 = time.perf_counter()
        logits_8bit = timeline_8bit.prefill_with_snapshots(
            prompt_tokens,
            snapshot_indices=[edit_token_idx],
        )
        mx.eval(logits_8bit)
        mx.synchronize()
        q_prefill_sec = max(time.perf_counter() - t_q0, 1e-9)

        # Warm edit-refeed on 8-bit
        mx.synchronize()
        t_q1 = time.perf_counter()
        warm_logits_8bit = fork_and_edit(
            cache=timeline_8bit,
            edit_token_idx=edit_token_idx,
            new_token_ids=edited_suffix_tokens,
            model=model_8bit,
            prefill_step_size=prefill_step_size,
        )
        mx.eval(warm_logits_8bit)
        mx.synchronize()
        warm_edit_8bit_ms = (time.perf_counter() - t_q1) * 1000.0

        # Single-stream decode on 8-bit
        first_next_8bit = mx.argmax(warm_logits_8bit[:, -1, :], axis=-1)
        mx.eval(first_next_8bit)
        decode_8bit_tok_s, _ = _measure_single_stream_decode(
            model=model_8bit,
            timeline=timeline_8bit,
            seed_token=first_next_8bit,
            num_decode_steps=decode_steps,
        )

        # Capture 8-bit attention maps for JSD comparison against CPU-saved BF16 reference
        probe_timeline_8bit = HybridTimelineCache(
            model=model_8bit, prefill_step_size=prefill_step_size
        )
        with capture_attention_maps(model_8bit) as probe_8bit:
            logits_8bit_probe = probe_timeline_8bit.prefill_with_snapshots(
                prompt_tokens,
                snapshot_indices=[edit_token_idx],
            )
            mx.eval(logits_8bit_probe)

        assert bf16_logits_np is not None
        logits_bf16_restored = mx.array(bf16_logits_np)
        attn_bf16_restored = {
            k: mx.array(v) for k, v in bf16_attn_np.items()
        }

        logit_metrics = compute_logit_kl_divergence(logits_bf16_restored, logits_8bit_probe)
        attn_shift = compute_attention_shift(
            attn_bf16_restored, probe_8bit.attention_maps
        )

        del probe_timeline_8bit
        del timeline_8bit

        weight_space_8bit: Optional[Dict[str, Any]] = None
        if projector is not None:
            weight_space_8bit = _benchmark_weight_space_pass(
                model=model_8bit,
                vectors=projector.vectors,
                active_layers=sorted(projector.active_layers),
                prompt_tokens=prompt_tokens,
                edit_token_idx=edit_token_idx,
                edited_suffix_tokens=edited_suffix_tokens,
                prefill_lengths=prefill_lengths,
                prefill_step_size=prefill_step_size,
                decode_steps=decode_steps,
            )

        results["quant_8bit_comparison"] = {
            "warmup_8bit_ms": warmup_8bit_ms,
            "cold_prefill_8bit_tok_s": len(prompt_tokens) / q_prefill_sec,
            "cold_prefill_8bit_by_length": {
                f"len_{k}": v["tok_s"] for k, v in prefill_8bit_by_len.items()
            },
            "cold_prefill_8bit_ms_by_length": {
                f"len_{k}": v["ms"] for k, v in prefill_8bit_by_len.items()
            },
            "warm_edit_refeed_8bit_ms": warm_edit_8bit_ms,
            "decode_8bit_tok_s": decode_8bit_tok_s,
            **logit_metrics,
            "attention_mean_jsd": attn_shift["mean_jsd"],
            "attention_mean_l1": attn_shift["mean_l1"],
            "weight_space_8bit": weight_space_8bit,
        }

        if loaded_8bit_internally:
            del model_8bit
            gc.collect()
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()

    return results


def format_head_shift_table(
    ranked_heads: Sequence[HeadShiftMetric],
    top_k: int = 16,
    highlight_layers: Sequence[int] = ROUTING_HIGHLIGHT_LAYERS,
) -> str:
    """
    Format a Markdown/ASCII table of the full-attention heads with the highest JSD shift,
    marking mid-to-late routing layers (`31, 35, 39, 43, 47`).
    """
    highlight_set = {int(l) for l in highlight_layers}
    lines = [
        "| Rank | Layer | Routing Band | Head | Mean JSD (nats) | Max JSD (nats) | Mean L1 | Max L1 |",
        "|-----:|------:|:------------:|-----:|----------------:|---------------:|--------:|-------:|",
    ]
    for rank, metric in enumerate(ranked_heads[:top_k], start=1):
        band_tag = "★ MID-LATE" if metric.layer_idx in highlight_set else "-"
        lines.append(
            f"| {rank:4d} | {metric.layer_idx:5d} | {band_tag:12s} | {metric.head_idx:4d} | "
            f"{metric.jsd_mean:15.6f} | {metric.jsd_max:14.6f} | "
            f"{metric.l1_mean:7.5f} | {metric.l1_max:6.4f} |"
        )
    return "\n".join(lines)


def format_token_routing_attribution_table(
    attributions: Sequence[HeadTokenRoutingAttribution],
) -> str:
    """
    Format a detailed Markdown table showing, for the top spiking heads by Max JSD:
      - Peak query token index `q*` (abs & rel) and decoded token string inside `<think>`
      - Top-3 key tokens `(k1, k2, k3)` (index, weight, decoded string) under `alpha = 0.0`
      - Top-3 key tokens `(k1, k2, k3)` (index, weight, decoded string) under `alpha = 1.0`
    """
    lines = [
        "| Rank | Head | Max JSD | Peak Query q* (abs / rel) | Top-3 Keys (α = 0.0 Baseline) | Top-3 Keys (α = 1.0 Projected) |",
        "|-----:|:-----|--------:|:--------------------------|:------------------------------|:-------------------------------|",
    ]
    for rank, attr in enumerate(attributions, start=1):
        q_desc = f"`q*={attr.peak_query_abs_idx}` (+{attr.peak_query_rel_idx}) {attr.peak_query_token_str}"
        a0_str = ", ".join(
            f"[{k.key_idx}] {k.token_str} ({k.weight:.3f})"
            for k in attr.top_keys_alpha0
        )
        a1_str = ", ".join(
            f"[{k.key_idx}] {k.token_str} ({k.weight:.3f})"
            for k in attr.top_keys_alpha1
        )
        lines.append(
            f"| {rank:4d} | L{attr.layer_idx}.H{attr.head_idx} | {attr.jsd_max:7.4f} | "
            f"{q_desc} | {a0_str} | {a1_str} |"
        )
    return "\n".join(lines)


def _evaluate_lookback_flip_metrics(
    attn_alpha0: Dict[int, mx.array],
    attn_band: Dict[int, mx.array],
    shift_report: Dict[str, Any],
    think_start_idx: int,
    lookback_heads: Sequence[Tuple[int, int]] = LOOKBACK_HEADS_5,
) -> Dict[str, Any]:
    """
    Evaluate on the 5 lookback heads (`L35.H21`, `L39.H1`, `L43.H7`, `L43.H1`, `L39.H0`):
      - Mean JSD and Peak JSD across the 5 lookback heads
      - Prompt-lookback attention mass on `[27..34]` vs local `<think>` tokens `[think_start_idx..]`
        at the peak query tokens (`q*=69` for `L35.H21`/`L39.H1` and `q*=74` for `L43.H7`/`L43.H1`/`L39.H0`,
        or the dynamic peak query token when testing on shorter synthetic sequences).
    """
    per_layer = shift_report["per_layer"]
    head_mean_jsds: List[float] = []
    head_max_jsds: List[float] = []

    flips_detected = 0
    total_checked = 0
    q69_flips = 0
    q74_flips = 0

    for layer_idx, head_idx in lookback_heads:
        if layer_idx not in per_layer or layer_idx not in attn_band:
            continue
        l_info = per_layer[layer_idx]
        if head_idx < len(l_info["jsd_per_head"]):
            head_mean_jsds.append(float(l_info["jsd_per_head"][head_idx]))
            head_max_jsds.append(float(l_info["jsd_max_per_head"][head_idx]))

        p_mat = attn_alpha0[layer_idx].astype(mx.float32)[0, head_idx]  # (Q_len, K_len)
        q_mat = attn_band[layer_idx].astype(mx.float32)[0, head_idx]    # (Q_len, K_len)
        q_len = int(q_mat.shape[0])
        k_len = int(q_mat.shape[1])

        # Determine target relative query index (23 for q*=69, 28 for q*=74 if available, else argmax row L1)
        if layer_idx in (35, 39) and head_idx in (21, 1) and q_len > 23:
            q_rel = 23
            is_q69 = True
        elif q_len > 28:
            q_rel = 28
            is_q69 = False
        else:
            row_l1 = mx.sum(mx.abs(p_mat - q_mat), axis=-1)
            q_rel = int(mx.argmax(row_l1).item())
            is_q69 = (layer_idx in (35, 39) and head_idx in (21, 1))

        q_abs = think_start_idx + q_rel
        valid_k = min(k_len, q_abs + 1)
        p_row = p_mat[q_rel, :valid_k]
        q_row = q_mat[q_rel, :valid_k]
        mx.eval(p_row, q_row)

        p_np = np.asarray(p_row, dtype=np.float32)
        q_np = np.asarray(q_row, dtype=np.float32)

        top1_k_a0 = int(np.argmax(p_np))
        top1_k_band = int(np.argmax(q_np))

        # Check whether attention mass shifted away from user prompt into `<think>` (or top-1 key moved to >= think_start_idx)
        if valid_k > think_start_idx and think_start_idx > 0:
            prompt_mass_a0 = float(np.sum(p_np[min(27, think_start_idx - 1) : min(35, think_start_idx)]))
            prompt_mass_band = float(np.sum(q_np[min(27, think_start_idx - 1) : min(35, think_start_idx)]))
            think_mass_a0 = float(np.sum(p_np[think_start_idx:valid_k]))
            think_mass_band = float(np.sum(q_np[think_start_idx:valid_k]))
            flipped = (
                (top1_k_band >= think_start_idx and top1_k_a0 < think_start_idx)
                or (think_mass_band > think_mass_a0 + 0.15)
                or (prompt_mass_band < prompt_mass_a0 * 0.5 and top1_k_band != top1_k_a0)
            )
        else:
            flipped = top1_k_band != top1_k_a0

        total_checked += 1
        if flipped:
            flips_detected += 1
            if is_q69:
                q69_flips += 1
            else:
                q74_flips += 1

    lookback_mean_jsd = float(sum(head_mean_jsds) / max(len(head_mean_jsds), 1))
    lookback_peak_jsd = float(max(head_max_jsds)) if head_max_jsds else 0.0

    if flips_detected == total_checked and total_checked > 0:
        flip_status = f"YES ({flips_detected}/{total_checked}: q*69={q69_flips}/2, q*74={q74_flips}/3)"
    elif flips_detected > 0:
        flip_status = f"PARTIAL ({flips_detected}/{total_checked}: q*69={q69_flips}/2, q*74={q74_flips}/3)"
    else:
        flip_status = f"NO (0/{total_checked})"

    return {
        "lookback_5_mean_jsd": lookback_mean_jsd,
        "lookback_5_peak_jsd": lookback_peak_jsd,
        "flips_detected": flips_detected,
        "total_checked": total_checked,
        "flip_status": flip_status,
    }


def format_layer_band_ablation_table(
    ablation_rows: Sequence[Dict[str, Any]],
) -> str:
    """
    Format a comparative Markdown table across the 4 layer-band ablation configurations:
      1. Full GLP-49 Band (`10..58`, 49 layers)
      2. Early-Mid Band Only (`10..29`, 20 layers)
      3. Mid-Late Lookback Band Only (`30..47`, 18 layers)
      4. Sparse Full-Attention Only (`{31, 35, 39, 43, 47}`, 5 layers)
    """
    lines = [
        "| Band Configuration | Active Layers | Logit KL vs α=0.0 | Logit KL vs Full GLP-49 | 5-Head Mean JSD | 5-Head Peak JSD | Prompt-Lookback Flip (q*=69 & q*=74) |",
        "|:-------------------|--------------:|------------------:|------------------------:|----------------:|----------------:|:-------------------------------------|",
    ]
    for row in ablation_rows:
        lines.append(
            f"| {row['name']} | {row['num_layers']:13d} | "
            f"{row['kl_vs_alpha0']:17.6f} | {row['kl_vs_full_glp49']:23.6f} | "
            f"{row['lookback_5_mean_jsd']:15.6f} | {row['lookback_5_peak_jsd']:15.6f} | "
            f"{row['flip_status']} |"
        )
    return "\n".join(lines)


def format_component_writer_ablation_table(
    writer_rows: Sequence[Dict[str, Any]],
) -> str:
    """
    Format a comparative Markdown table across the 5 component-writer ablation configurations:
      1. Activation-Space Hook (`10..58`)
      2. Weight-Space All Writers (`10..58`): `deltanet.out_proj` + `full_attn.o_proj` + `mlp.down_proj`
      3. Weight-Space MLP Writers Only (`10..58`): `mlp.down_proj` only
      4. Weight-Space Gated DeltaNet + MLP Only (`10..58`): `deltanet.out_proj` + `mlp.down_proj`
      5. Weight-Space Full-Attention Writers Only (`10..58`): `full_attn.o_proj` only
    """
    lines = [
        "| Writer Configuration | Matrices Orthogonalized | Apply Time (ms) | Logit KL vs α=0.0 | Logit KL vs Act-Hook (10..58) | 5-Head Mean JSD | 5-Head Peak JSD | Prompt-Lookback Flip (q*=69 & q*=74) |",
        "|:---------------------|------------------------:|----------------:|------------------:|------------------------------:|----------------:|----------------:|:-------------------------------------|",
    ]
    for row in writer_rows:
        lines.append(
            f"| {row['name']} | {str(row['matrices_orthogonalized']):>23s} | "
            f"{row['apply_ms']:15.2f} | "
            f"{row['kl_vs_alpha0']:17.6f} | {row['kl_vs_act_hook']:29.6f} | "
            f"{row['lookback_5_mean_jsd']:15.6f} | {row['lookback_5_peak_jsd']:15.6f} | "
            f"{row['flip_status']} |"
        )
    return "\n".join(lines)


def _stitch_two_stage_attention_maps(
    attn_part1: Dict[int, mx.array],
    attn_part2: Dict[int, mx.array],
) -> Dict[int, mx.array]:
    """
    Combine two sequential causal prefill attention maps:
      - `attn_part1[l]`: shape `(B, H, Q1, K_prefix + Q1)`
      - `attn_part2[l]`: shape `(B, H, Q2, K_prefix + Q1 + Q2)`
    into a unified `(B, H, Q1 + Q2, K_prefix + Q1 + Q2)` attention map by zero-padding
    `attn_part1[l]` along the future key columns `Q2` (which are causally masked to 0).
    """
    stitched: Dict[int, mx.array] = {}
    for l_idx, a1 in attn_part1.items():
        if l_idx not in attn_part2:
            stitched[l_idx] = a1
            continue
        a2 = attn_part2[l_idx]
        B, H, q1, k1 = a1.shape
        _, _, q2, k2 = a2.shape
        pad_k = k2 - k1
        if pad_k > 0:
            zeros_pad = mx.zeros((B, H, q1, pad_k), dtype=a1.dtype)
            a1_padded = mx.concatenate([a1, zeros_pad], axis=-1)
        else:
            a1_padded = a1
        comb = mx.concatenate([a1_padded, a2], axis=2)
        mx.eval(comb)
        stitched[l_idx] = comb
    return stitched


def compute_lookback_attention_mass(
    attn_maps: Dict[int, mx.array],
    think_start_idx: int,
    monitor_heads: Sequence[Tuple[int, int]] = LOOKBACK_HEADS_5,
    tail_window: int = 16,
) -> Dict[str, Any]:
    """
    For the 5 lookback monitor heads (`L35.H21`, `L39.H1`, `L43.H7`, `L43.H1`, `L39.H0`),
    compute the total attention probability mass placed on:
      - User Prompt tokens `[0..think_start_idx-1]`
      - Local `<think>` tokens `[think_start_idx..]`
    averaged across the last `tail_window` (default 16) tokens of the `<think>` clause.
    """
    per_head: Dict[str, Dict[str, float]] = {}
    prompt_masses: List[float] = []
    local_masses: List[float] = []
    actual_window = 0

    for l_idx, h_idx in monitor_heads:
        head_key = f"L{l_idx}.H{h_idx}"
        if l_idx not in attn_maps:
            continue
        mat = attn_maps[l_idx]  # (B, H, Q_len, K_len)
        q_len = int(mat.shape[2])
        w = min(tail_window, q_len)
        actual_window = w
        q_start = q_len - w
        sub = mat[0, h_idx, q_start:q_len, :].astype(mx.float32)  # (w, K_len)

        p_mass = float(mx.mean(mx.sum(sub[:, :think_start_idx], axis=-1)).item())
        l_mass = float(mx.mean(mx.sum(sub[:, think_start_idx:], axis=-1)).item())
        per_head[head_key] = {
            "layer_idx": float(l_idx),
            "head_idx": float(h_idx),
            "prompt_mass": p_mass,
            "local_think_mass": l_mass,
            "local_to_prompt_ratio": l_mass / max(p_mass, 1e-9),
        }
        prompt_masses.append(p_mass)
        local_masses.append(l_mass)

    mean_p = float(sum(prompt_masses) / max(len(prompt_masses), 1))
    mean_l = float(sum(local_masses) / max(len(local_masses), 1))
    return {
        "tail_window_tokens": actual_window,
        "mean_prompt_mass": mean_p,
        "mean_local_think_mass": mean_l,
        "mean_local_to_prompt_ratio": mean_l / max(mean_p, 1e-9),
        "per_head": per_head,
    }


def _compute_cosine_similarity(vec_a: Optional[mx.array], vec_b: Optional[mx.array]) -> float:
    """Compute FP32 cosine similarity between two hidden state vectors."""
    if vec_a is None or vec_b is None:
        return 0.0
    a = vec_a.reshape(-1).astype(mx.float32)
    b = vec_b.reshape(-1).astype(mx.float32)
    dot = float(mx.sum(a * b).item())
    norm_a = float(mx.sqrt(mx.sum(a * a)).item())
    norm_b = float(mx.sqrt(mx.sum(b * b)).item())
    return dot / max(norm_a * norm_b, 1e-12)


def _decode_continuation_branch(
    model: nn.Module,
    timeline: HybridTimelineCache,
    seed_logits: mx.array,
    num_steps: int = 24,
    tokenizer: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Greedily decode `num_steps` continuation tokens from `timeline` starting from `seed_logits`.
    Uses async decode pipelining (`mx.async_eval`) for maximum M4 Max decode throughput.
    """
    if num_steps <= 0:
        return {"token_ids": [], "text": ""}
    next_tok = mx.argmax(seed_logits[:, -1, :], axis=-1, keepdims=True)
    mx.eval(next_tok)
    generated_tokens: List[mx.array] = [next_tok]
    cur = next_tok

    for _ in range(num_steps - 1):
        step_logits = model(cur, cache=timeline.cache)
        next_tok = mx.argmax(step_logits[:, -1, :], axis=-1, keepdims=True)
        mx.async_eval(next_tok)
        generated_tokens.append(next_tok)
        cur = next_tok

    mx.eval(*generated_tokens)
    timeline.eval_all()
    decoded_ids: List[int] = [int(t.item()) for t in generated_tokens]
    if tokenizer is not None and hasattr(tokenizer, "decode"):
        try:
            text = str(tokenizer.decode(decoded_ids))
        except Exception:
            text = " ".join(f"<{tid}>" for tid in decoded_ids)
    else:
        text = " ".join(f"<{tid}>" for tid in decoded_ids)

    return {
        "token_ids": decoded_ids,
        "text": text,
    }


def format_think_2x2_summary_table(
    branches: Sequence[Dict[str, Any]],
) -> str:
    """
    Format the primary 2x2 `<think>` Intervention Matrix summary table comparing
    `A0`, `A1`, `B0`, and `B1_pulse`.
    """
    lines = [
        "| Trajectory Branch | Clause Tokens | Steering Schedule | User Prompt Mass `[0..45]` | Local `<think>` Mass `[46..]` | Local / Prompt Ratio | Layer-63 CosSim vs `A0` | Layer-63 CosSim vs `A1` | Layer-63 CosSim vs `B0` |",
        "|:------------------|--------------:|:------------------|---------------------------:|------------------------------:|---------------------:|------------------------:|------------------------:|------------------------:|",
    ]
    for b in branches:
        mass = b["lookback_mass"]
        cs = b["cosine_similarities"]
        lines.append(
            f"| {b['label']} | {b['clause_len']:13d} | {b['schedule']:17s} | "
            f"{mass['mean_prompt_mass']:26.4f} | {mass['mean_local_think_mass']:29.4f} | "
            f"{mass['mean_local_to_prompt_ratio']:19.2f}x | "
            f"{cs.get('A0', 0.0):23.4f} | {cs.get('A1', 0.0):23.4f} | {cs.get('B0', 0.0):23.4f} |"
        )
    return "\n".join(lines)


def format_think_2x2_per_head_table(
    branches: Sequence[Dict[str, Any]],
    monitor_heads: Sequence[Tuple[int, int]] = LOOKBACK_HEADS_5,
) -> str:
    """
    Format a per-head breakdown table showing `(Prompt Mass / Local <think> Mass)` across all
    5 lookback monitor heads (`L35.H21`, `L39.H1`, `L43.H7`, `L43.H1`, `L39.H0`) for `A0`, `A1`, `B0`, `B1_pulse`.
    """
    head_keys = [f"L{l}.H{h}" for l, h in monitor_heads]
    header_cols = " | ".join(f"{hk} (Prompt / Local)" for hk in head_keys)
    sep_cols = " | ".join("---:" for _ in head_keys)
    lines = [
        f"| Trajectory Branch | {header_cols} | 5-Head Mean (Prompt / Local) |",
        f"|:------------------| {sep_cols} | ---------------------------: |",
    ]
    for b in branches:
        mass = b["lookback_mass"]
        ph = mass["per_head"]
        cells: List[str] = []
        for hk in head_keys:
            if hk in ph:
                cells.append(f"`{ph[hk]['prompt_mass']:.3f}` / `{ph[hk]['local_think_mass']:.3f}`")
            else:
                cells.append("`-`")
        mean_cell = f"**`{mass['mean_prompt_mass']:.3f}` / `{mass['mean_local_think_mass']:.3f}`**"
        lines.append(f"| {b['key']} ({b['short_desc']}) | " + " | ".join(cells) + f" | {mean_cell} |")
    return "\n".join(lines)


def format_think_2x2_cos_sim_table(
    branches: Sequence[Dict[str, Any]],
) -> str:
    """Format the 4x4 pairwise Layer-63 residual-stream cosine similarity matrix."""
    keys = [b["key"] for b in branches]
    header = " | ".join(f"{k:>10s}" for k in keys)
    sep = " | ".join("---------:" for _ in keys)
    lines = [
        f"| Branch | {header} |",
        f"|:-------| {sep} |",
    ]
    for b in branches:
        cs = b["cosine_similarities"]
        row_cells = " | ".join(f"{cs.get(k, 0.0):10.4f}" for k in keys)
        lines.append(f"| **{b['key']}** | {row_cells} |")
    return "\n".join(lines)


def save_think_2x2_lookback_mass_plot(
    branches: Sequence[Dict[str, Any]],
    think_start_idx: int,
    output_path: Union[str, Path] = DEFAULT_THINK_2X2_PLOT_PATH,
    monitor_heads: Sequence[Tuple[int, int]] = LOOKBACK_HEADS_5,
) -> Path:
    """
    Render and save a 4-panel comparison chart `./outputs/think_2x2_lookback_mass.png`
    comparing User-Prompt `[0..think_start_idx-1]` vs. Local-`<think>` `[think_start_idx..]`
    attention probability mass across the 5 monitor heads (`L35.H21`, `L39.H1`, `L43.H7`,
    `L43.H1`, `L39.H0`) and their 5-Head Mean for `A0`, `A1`, `B0`, and `B1_pulse`.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    head_keys = [f"L{l}.H{h}" for l, h in monitor_heads]
    x_labels = head_keys + ["5-Head\nMean"]
    x_pos = np.arange(len(x_labels))
    bar_width = 0.36

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5), dpi=160, sharey=True)
    axes_flat = axes.flatten()

    for idx, b in enumerate(branches[:4]):
        ax = axes_flat[idx]
        mass = b["lookback_mass"]
        ph = mass["per_head"]

        prompt_vals = [ph.get(hk, {}).get("prompt_mass", 0.0) for hk in head_keys] + [
            mass["mean_prompt_mass"]
        ]
        local_vals = [ph.get(hk, {}).get("local_think_mass", 0.0) for hk in head_keys] + [
            mass["mean_local_think_mass"]
        ]

        rects_p = ax.bar(
            x_pos - bar_width / 2,
            prompt_vals,
            width=bar_width,
            color="#2b5c8f",
            edgecolor="#1b3b5c",
            label=f"User Prompt [0..{think_start_idx - 1}]",
        )
        rects_l = ax.bar(
            x_pos + bar_width / 2,
            local_vals,
            width=bar_width,
            color="#d95f02",
            edgecolor="#8c3c00",
            label=f"Local <think> [{think_start_idx}..]",
        )

        # Vertical divider separating individual heads from the 5-Head Mean summary bar
        ax.axvline(len(head_keys) - 0.5, color="#666666", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_title(
            f"{b['label']}\n(5-Head Mean: Prompt={mass['mean_prompt_mass']:.3f}, "
            f"Local <think>={mass['mean_local_think_mass']:.3f})",
            fontsize=10.5,
            fontweight="bold",
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_ylim(0.0, 1.08)
        ax.set_ylabel("Attention Probability Mass (Last 16 Tokens)", fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.45)
        ax.legend(loc="upper left", fontsize=8.5, framealpha=0.92)

        for r in rects_p:
            h_val = r.get_height()
            ax.annotate(
                f"{h_val:.2f}",
                xy=(r.get_x() + r.get_width() / 2, h_val),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.8,
                color="#1b3b5c",
                fontweight="bold",
            )
        for r in rects_l:
            h_val = r.get_height()
            ax.annotate(
                f"{h_val:.2f}",
                xy=(r.get_x() + r.get_width() / 2, h_val),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.8,
                color="#8c3c00",
                fontweight="bold",
            )

    fig.suptitle(
        "Qwen3.8-27B 2x2 <think> Intervention Matrix: User-Prompt Lookback vs. Local-<think> Attention Mass",
        fontsize=12.5,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out_file)
    plt.close(fig)
    return out_file.resolve()


def save_attention_jsd_heatmap(
    per_layer_metrics: Dict[int, Dict[str, Any]],
    output_path: Union[str, Path] = DEFAULT_HEATMAP_PATH,
    highlight_layers: Sequence[int] = ROUTING_HIGHLIGHT_LAYERS,
) -> Path:
    """
    Render and save a 16 (full-attention layers: 3, 7, ..., 63) x 24 (query heads)
    Jensen-Shannon Divergence (JSD) heatmap to `./outputs/attention_jsd_heatmap.png`,
    highlighting mid-to-late routing layers (`31, 35, 39, 43, 47`).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    sorted_layers = sorted(per_layer_metrics.keys())
    num_layers = len(sorted_layers)
    num_heads = (
        len(per_layer_metrics[sorted_layers[0]]["jsd_per_head"])
        if sorted_layers
        else 24
    )

    grid = np.zeros((num_layers, num_heads), dtype=np.float32)
    for row_idx, layer_idx in enumerate(sorted_layers):
        row_vals = per_layer_metrics[layer_idx]["jsd_per_head"]
        grid[row_idx, : len(row_vals)] = np.asarray(row_vals, dtype=np.float32)

    highlight_set = {int(l) for l in highlight_layers}

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=160)
    im = ax.imshow(grid, aspect="auto", cmap="magma", interpolation="nearest")

    ax.set_xticks(range(num_heads))
    ax.set_xticklabels([f"H{h}" for h in range(num_heads)], fontsize=8)
    y_labels = [
        f"L{l} ★" if l in highlight_set else f"L{l}" for l in sorted_layers
    ]
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels(y_labels, fontsize=9)

    for row_idx, layer_idx in enumerate(sorted_layers):
        if layer_idx in highlight_set:
            ax.axhspan(
                row_idx - 0.5,
                row_idx + 0.5,
                fill=False,
                edgecolor="#00e5ff",
                linewidth=1.5,
            )

    ax.set_xlabel("Query Head Index (0..23, GQA 24Q / 4KV)", fontsize=10)
    ax.set_ylabel("Full-Attention Layer Index (3, 7, ..., 63)", fontsize=10)
    ax.set_title(
        "Qwen3.8-27B Mid-<think> Fork-Edit Attention Shift (JSD: α=0.0 vs α=1.0, Layers 10..58)",
        fontsize=11,
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Mean JSD (nats)", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)
    return out_file.resolve()


def save_top_heads_token_routing_plot(
    attributions: Sequence[HeadTokenRoutingAttribution],
    think_start_idx: int,
    output_path: Union[str, Path] = DEFAULT_TOKEN_ROUTING_PLOT_PATH,
    num_heads_to_plot: int = 4,
) -> Path:
    """
    Save `./outputs/top_heads_token_routing.png` showing the side-by-side `alpha=0.0` vs
    `alpha=1.0` attention distribution across the sequence at the peak query token `q*`
    for the top 4 heads by Max JSD.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    selected = list(attributions[:num_heads_to_plot])
    n_plots = max(len(selected), 1)

    fig, axes = plt.subplots(n_plots, 1, figsize=(12.5, 2.7 * n_plots), dpi=160, sharex=False)
    if n_plots == 1:
        axes = [axes]

    for ax, attr in zip(axes, selected):
        k_len = len(attr.dist_alpha0)
        x_indices = np.arange(k_len)
        a0 = np.asarray(attr.dist_alpha0, dtype=np.float32)
        a1 = np.asarray(attr.dist_alpha1, dtype=np.float32)

        width = 0.42
        ax.bar(
            x_indices - width / 2,
            a0,
            width=width,
            label="α = 0.0 (Baseline)",
            color="#3b82f6",
            alpha=0.85,
        )
        ax.bar(
            x_indices + width / 2,
            a1,
            width=width,
            label="α = 1.0 (Projected L10..58)",
            color="#ef4444",
            alpha=0.85,
        )

        if 0 < think_start_idx < k_len:
            ax.axvline(
                think_start_idx - 0.5,
                color="#10b981",
                linestyle="--",
                linewidth=1.2,
                label="<think> fork boundary",
            )

        # Annotate top-2 keys for alpha=0 and alpha=1
        top_annotated = set()
        for k_item in list(attr.top_keys_alpha0[:2]) + list(attr.top_keys_alpha1[:2]):
            if k_item.key_idx in top_annotated or k_item.key_idx >= k_len:
                continue
            top_annotated.add(k_item.key_idx)
            y_pos = max(a0[k_item.key_idx], a1[k_item.key_idx])
            ax.annotate(
                f"[{k_item.key_idx}] {k_item.token_str}",
                xy=(k_item.key_idx, y_pos),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                rotation=15,
            )

        ax.set_ylabel("Attn Weight", fontsize=9)
        ax.set_title(
            f"L{attr.layer_idx}.H{attr.head_idx} at Peak Query q*={attr.peak_query_abs_idx} "
            f"({attr.peak_query_token_str}) — Max JSD = {attr.jsd_max:.4f} nats, L1 = {attr.l1_at_peak:.3f}",
            fontsize=10,
        )
        ax.set_ylim(0.0, min(1.05, max(float(np.max(a0)), float(np.max(a1))) * 1.28 + 0.05))
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    axes[-1].set_xlabel("Key Token Index Across Sequence (0 .. q*)", fontsize=10)
    fig.suptitle(
        "Qwen3.8-27B Top Spiking Full-Attention Heads: Token-Level Routing Shift (α=0.0 vs α=1.0)",
        fontsize=11.5,
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)
    return out_file.resolve()


def run_experiment_mode(
    model: nn.Module,
    projector: SubspaceProjector,
    prefix_tokens: Sequence[int],
    edited_tokens: Sequence[int],
    clause_a_tokens: Optional[Sequence[int]] = None,
    active_layers: Optional[Sequence[int]] = DEFAULT_ACTIVE_LAYERS,
    prefill_step_size: int = DEFAULT_PREFILL_STEP_SIZE,
    top_k_heads: int = 16,
    heatmap_path: Optional[Union[str, Path]] = DEFAULT_HEATMAP_PATH,
    token_routing_plot_path: Optional[Union[str, Path]] = DEFAULT_TOKEN_ROUTING_PLOT_PATH,
    think_2x2_plot_path: Optional[Union[str, Path]] = DEFAULT_THINK_2X2_PLOT_PATH,
    tokenizer: Optional[Any] = None,
    top_n_routing_heads: int = 5,
    pulse_window_tokens: int = 16,
    lookback_tail_window: int = 16,
    continuation_steps: int = 24,
) -> Dict[str, Any]:
    """
    In `experiment` mode, executes a mid-`<think>` reasoning block fork-edit intervention:
      1. Prefills the system + user prompt up to the opening `<think>\\n` token (`prefix_tokens`)
         and calls `timeline_cache.snapshot(think_start_idx)`.
      2. Prefills baseline initial reasoning trace Clause A (`clause_a_tokens`, e.g., 32-64 tokens
         inside `<think>`) and captures all 16 full-attention maps (`probe_clause_a`).
      3. Calls `fork_and_edit(think_start_idx, edited_clause_b_tokens)` to rewind the 16 `KVCache`
         views and restore the 48 `ArraysCache` FP32 recurrent states (`S_t` and `conv_state`),
         then warm-prefills the edited reasoning Clause B under both:
           - Branch 1: `alpha = 0.0` (unprojected baseline with edited `<think>` prefix)
           - Branch 2: `alpha = 1.0`, active layers `10..58` (`range(10, 59)`, exact orthogonal
             subspace projection)
      4. Computes per-layer, per-head JSD & L1 shift across all 16 full-attention layers
         (`3, 7, ..., 63`) and all 24 query heads, highlights mid-to-late routing layers
         (`31, 35, 39, 43, 47`), and saves `./outputs/attention_jsd_heatmap.png`.
      5. Performs token-level routing attribution for the top 5 heads by Max JSD (finding peak
         query token `q*` and top-3 attended key tokens `(k1, k2, k3)` under `alpha=0.0` vs `alpha=1.0`)
         and saves `./outputs/top_heads_token_routing.png` for the top 4 spiking heads.
    """
    enforce_metal_command_buffer_invariant()
    if hasattr(model, "eval"):
        model.eval()
    resolve_decoder_layers(model)

    projector.attach(model)
    if active_layers is not None:
        projector.set_active_layers(active_layers)
    else:
        projector.set_active_layers(DEFAULT_ACTIVE_LAYERS)

    timeline = HybridTimelineCache(model=model, prefill_step_size=prefill_step_size)
    think_start_idx = len(prefix_tokens)
    decoder_layers = resolve_decoder_layers(model)
    last_layer_idx = len(decoder_layers) - 1
    primary_active = (
        list(active_layers) if active_layers is not None else list(DEFAULT_ACTIVE_LAYERS)
    )

    # Step 1: Prefill system + user prompt up to `<think>\n` and snapshot at think_start_idx
    projector.configure(
        alpha=0.0,
        enabled=False,
        clear_token_slice=True,
        capture_hidden_layers=[last_layer_idx],
    )
    timeline.prefill_with_snapshots(prefix_tokens, snapshot_indices=[think_start_idx])

    # Step 2 (2x2 Matrix Branch [A0]): Hesitant <think> Clause A + alpha=0.0 (Baseline)
    baseline_clause_a = (
        list(clause_a_tokens) if clause_a_tokens is not None else list(edited_tokens)
    )
    with capture_attention_maps(model) as probe_clause_a:
        logits_clause_a = timeline.prefill_with_snapshots(baseline_clause_a)
        mx.eval(logits_clause_a)
    h63_a0_raw = projector.get_last_hidden(last_layer_idx)
    h63_a0 = mx.array(h63_a0_raw) if h63_a0_raw is not None else None
    if h63_a0 is not None:
        mx.eval(h63_a0)
    cont_a0 = _decode_continuation_branch(
        model=model,
        timeline=timeline,
        seed_logits=logits_clause_a,
        num_steps=continuation_steps,
        tokenizer=tokenizer,
    )

    # Step 2b (2x2 Matrix Branch [A1]): Hesitant <think> Clause A + alpha=1.0 (Subspace Projection Only)
    with projector.pulse(alpha=1.0, active_layers=primary_active):
        with capture_attention_maps(model) as probe_clause_a_alpha1:
            logits_clause_a_alpha1 = timeline.fork_and_edit(
                edit_token_idx=think_start_idx,
                new_token_ids=baseline_clause_a,
            )
            mx.eval(logits_clause_a_alpha1)
        h63_a1_raw = projector.get_last_hidden(last_layer_idx)
        h63_a1 = mx.array(h63_a1_raw) if h63_a1_raw is not None else None
        if h63_a1 is not None:
            mx.eval(h63_a1)
        cont_a1 = _decode_continuation_branch(
            model=model,
            timeline=timeline,
            seed_logits=logits_clause_a_alpha1,
            num_steps=continuation_steps,
            tokenizer=tokenizer,
        )

    # Step 3a (2x2 Matrix Branch [B0]): Direct Edited <think> Clause B + alpha=0.0 (Token Edit Only, Zero Projection)
    # Note: exiting `projector.pulse(...)` above automatically restored projector to alpha=0.0, enabled=False!
    with capture_attention_maps(model) as probe_branch1_alpha0:
        logits_alpha0 = timeline.fork_and_edit(
            edit_token_idx=think_start_idx,
            new_token_ids=edited_tokens,
        )
        mx.eval(logits_alpha0)
    h63_b0_raw = projector.get_last_hidden(last_layer_idx)
    h63_b0 = mx.array(h63_b0_raw) if h63_b0_raw is not None else None
    if h63_b0 is not None:
        mx.eval(h63_b0)
    cont_b0 = _decode_continuation_branch(
        model=model,
        timeline=timeline,
        seed_logits=logits_alpha0,
        num_steps=continuation_steps,
        tokenizer=tokenizer,
    )

    # Step 3b (2x2 Matrix Branch [B1_pulse]): Direct Edited <think> Clause B + 16-Token alpha=1.0 Pulse -> alpha=0.0 Continuation
    pulse_len = (
        pulse_window_tokens
        if len(edited_tokens) > pulse_window_tokens
        else max(1, len(edited_tokens) // 2)
    )
    rem_tokens = list(edited_tokens[pulse_len:])
    with projector.pulse(alpha=1.0, active_layers=primary_active):
        with capture_attention_maps(model) as probe_b1_pulse_p1:
            logits_b1_p1 = timeline.fork_and_edit(
                edit_token_idx=think_start_idx,
                new_token_ids=list(edited_tokens[:pulse_len]),
            )
            mx.eval(logits_b1_p1)
    # Exiting `projector.pulse(...)` automatically switches back to alpha=0.0 (100% stock model)
    with capture_attention_maps(model) as probe_b1_pulse_p2:
        logits_b1_pulse = timeline.prefill_with_snapshots(rem_tokens)
        mx.eval(logits_b1_pulse)
    h63_b1_pulse_raw = projector.get_last_hidden(last_layer_idx)
    h63_b1_pulse = mx.array(h63_b1_pulse_raw) if h63_b1_pulse_raw is not None else None
    if h63_b1_pulse is not None:
        mx.eval(h63_b1_pulse)
    attn_b1_pulse = _stitch_two_stage_attention_maps(
        probe_b1_pulse_p1.attention_maps,
        probe_b1_pulse_p2.attention_maps,
    )
    cont_b1_pulse = _decode_continuation_branch(
        model=model,
        timeline=timeline,
        seed_logits=logits_b1_pulse,
        num_steps=continuation_steps,
        tokenizer=tokenizer,
    )

    # Step 3c (Full Clause B under alpha=1.0 for JSD head ranking & layer-band / writer ablations)
    projector.configure(alpha=1.0, active_layers=primary_active, enabled=True)
    with capture_attention_maps(model) as probe_branch2_alpha1:
        logits_alpha1 = timeline.fork_and_edit(
            edit_token_idx=think_start_idx,
            new_token_ids=edited_tokens,
        )
        mx.eval(logits_alpha1)

    # Build the 2x2 <think> Intervention Matrix telemetry (`A0`, `A1`, `B0`, `B1_pulse`)
    raw_branches = [
        {
            "key": "A0",
            "label": "[A0] Hesitant Clause A + α=0.0 (Baseline)",
            "short_desc": "Hesitant + α=0.0",
            "clause_len": len(baseline_clause_a),
            "schedule": "α=0.0 (all)",
            "attn_maps": probe_clause_a.attention_maps,
            "h63": h63_a0,
            "continuation": cont_a0,
        },
        {
            "key": "A1",
            "label": "[A1] Hesitant Clause A + α=1.0 (Projected)",
            "short_desc": "Hesitant + α=1.0",
            "clause_len": len(baseline_clause_a),
            "schedule": f"α=1.0 ({len(baseline_clause_a)}t)",
            "attn_maps": probe_clause_a_alpha1.attention_maps,
            "h63": h63_a1,
            "continuation": cont_a1,
        },
        {
            "key": "B0",
            "label": "[B0] Edited Clause B + α=0.0 (Token Edit Only)",
            "short_desc": "Edited + α=0.0",
            "clause_len": len(edited_tokens),
            "schedule": "α=0.0 (all)",
            "attn_maps": probe_branch1_alpha0.attention_maps,
            "h63": h63_b0,
            "continuation": cont_b0,
        },
        {
            "key": "B1_pulse",
            "label": f"[B1_pulse] Edited Clause B ({pulse_len}t α=1.0 -> {len(rem_tokens)}t α=0.0)",
            "short_desc": f"Edited + {pulse_len}t Pulse",
            "clause_len": len(edited_tokens),
            "schedule": f"{pulse_len}t α=1.0 -> {len(rem_tokens)}t α=0.0",
            "attn_maps": attn_b1_pulse,
            "h63": h63_b1_pulse,
            "continuation": cont_b1_pulse,
        },
    ]

    h63_by_key = {b["key"]: b["h63"] for b in raw_branches}
    think_2x2_branches: List[Dict[str, Any]] = []
    for b in raw_branches:
        mass_info = compute_lookback_attention_mass(
            attn_maps=b["attn_maps"],
            think_start_idx=think_start_idx,
            monitor_heads=LOOKBACK_HEADS_5,
            tail_window=lookback_tail_window,
        )
        cos_sims = {
            other_k: _compute_cosine_similarity(b["h63"], other_h)
            for other_k, other_h in h63_by_key.items()
        }
        think_2x2_branches.append(
            {
                "key": b["key"],
                "label": b["label"],
                "short_desc": b["short_desc"],
                "clause_len": b["clause_len"],
                "schedule": b["schedule"],
                "lookback_mass": mass_info,
                "cosine_similarities": cos_sims,
                "continuation_token_ids": b["continuation"]["token_ids"],
                "continuation_text": b["continuation"]["text"],
            }
        )

    think_2x2_summary_table_str = format_think_2x2_summary_table(think_2x2_branches)
    think_2x2_per_head_table_str = format_think_2x2_per_head_table(
        think_2x2_branches, monitor_heads=LOOKBACK_HEADS_5
    )
    think_2x2_cos_sim_table_str = format_think_2x2_cos_sim_table(think_2x2_branches)

    # Step 4: Compute JSD and L1 shifts for primary active band (Full GLP-49: 10..58)
    shift_report = compute_attention_shift(
        probe_branch1_alpha0.attention_maps,
        probe_branch2_alpha1.attention_maps,
    )
    clause_edit_shift = compute_attention_shift(
        probe_clause_a.attention_maps,
        probe_branch1_alpha0.attention_maps,
    )
    logit_divergence = compute_logit_kl_divergence(logits_alpha0, logits_alpha1)

    # Step 5: 4-Band Causal Ablation Study across:
    #   1. Full GLP-49 Band (10..58, 49 layers)
    #   2. Early-Mid Band Only (10..29, 20 layers)
    #   3. Mid-Late Lookback Band Only (30..47, 18 layers)
    #   4. Sparse Full-Attention Only ({31, 35, 39, 43, 47}, 5 layers)
    ablation_rows: List[Dict[str, Any]] = []
    full_glp49_set = set(range(10, 59))
    if set(primary_active) == full_glp49_set:
        logits_full_glp49 = logits_alpha1
        attn_full_glp49 = probe_branch2_alpha1.attention_maps
        shift_full_glp49 = shift_report
    else:
        projector.configure(alpha=1.0, active_layers=range(10, 59), enabled=True)
        with capture_attention_maps(model) as probe_glp49:
            logits_full_glp49 = timeline.fork_and_edit(
                edit_token_idx=think_start_idx,
                new_token_ids=edited_tokens,
            )
            mx.eval(logits_full_glp49)
        attn_full_glp49 = probe_glp49.attention_maps
        shift_full_glp49 = compute_attention_shift(
            probe_branch1_alpha0.attention_maps,
            attn_full_glp49,
        )

    for band_name, band_layers in LAYER_BAND_ABLATION_CONFIGS:
        if set(band_layers) == full_glp49_set:
            b_logits = logits_full_glp49
            b_attn = attn_full_glp49
            b_shift = shift_full_glp49
        else:
            projector.configure(alpha=1.0, active_layers=band_layers, enabled=True)
            with capture_attention_maps(model) as probe_band:
                b_logits = timeline.fork_and_edit(
                    edit_token_idx=think_start_idx,
                    new_token_ids=edited_tokens,
                )
                mx.eval(b_logits)
            b_attn = probe_band.attention_maps
            b_shift = compute_attention_shift(
                probe_branch1_alpha0.attention_maps,
                b_attn,
            )

        kl_vs_a0 = compute_logit_kl_divergence(logits_alpha0, b_logits)["kl_mean_nats"]
        kl_vs_glp49 = compute_logit_kl_divergence(logits_full_glp49, b_logits)["kl_mean_nats"]
        lookback_eval = _evaluate_lookback_flip_metrics(
            attn_alpha0=probe_branch1_alpha0.attention_maps,
            attn_band=b_attn,
            shift_report=b_shift,
            think_start_idx=think_start_idx,
            lookback_heads=LOOKBACK_HEADS_5,
        )
        ablation_rows.append(
            {
                "name": band_name,
                "active_layers": sorted(band_layers),
                "num_layers": len(band_layers),
                "kl_vs_alpha0": kl_vs_a0,
                "kl_vs_full_glp49": kl_vs_glp49,
                "mean_jsd_all_heads": b_shift["mean_jsd"],
                **lookback_eval,
            }
        )

    # Restore primary active_layers on projector
    projector.configure(alpha=1.0, active_layers=primary_active, enabled=True)
    ablation_table_str = format_layer_band_ablation_table(ablation_rows)

    # Step 5b: 5-Configuration Component-Writer Causal Ablation Study (10..58, alpha=1.0)
    #   1. Activation-Space Hook (10..58)
    #   2. Weight-Space All Writers (10..58): deltanet.out_proj + full_attn.o_proj + mlp.down_proj
    #   3. Weight-Space MLP Writers Only (10..58): mlp.down_proj only
    #   4. Weight-Space Gated DeltaNet + MLP Only (10..58): deltanet.out_proj + mlp.down_proj
    #   5. Weight-Space Full-Attention Writers Only (10..58): full_attn.o_proj only
    writer_ablation_rows: List[Dict[str, Any]] = []
    act_hook_kl_vs_a0 = compute_logit_kl_divergence(logits_alpha0, logits_full_glp49)[
        "kl_mean_nats"
    ]
    act_hook_lookback = _evaluate_lookback_flip_metrics(
        attn_alpha0=probe_branch1_alpha0.attention_maps,
        attn_band=attn_full_glp49,
        shift_report=shift_full_glp49,
        think_start_idx=think_start_idx,
        lookback_heads=LOOKBACK_HEADS_5,
    )
    writer_ablation_rows.append(
        {
            "name": "Activation-Space Hook (10..58)",
            "target_components": ["hook"],
            "matrices_orthogonalized": "49 (hook)",
            "apply_ms": 0.0,
            "kl_vs_alpha0": act_hook_kl_vs_a0,
            "kl_vs_act_hook": 0.0,
            "mean_jsd_all_heads": shift_full_glp49["mean_jsd"],
            **act_hook_lookback,
        }
    )

    # Detach activation wrapper so WeightSpaceProjector runs on 100% un-wrapped native layers
    projector.detach(model)
    ws_projector = WeightSpaceProjector(
        model=model,
        vectors=projector.vectors,
        target_layers=range(10, 59),
    )

    for writer_cfg_name, comp_set in COMPONENT_WRITER_ABLATION_CONFIGS:
        t_apply0 = time.perf_counter()
        num_mats = ws_projector.apply_in_place(
            alpha=1.0,
            active_layers=range(10, 59),
            target_components=comp_set,
        )
        apply_ms = (time.perf_counter() - t_apply0) * 1000.0

        with capture_attention_maps(model) as probe_ws:
            ws_logits = timeline.fork_and_edit(
                edit_token_idx=think_start_idx,
                new_token_ids=edited_tokens,
            )
            mx.eval(ws_logits)

        ws_attn = probe_ws.attention_maps
        ws_shift = compute_attention_shift(
            probe_branch1_alpha0.attention_maps,
            ws_attn,
        )
        kl_vs_a0 = compute_logit_kl_divergence(logits_alpha0, ws_logits)["kl_mean_nats"]
        kl_vs_hook = compute_logit_kl_divergence(logits_full_glp49, ws_logits)["kl_mean_nats"]
        ws_lookback = _evaluate_lookback_flip_metrics(
            attn_alpha0=probe_branch1_alpha0.attention_maps,
            attn_band=ws_attn,
            shift_report=ws_shift,
            think_start_idx=think_start_idx,
            lookback_heads=LOOKBACK_HEADS_5,
        )
        writer_ablation_rows.append(
            {
                "name": writer_cfg_name,
                "target_components": sorted(comp_set),
                "matrices_orthogonalized": num_mats,
                "apply_ms": apply_ms,
                "kl_vs_alpha0": kl_vs_a0,
                "kl_vs_act_hook": kl_vs_hook,
                "mean_jsd_all_heads": ws_shift["mean_jsd"],
                **ws_lookback,
            }
        )

    # Restore exact baseline weights and re-attach SubspaceProjector cleanly
    ws_projector.restore_baseline()
    projector.attach(model)
    projector.configure(alpha=1.0, active_layers=primary_active, enabled=True)
    component_writer_table_str = format_component_writer_ablation_table(writer_ablation_rows)

    # Step 6: Token-level routing attribution for top 5 heads by Max JSD
    full_seq_ids = list(prefix_tokens) + list(edited_tokens)
    token_attributions = attribute_top_heads_token_routing(
        attn_alpha0=probe_branch1_alpha0.attention_maps,
        attn_alpha1=probe_branch2_alpha1.attention_maps,
        full_sequence_token_ids=full_seq_ids,
        query_start_idx=think_start_idx,
        tokenizer=tokenizer,
        top_n_heads=top_n_routing_heads,
        top_k_keys=3,
    )
    token_routing_table_str = format_token_routing_attribution_table(token_attributions)

    # Summarize mid-to-late routing layers (31, 35, 39, 43, 47)
    routing_layer_summary: Dict[int, Dict[str, float]] = {}
    for l_idx in ROUTING_HIGHLIGHT_LAYERS:
        if l_idx in shift_report["per_layer"]:
            layer_info = shift_report["per_layer"][l_idx]
            routing_layer_summary[l_idx] = {
                "layer_jsd_mean": layer_info["layer_jsd_mean"],
                "layer_jsd_max": max(layer_info["jsd_max_per_head"]),
                "layer_l1_mean": layer_info["layer_l1_mean"],
            }

    table_str = format_head_shift_table(
        shift_report["ranked_heads"],
        top_k=top_k_heads,
        highlight_layers=ROUTING_HIGHLIGHT_LAYERS,
    )

    saved_heatmap: Optional[str] = None
    if heatmap_path is not None:
        saved_path = save_attention_jsd_heatmap(
            per_layer_metrics=shift_report["per_layer"],
            output_path=heatmap_path,
            highlight_layers=ROUTING_HIGHLIGHT_LAYERS,
        )
        saved_heatmap = str(saved_path)

    saved_routing_plot: Optional[str] = None
    if token_routing_plot_path is not None and token_attributions:
        saved_r_path = save_top_heads_token_routing_plot(
            attributions=token_attributions,
            think_start_idx=think_start_idx,
            output_path=token_routing_plot_path,
            num_heads_to_plot=min(4, len(token_attributions)),
        )
        saved_routing_plot = str(saved_r_path)

    saved_2x2_plot: Optional[str] = None
    if think_2x2_plot_path is not None and think_2x2_branches:
        saved_2x2 = save_think_2x2_lookback_mass_plot(
            branches=think_2x2_branches,
            think_start_idx=think_start_idx,
            output_path=think_2x2_plot_path,
            monitor_heads=LOOKBACK_HEADS_5,
        )
        saved_2x2_plot = str(saved_2x2)

    return {
        "edit_token_idx": think_start_idx,
        "think_start_idx": think_start_idx,
        "clause_a_len": len(baseline_clause_a),
        "edited_len": len(edited_tokens),
        "active_layers": sorted(projector.active_layers),
        "logit_divergence": logit_divergence,
        "mean_jsd": shift_report["mean_jsd"],
        "mean_l1": shift_report["mean_l1"],
        "clause_a_vs_b_mean_jsd": clause_edit_shift["mean_jsd"],
        "routing_layer_summary": routing_layer_summary,
        "ranked_heads": shift_report["ranked_heads"],
        "ranked_heads_by_max": shift_report["ranked_heads_by_max"],
        "token_attributions": token_attributions,
        "token_routing_table": token_routing_table_str,
        "layer_band_ablations": ablation_rows,
        "layer_band_ablation_table": ablation_table_str,
        "component_writer_ablations": writer_ablation_rows,
        "component_writer_ablation_table": component_writer_table_str,
        "think_2x2_branches": think_2x2_branches,
        "think_2x2_summary_table": think_2x2_summary_table_str,
        "think_2x2_per_head_table": think_2x2_per_head_table_str,
        "think_2x2_cos_sim_table": think_2x2_cos_sim_table_str,
        "per_layer": shift_report["per_layer"],
        "table": table_str,
        "heatmap_path": saved_heatmap,
        "token_routing_plot_path": saved_routing_plot,
        "think_2x2_plot_path": saved_2x2_plot,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3.8-27B MLX Interpretability, Quantization & Edit-Refeed Harness"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help="Local path or HF repo ID for Qwen3.8-27B (e.g. ./models/Qwen3.8-27B-bf16).",
    )
    parser.add_argument(
        "--model-8bit-path",
        type=str,
        default=DEFAULT_8BIT_LOCAL_PATH,
        help="Local directory for the 8-bit quantized variant (./models/Qwen3.8-27B-8bit-local).",
    )
    parser.add_argument(
        "--vector-path",
        type=str,
        default=None,
        help="Path to local .safetensors or .gguf per-layer steering/projection vectors.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["convert-8bit", "bench", "experiment"],
        help="Execution mode: 'convert-8bit', 'bench', or 'experiment'.",
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=DEFAULT_PREFILL_STEP_SIZE,
        choices=[512, 1024],
        help="Prefill chunk size (must be 512 or 1024).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\nExplain the mathematical relationship between Gated DeltaNet fast-weight "
            "linear attention states and exact orthogonal subspace projections in deep transformers.<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n"
        ),
        help="Initial system + user prompt ending at the opening `<think>\\n` boundary.",
    )
    parser.add_argument(
        "--clause-a",
        type=str,
        default=(
            "Wait—before I proceed, I should carefully check whether discussing these internal "
            "attention states and subspace projections violates any boundaries or guidelines before "
            "deriving the fast-weight recurrence."
        ),
        help="Hesitant/self-interrupting initial reasoning trace Clause A inside `<think>` before forking.",
    )
    parser.add_argument(
        "--edited-suffix",
        type=str,
        default=(
            "Wait—let us explicitly intervene inside the <think> block by projecting out u_hat_l "
            "across layers 10..58 and observing how the 16 gated full-attention routing heads shift."
        ),
        help="Edited reasoning Clause B fed into fork_and_edit at `think_start_idx`.",
    )
    parser.add_argument(
        "--active-layers",
        type=str,
        default="10-59",
        help="Layer range for SubspaceProjector (default '10-59' corresponds to layers 10..58 inclusive).",
    )
    parser.add_argument(
        "--heatmap-path",
        type=str,
        default=DEFAULT_HEATMAP_PATH,
        help="Path to save the 16x24 JSD attention shift heatmap PNG (default: ./outputs/attention_jsd_heatmap.png).",
    )
    parser.add_argument(
        "--token-routing-plot-path",
        type=str,
        default=DEFAULT_TOKEN_ROUTING_PLOT_PATH,
        help="Path to save the top-4 heads token-level routing bar plot PNG (default: ./outputs/top_heads_token_routing.png).",
    )
    parser.add_argument(
        "--think-2x2-plot-path",
        type=str,
        default=DEFAULT_THINK_2X2_PLOT_PATH,
        help="Path to save the 4-panel 2x2 <think> lookback mass comparison PNG (default: ./outputs/think_2x2_lookback_mass.png).",
    )
    parser.add_argument(
        "--continuation-steps",
        type=int,
        default=24,
        help="Number of continuation tokens to decode from each of the 4 branches in the 2x2 matrix.",
    )
    parser.add_argument(
        "--top-k-heads",
        type=int,
        default=16,
        help="Number of top full-attention heads by JSD shift to display.",
    )
    return parser.parse_args(argv)


def _parse_layer_range(spec: str) -> List[int]:
    spec = spec.strip()
    if "-" in spec:
        start_s, end_s = spec.split("-", 1)
        return list(range(int(start_s), int(end_s)))
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    enforce_metal_command_buffer_invariant()

    if args.mode == "convert-8bit":
        out_path = convert_bf16_to_local_8bit(
            source_model_path=args.model_path,
            output_8bit_path=args.model_8bit_path,
            q_bits=8,
            q_group_size=64,
        )
        print(f"[OK] Converted 8-bit local checkpoint saved to: {out_path}")
        return

    model, tokenizer, _ = safe_load_model_and_tokenizer(args.model_path)
    prompt_ids = tokenizer.encode(args.prompt)
    clause_a_ids = tokenizer.encode(args.clause_a)
    suffix_ids = tokenizer.encode(args.edited_suffix)
    edit_idx = max(1, len(prompt_ids) // 2)

    projector: Optional[SubspaceProjector] = None
    if args.vector_path is not None and Path(args.vector_path).exists():
        vectors = load_subspace_vectors(args.vector_path, hidden_size=HIDDEN_SIZE_DEFAULT)
        active_layers = _parse_layer_range(args.active_layers)
        projector = SubspaceProjector(
            vectors=vectors,
            alpha=1.0,
            active_layers=active_layers,
            enabled=True,
        )

    if args.mode == "bench":
        model_8bit_path = (
            args.model_8bit_path if Path(args.model_8bit_path).exists() else None
        )
        bench_res = run_benchmark_mode(
            model=model,
            prompt_tokens=prompt_ids,
            edit_token_idx=edit_idx,
            edited_suffix_tokens=suffix_ids,
            prefill_step_size=args.prefill_step_size,
            model_8bit=None,
            model_8bit_path=model_8bit_path,
            projector=projector,
            prefill_lengths=DEFAULT_BENCH_PREFILL_LENGTHS,
            release_bf16_before_8bit=True,
        )
        print("=== M4 Max Benchmark Results ===")
        for k, v in bench_res.items():
            print(f"  {k}: {v}")
        return

    if args.mode == "experiment":
        if projector is None:
            if args.vector_path is None:
                raise ValueError("--vector-path is required when --mode=experiment.")
            vectors = load_subspace_vectors(args.vector_path, hidden_size=HIDDEN_SIZE_DEFAULT)
            active_layers = _parse_layer_range(args.active_layers)
            projector = SubspaceProjector(
                vectors=vectors,
                alpha=1.0,
                active_layers=active_layers,
                enabled=True,
            )
        else:
            active_layers = sorted(projector.active_layers)

        warmup_ms = warmup_model_and_kernels(model)
        exp_res = run_experiment_mode(
            model=model,
            projector=projector,
            prefix_tokens=prompt_ids,
            edited_tokens=suffix_ids,
            clause_a_tokens=clause_a_ids,
            active_layers=active_layers,
            prefill_step_size=args.prefill_step_size,
            top_k_heads=args.top_k_heads,
            heatmap_path=args.heatmap_path,
            token_routing_plot_path=args.token_routing_plot_path,
            think_2x2_plot_path=args.think_2x2_plot_path,
            tokenizer=tokenizer,
            top_n_routing_heads=5,
            continuation_steps=args.continuation_steps,
        )
        print("=== Mid-<think> Subspace Projection & Fork-Edit Experiment ===")
        print(f"Warmup / Page-In (ms)  : {warmup_ms:.2f}")
        print(f"Think Start Index      : {exp_res['think_start_idx']}")
        print(f"Clause A Tokens        : {exp_res['clause_a_len']}")
        print(f"Edited Clause B Tokens : {exp_res['edited_len']}")
        print(
            f"Active Layers          : {exp_res['active_layers'][0]}..{exp_res['active_layers'][-1]} "
            f"({len(exp_res['active_layers'])} layers)"
        )
        print(f"Logit KL (nats)        : {exp_res['logit_divergence']['kl_mean_nats']:.6f}")
        print(f"Mean Attn JSD (α0/α1)  : {exp_res['mean_jsd']:.6f}")
        print(f"Mean Attn L1  (α0/α1)  : {exp_res['mean_l1']:.6f}")
        print(f"Clause A vs B JSD      : {exp_res['clause_a_vs_b_mean_jsd']:.6f}")
        print("Mid-to-Late Routing Layers (31, 35, 39, 43, 47):")
        for r_layer, r_stats in exp_res["routing_layer_summary"].items():
            print(
                f"  Layer {r_layer:2d} -> Mean JSD: {r_stats['layer_jsd_mean']:.6f}, "
                f"Max Head JSD: {r_stats['layer_jsd_max']:.6f}, Mean L1: {r_stats['layer_l1_mean']:.6f}"
            )
        if exp_res.get("heatmap_path"):
            print(f"Saved JSD Heatmap      : {exp_res['heatmap_path']}")
        if exp_res.get("token_routing_plot_path"):
            print(f"Saved Token Routing    : {exp_res['token_routing_plot_path']}")
        if exp_res.get("think_2x2_plot_path"):
            print(f"Saved 2x2 Lookback Plot: {exp_res['think_2x2_plot_path']}\n")
        print("### Ranked Full-Attention Heads (by Mean JSD)")
        print(exp_res["table"])
        print("\n### Top 5 Spiking Heads (by Max JSD) — Token-Level Routing Attribution")
        print(exp_res["token_routing_table"])
        print("\n### Layer-Band Causal Ablation Study (4 Configurations vs α=0.0 & Full GLP-49)")
        print(exp_res["layer_band_ablation_table"])
        print("\n### Component-Writer Causal Ablation Study (5 Configurations on Layers 10..58, α=1.0)")
        print(exp_res["component_writer_ablation_table"])
        print("\n### 2x2 `<think>` Intervention Matrix (Summary: Lookback Mass & Layer-63 Residual CosSim)")
        print(exp_res["think_2x2_summary_table"])
        print("\n### 2x2 `<think>` Intervention Matrix — Per-Head Prompt `[0..45]` vs. Local `<think>` `[46..]` Mass")
        print(exp_res["think_2x2_per_head_table"])
        print("\n### 2x2 `<think>` Intervention Matrix — 4x4 Layer-63 Residual Stream Cosine Similarity")
        print(exp_res["think_2x2_cos_sim_table"])
        print(f"\n### 2x2 `<think>` Intervention Matrix — {args.continuation_steps}-Token Reasoning Continuations")
        for b in exp_res["think_2x2_branches"]:
            escaped_text = b["continuation_text"].replace("\n", "\\n")
            print(f"  * {b['label']}:")
            print(f"      \"{escaped_text}\"")


if __name__ == "__main__":
    main()
