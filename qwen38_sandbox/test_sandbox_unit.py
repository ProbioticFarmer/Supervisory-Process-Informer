"""
test_sandbox_unit.py — Dry-run unit tests for Qwen3.8-27B MLX Interpretability Sandbox

Verifies with synthetic 64-layer, 5120-d tensors:
  1. `subspace_projector.py`:
     - Loading unit vectors from both `.safetensors` (`direction.{l}` and PEFT `lora_B`) and `.gguf`
     - Unit L2 normalization in `mx.bfloat16`
     - Invariant 1: `assert 0 not in active_layers` raises `AssertionError`
     - Invariant 2: `alpha > 1.0` emits a `UserWarning`
     - Exact rank-1 orthogonal projection `h <- h - alpha * (h . u_hat_l) * u_hat_l`
       zeroes out the parallel component at `alpha = 1.0`, leaves Layer 0 untouched,
       and dynamically switches `active_layers` (`range(10, 59)` vs `range(1, 64)`)
       and `alpha` without reloading weights.
  2. `hybrid_cache.py`:
     - 64-layer hybrid cache (`make_prompt_cache` -> 48 `ArraysCache` + 16 `KVCache` at 3, 7, ..., 63)
     - `rewind_to(token_idx)` slices `KVCache` keys/values along sequence axis and updates `offset`
     - `snapshot(token_idx)` and `restore(token_idx)` deep-copy and `mx.eval` the 48 FP32
       Gated DeltaNet fast-weight states + Conv1d buffers
     - `fork_and_edit(cache, edit_token_idx, new_token_ids)` matches a clean full prefill of
       `prefix[:edit_token_idx] + new_token_ids` to machine precision!
  3. `attention_probe.py`:
     - `capture_attention_maps` context manager records all 16 full-attention layers
       (24 Q heads, 4 KV heads, head_dim 256, partial RoPE dim 64)
     - Unfused attention forward output matches fused `Qwen3NextAttention` output
     - `compute_attention_shift` computes per-layer, per-head JSD and L1 distance
  4. `benchmark_and_experiment.py`:
     - End-to-end `run_experiment_mode` and `run_benchmark_mode` on the synthetic 64-layer hybrid model.
"""

from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path
from typing import Any, List, Optional

import gguf
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import ArraysCache, KVCache, make_prompt_cache
from mlx_lm.models.rope_utils import initialize_rope

from attention_probe import (
    FULL_ATTENTION_LAYERS_DEFAULT,
    ROUTING_HIGHLIGHT_LAYERS,
    capture_attention_maps,
    compute_attention_shift,
)
from benchmark_and_experiment import (
    SAFE_ALLOW_PATTERNS,
    format_head_shift_table,
    run_benchmark_mode,
    run_experiment_mode,
)
from download_assets import (
    MODEL_ALLOW_PATTERNS,
    PINNED_QWEN38_REPO_ID,
    PINNED_QWEN38_REVISION,
    pull_or_verify_model,
    smoke_check_vectors,
)
from hybrid_cache import (
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
    normalize_unit_vector,
    resolve_decoder_layers,
    unwrap_model,
)


class SyntheticGatedDeltaNet(nn.Module):
    """
    Lightweight 5120-d Gated DeltaNet recurrent linear-attention mock that faithfully
    mirrors `mlx_lm.models.qwen3_5.GatedDeltaNet` cache semantics:
      - `cache[0]`: Conv1d state buffer of shape `(B, conv_kernel_size - 1, conv_dim)` (BF16)
      - `cache[1]`: Fast-weight state S_t of shape `(B, num_v_heads, head_v_dim, head_k_dim)` (FP32)
      - Updates `cache.advance(S)` on each forward step.
    """

    def __init__(
        self,
        layer_idx: int,
        hidden_size: int = HIDDEN_SIZE_DEFAULT,
        num_v_heads: int = 8,
        head_k_dim: int = 16,
        head_v_dim: int = 16,
        conv_kernel_size: int = 4,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.conv_kernel_size = conv_kernel_size
        self.conv_dim = 64

        # Deterministic token-dependent state transition factors
        scale = 1.0 / (hidden_size**0.5)
        self.proj_down = (
            mx.arange(hidden_size * self.conv_dim, dtype=mx.float32)
            .reshape(hidden_size, self.conv_dim)
            % 17
            - 8.0
        ) * (scale * 0.05)
        self.proj_down = self.proj_down.astype(mx.bfloat16)

        self.proj_up = (
            mx.arange(self.conv_dim * hidden_size, dtype=mx.float32)
            .reshape(self.conv_dim, hidden_size)
            % 13
            - 6.0
        ) * (scale * 0.05)
        self.proj_up = self.proj_up.astype(mx.bfloat16)
        self.out_proj = nn.Linear(self.conv_dim, hidden_size, bias=False)
        self.out_proj.weight = mx.contiguous(self.proj_up.T)
        mx.eval(self.proj_down, self.proj_up, self.out_proj.weight)

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[ArraysCache] = None,
    ) -> mx.array:
        B, S, _ = inputs.shape
        qkv = mx.matmul(inputs, self.proj_down)  # (B, S, conv_dim)

        n_keep = self.conv_kernel_size - 1
        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros((B, n_keep, self.conv_dim), dtype=inputs.dtype)

        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        if cache is not None:
            cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])

        if cache is not None and cache[1] is not None:
            ssm_state = cache[1]
        else:
            ssm_state = mx.zeros(
                (B, self.num_v_heads, self.head_v_dim, self.head_k_dim),
                dtype=mx.float32,
            )

        # Causal step-by-step fast-weight contraction so chunked prefill and token-by-token
        # prefill yield identical recurrent state S_t and outputs
        outputs = []
        decay = 0.92
        for t in range(S):
            window = conv_input[:, t : t + self.conv_kernel_size, :]  # (B, K, conv_dim)
            conv_step = mx.mean(window, axis=1)  # (B, conv_dim)
            step_f32 = conv_step.astype(mx.float32)
            k_vec = step_f32[:, : self.head_k_dim]  # (B, d_k)
            v_vec = step_f32[:, self.head_k_dim : self.head_k_dim + self.head_v_dim]  # (B, d_v)
            outer = v_vec[:, None, :, None] * k_vec[:, None, None, :]  # (B, 1, d_v, d_k)
            ssm_state = decay * ssm_state + 0.08 * outer
            readout = mx.mean(ssm_state, axis=(1, 2))  # (B, d_k)
            # Combine readout into conv_step
            mod_step = conv_step + readout[:, :1].astype(inputs.dtype) * 0.1
            outputs.append(mod_step[:, None, :])

        if cache is not None:
            cache[1] = ssm_state
            cache.advance(S)

        stacked = mx.concatenate(outputs, axis=1)  # (B, S, conv_dim)
        return self.out_proj(stacked)


class SyntheticFullAttention(nn.Module):
    """
    Full-Attention module matching Qwen3.8-27B's `Qwen3NextAttention` architecture:
      - 24 Q heads, 4 KV heads, head_dim = 256, partial RoPE factor = 0.25 (RoPE dim = 64)
      - `q_proj` outputs `num_attention_heads * head_dim * 2` (queries + sigmoid gate)
      - `q_norm` and `k_norm` RMSNorm(256)
      - Updates `KVCache` via `cache.update_and_fetch(keys, values)`
    """

    def __init__(
        self,
        hidden_size: int = HIDDEN_SIZE_DEFAULT,
        num_attention_heads: int = 24,
        num_key_value_heads: int = 4,
        head_dim: int = 256,
        partial_rotary_factor: float = 0.25,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(
            hidden_size, num_attention_heads * head_dim * 2, bias=False
        )
        self.k_proj = nn.Linear(
            hidden_size, num_key_value_heads * head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            hidden_size, num_key_value_heads * head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            num_attention_heads * head_dim, hidden_size, bias=False
        )

        self.q_norm = nn.RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=rms_norm_eps)

        rope_dim = int(head_dim * partial_rotary_factor)
        assert rope_dim == 64, f"Expected partial RoPE dim 64, got {rope_dim}"
        self.rope = initialize_rope(
            rope_dim,
            base=100000.0,
            traditional=False,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[Any] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        q_proj_output = self.q_proj(x)
        queries, gate = mx.split(
            q_proj_output.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(B, L, -1)

        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.num_key_value_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output * mx.sigmoid(gate))


class SyntheticMLP(nn.Module):
    def __init__(self, hidden_size: int = HIDDEN_SIZE_DEFAULT, inter_dim: int = 64):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, inter_dim, bias=False)
        self.down_proj = nn.Linear(inter_dim, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.up_proj(x)))


class SyntheticHybridDecoderLayer(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        hidden_size: int = HIDDEN_SIZE_DEFAULT,
        full_attention_interval: int = 4,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_linear = (layer_idx + 1) % full_attention_interval != 0
        if self.is_linear:
            self.linear_attn = SyntheticGatedDeltaNet(
                layer_idx=layer_idx, hidden_size=hidden_size
            )
        else:
            self.self_attn = SyntheticFullAttention(hidden_size=hidden_size)
        self.mlp = SyntheticMLP(hidden_size=hidden_size, inter_dim=64)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[Any] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if self.is_linear:
            h = x + self.linear_attn(x, mask=mask, cache=cache)
        else:
            h = x + self.self_attn(x, mask=mask, cache=cache)
        return h + self.mlp(h)


class SyntheticQwen38InnerModel(nn.Module):
    def __init__(
        self,
        num_layers: int = NUM_LAYERS_DEFAULT,
        hidden_size: int = HIDDEN_SIZE_DEFAULT,
        vocab_size: int = 256,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [
            SyntheticHybridDecoderLayer(layer_idx=i, hidden_size=hidden_size)
            for i in range(num_layers)
        ]
        self.norm = nn.RMSNorm(hidden_size)
        self.ssm_idx = 0
        self.fa_idx = 3

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs).astype(mx.bfloat16)
        if cache is None:
            cache = [None] * len(self.layers)
        fa_mask = create_attention_mask(h, cache[self.fa_idx])
        ssm_mask = create_ssm_mask(h, cache[self.ssm_idx])
        for layer, c in zip(self.layers, cache):
            mask = ssm_mask if layer.is_linear else fa_mask
            h = layer(h, mask=mask, cache=c)
        return self.norm(h)


class SyntheticQwen38Model(nn.Module):
    """
    64-layer, 5120-d synthetic Qwen3.8-27B hybrid model (`qwen3_5` 3:1 architecture:
    48 Gated DeltaNet linear-attention layers + 16 full-attention layers at 3, 7, ..., 63).
    """

    def __init__(
        self,
        num_layers: int = NUM_LAYERS_DEFAULT,
        hidden_size: int = HIDDEN_SIZE_DEFAULT,
        vocab_size: int = 256,
    ):
        super().__init__()
        self.model = SyntheticQwen38InnerModel(
            num_layers=num_layers, hidden_size=hidden_size, vocab_size=vocab_size
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        # Cast parameters to bfloat16 to match Qwen3.8-27B-bf16 activations
        self.apply(lambda x: x.astype(mx.bfloat16) if mx.issubdtype(x.dtype, mx.floating) else x)
        mx.eval(self.parameters())

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self) -> List[Any]:
        return [ArraysCache(size=2) if l.is_linear else KVCache() for l in self.layers]

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        h = self.model(inputs, cache=cache)
        return self.lm_head(h)


class TestQwen38Sandbox(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mx.random.seed(42)
        cls.model = SyntheticQwen38Model(
            num_layers=NUM_LAYERS_DEFAULT,
            hidden_size=HIDDEN_SIZE_DEFAULT,
            vocab_size=256,
        )

    def tearDown(self):
        # Ensure model is always unwrapped between tests
        for idx, layer in enumerate(self.model.model.layers):
            if hasattr(layer, "base_layer"):
                self.model.model.layers[idx] = layer.base_layer

    def test_01_subspace_projector_invariants_and_file_loading(self):
        """Verify safetensors/GGUF loading, unit norm in bf16, Layer 0 invariant, and alpha>1 warning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # 1. Test .safetensors with both `direction.{l}` and `lora_B` keys
            st_file = tmp_path / "vectors.safetensors"
            st_tensors = {}
            for l_idx in range(NUM_LAYERS_DEFAULT):
                raw_v = mx.random.normal((HIDDEN_SIZE_DEFAULT,)) * 5.0
                if l_idx % 2 == 0:
                    st_tensors[f"direction.{l_idx}"] = raw_v
                else:
                    # Rank-1 PEFT lora_B shape (5120, 1) + ignored lora_A
                    st_tensors[f"model.layers.{l_idx}.self_attn.o_proj.lora_B.weight"] = (
                        raw_v[:, None]
                    )
                    st_tensors[f"model.layers.{l_idx}.self_attn.o_proj.lora_A.weight"] = (
                        mx.ones((1, HIDDEN_SIZE_DEFAULT))
                    )
            mx.save_safetensors(str(st_file), st_tensors)

            loaded_st = load_subspace_vectors(st_file, hidden_size=HIDDEN_SIZE_DEFAULT)
            self.assertEqual(len(loaded_st), NUM_LAYERS_DEFAULT)
            for l_idx, u in loaded_st.items():
                self.assertEqual(u.shape, (HIDDEN_SIZE_DEFAULT,))
                self.assertEqual(u.dtype, mx.bfloat16)
                l2_norm = float(mx.sqrt(mx.sum(u.astype(mx.float32) ** 2)).item())
                self.assertAlmostEqual(l2_norm, 1.0, places=2)

            # 2. Test .gguf loading
            gguf_file = tmp_path / "vectors.gguf"
            writer = gguf.GGUFWriter(str(gguf_file), "qwen3_5")
            for l_idx in range(1, 5):
                arr_np = np.random.randn(HIDDEN_SIZE_DEFAULT).astype(np.float32)
                writer.add_tensor(f"direction.{l_idx}", arr_np)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()

            loaded_gguf = load_subspace_vectors(gguf_file, hidden_size=HIDDEN_SIZE_DEFAULT)
            self.assertEqual(sorted(loaded_gguf.keys()), [1, 2, 3, 4])
            for u in loaded_gguf.values():
                self.assertEqual(u.dtype, mx.bfloat16)
                self.assertEqual(u.shape, (HIDDEN_SIZE_DEFAULT,))

            # 3. Invariant 1: Never project Layer 0 (`assert 0 not in active_layers`)
            with self.assertRaises(AssertionError):
                SubspaceProjector(
                    vectors=loaded_st,
                    alpha=1.0,
                    active_layers=range(0, 64),
                )

            projector = SubspaceProjector(
                vectors=loaded_st,
                alpha=1.0,
                active_layers=range(1, 64),
            )
            with self.assertRaises(AssertionError):
                projector.set_active_layers([0, 10, 20])

            # 4. Invariant 2: Warn if alpha > 1.0
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                projector.set_alpha(1.25)
                self.assertTrue(
                    any(issubclass(w.category, UserWarning) for w in caught),
                    "Expected UserWarning when setting alpha > 1.0",
                )

    def test_02_subspace_projector_exact_orthogonal_projection_and_dynamic_reconfig(self):
        """Verify h <- h - alpha * (h . u_hat_l) * u_hat_l on synthetic 64-layer, 5120-d activations."""
        vectors = {
            l: normalize_unit_vector(mx.random.normal((HIDDEN_SIZE_DEFAULT,)), dtype=mx.bfloat16)
            for l in range(1, NUM_LAYERS_DEFAULT)
        }
        projector = SubspaceProjector(
            vectors=vectors,
            alpha=1.0,
            active_layers=range(10, 59),
            enabled=True,
        )
        projector.attach(self.model)

        # Layer 0 must NEVER be wrapped
        self.assertFalse(hasattr(self.model.model.layers[0], "base_layer"))
        # Layers 1..63 are wrapped
        for l_idx in range(1, NUM_LAYERS_DEFAULT):
            self.assertTrue(hasattr(self.model.model.layers[l_idx], "base_layer"))

        # Test exact orthogonal projection at alpha=1.0 on Layer 15 (active) vs Layer 5 (inactive)
        u15 = vectors[15]
        # Give x a strong parallel component (magnitude 10.0) along u15 so projection is unambiguous
        x = (
            mx.random.normal((1, 8, HIDDEN_SIZE_DEFAULT)).astype(mx.bfloat16) * 0.1
            + 10.0 * u15[None, None, :]
        )
        mx.eval(x)

        out_layer15 = self.model.model.layers[15](x)
        mx.eval(out_layer15)
        u15_f32 = u15.astype(mx.float32)
        dot_after = mx.max(mx.abs(mx.matmul(out_layer15.astype(mx.float32), u15_f32)))
        mx.eval(dot_after)
        # In bfloat16 (7-bit mantissa, ~0.0078 relative precision on norm ~10), residual is < 0.1
        self.assertLess(float(dot_after.item()), 0.1)

        # Dynamically switch active_layers to range(1, 64) and alpha without reloading weights
        projector.configure(alpha=0.0, active_layers=range(1, 64), enabled=True)
        out_unprojected = self.model.model.layers[15](x)
        mx.eval(out_unprojected)
        dot_unproj = mx.max(mx.abs(mx.matmul(out_unprojected.astype(mx.float32), u15_f32)))
        mx.eval(dot_unproj)
        self.assertGreater(float(dot_unproj.item()), 8.0)

        # Verify `projector.pulse(...)` context manager (whole-chunk & token_slice windowed pulse)
        with projector.pulse(alpha=1.0, active_layers=range(10, 59), token_slice=(2, 6)):
            out_sliced = self.model.model.layers[15](x)
            mx.eval(out_sliced)
            dots_per_tok = mx.abs(mx.matmul(out_sliced.astype(mx.float32), u15_f32))[0]
            mx.eval(dots_per_tok)
            # Tokens [2..5] inside the pulse slice are orthogonalized (< 0.1)
            self.assertLess(float(mx.max(dots_per_tok[2:6]).item()), 0.1)
            # Tokens [0..1] and [6..7] outside the slice remain unprojected (> 8.0)
            self.assertGreater(float(mx.min(dots_per_tok[:2]).item()), 8.0)
            self.assertGreater(float(mx.min(dots_per_tok[6:]).item()), 8.0)

        # Verify exiting `projector.pulse(...)` automatically restored alpha=0.0 and cleared token_slice
        self.assertEqual(projector.alpha, 0.0)
        self.assertIsNone(projector.token_slice)
        out_after_pulse = self.model.model.layers[15](x)
        mx.eval(out_after_pulse)
        dot_after_pulse = mx.max(mx.abs(mx.matmul(out_after_pulse.astype(mx.float32), u15_f32)))
        mx.eval(dot_after_pulse)
        self.assertGreater(float(dot_after_pulse.item()), 8.0)

        projector.detach(self.model)
        self.assertFalse(hasattr(self.model.model.layers[15], "base_layer"))

        # Verify WeightSpaceProjector apply_in_place and restore_baseline exactness (< 1e-5)
        # First test on an FP32 1-layer model so we verify exact mathematical identity < 1e-5
        class FP32WriterTestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [
                    SyntheticHybridDecoderLayer(layer_idx=0, hidden_size=256),
                    SyntheticHybridDecoderLayer(layer_idx=1, hidden_size=256),
                ]

        fp32_test_model = FP32WriterTestModel()
        fp32_test_model.apply(
            lambda t: t.astype(mx.float32) if mx.issubdtype(t.dtype, mx.floating) else t
        )
        mx.eval(fp32_test_model.parameters())
        u1_f32 = normalize_unit_vector(mx.random.normal((256,)), dtype=mx.float32)
        mx.eval(u1_f32)

        w_orig = mx.array(fp32_test_model.layers[1].mlp.down_proj.weight)
        z_in = mx.random.normal((1, 5, 64), dtype=mx.float32)
        y_orig = fp32_test_model.layers[1].mlp.down_proj(z_in)
        y_manual = y_orig - mx.matmul(y_orig, u1_f32[:, None]) * u1_f32[None, None, :]
        mx.eval(w_orig, y_orig, y_manual)

        ws_fp32 = WeightSpaceProjector(
            model=fp32_test_model,
            vectors={1: u1_f32},
            target_layers=[1],
        )
        n_mod = ws_fp32.apply_in_place(
            alpha=1.0, active_layers=[1], target_components={"mlp"}
        )
        self.assertEqual(n_mod, 1)
        y_ws = fp32_test_model.layers[1].mlp.down_proj(z_in)
        mx.eval(y_ws)
        ws_err = float(mx.max(mx.abs(y_ws - y_manual)).item())
        self.assertLess(
            ws_err,
            1e-5,
            f"WeightSpaceProjector output diverged from manual projection: {ws_err}",
        )

        # Verify restore_baseline restores exact weights (< 1e-5)
        ws_fp32.restore_baseline()
        w_restored = fp32_test_model.layers[1].mlp.down_proj.weight
        mx.eval(w_restored)
        restore_err = float(mx.max(mx.abs(w_restored - w_orig)).item())
        self.assertLess(
            restore_err,
            1e-5,
            f"WeightSpaceProjector restore_baseline diverged from original weight: {restore_err}",
        )

        # Also verify restored linear layer output matches original output to < 1e-5 on FP32
        y_restored_fp32 = fp32_test_model.layers[1].mlp.down_proj(z_in)
        mx.eval(y_restored_fp32)
        y_restore_err = float(mx.max(mx.abs(y_restored_fp32 - y_orig)).item())
        self.assertLess(y_restore_err, 1e-5)

        # Also verify WeightSpaceProjector on the 64-layer BF16 synthetic model and unwrap_model
        projector.attach(self.model)
        unwrapped_model = unwrap_model(self.model)
        self.assertIs(unwrapped_model, self.model)
        self.assertFalse(hasattr(self.model.model.layers[15], "base_layer"))

        w15_bf16_orig = mx.array(self.model.model.layers[15].self_attn.o_proj.weight)
        mx.eval(w15_bf16_orig)
        ws_bf16 = WeightSpaceProjector(
            model=self.model,
            vectors=vectors,
            target_layers=range(10, 59),
        )
        self.assertEqual(
            ws_bf16.apply_in_place(
                alpha=1.0,
                active_layers=range(10, 59),
                target_components={"deltanet", "full_attn", "mlp"},
            ),
            98,
        )
        ws_bf16.restore_baseline()
        w15_bf16_restored = self.model.model.layers[15].self_attn.o_proj.weight
        mx.eval(w15_bf16_restored)
        bf16_restore_err = float(
            mx.max(
                mx.abs(
                    w15_bf16_restored.astype(mx.float32)
                    - w15_bf16_orig.astype(mx.float32)
                )
            ).item()
        )
        # Within 1 BF16 ULP (~6.1e-5)
        self.assertLess(bf16_restore_err, 1e-4)

    def test_03_hybrid_timeline_cache_rewind_snapshot_restore_and_fork_edit(self):
        """
        Verify HybridTimelineCache across all 64 layers (48 Gated DeltaNet ArraysCache + 16 KVCache):
        prefill -> snapshot(edit_idx) -> continue prefill -> fork_and_edit(edit_idx, new_tokens)
        must match a fresh prefill of `prefix[:edit_idx] + new_tokens` to machine precision.
        """
        cache_list = make_prompt_cache(self.model)
        self.assertEqual(len(cache_list), 64)

        timeline = HybridTimelineCache(
            model=self.model, cache=cache_list, prefill_step_size=512
        )
        summary = timeline.inspect()
        self.assertEqual(summary.total_layers, 64)
        self.assertEqual(len(summary.kv_layer_indices), 16)
        self.assertEqual(len(summary.deltanet_layer_indices), 48)
        self.assertEqual(tuple(summary.kv_layer_indices), FULL_ATTENTION_LAYERS_DEFAULT)

        # Original token sequence of length 12, snapshot at token_idx = 5
        original_tokens = [10, 21, 32, 43, 54, 65, 76, 87, 98, 109, 120, 131]
        edit_token_idx = 5
        edited_suffix = [201, 202, 203, 204]

        # Run full prefill on original_tokens while snapshotting at token_idx = 5
        _ = timeline.prefill_with_snapshots(
            original_tokens, snapshot_indices=[edit_token_idx]
        )
        self.assertEqual(timeline.current_offset, len(original_tokens))
        self.assertTrue(timeline.has_snapshot(edit_token_idx))

        # Now fork at edit_token_idx=5 and warm-prefill ONLY edited_suffix
        forked_logits = fork_and_edit(
            cache=timeline,
            edit_token_idx=edit_token_idx,
            new_token_ids=edited_suffix,
            model=self.model,
            prefill_step_size=512,
        )
        mx.eval(forked_logits)
        self.assertEqual(timeline.current_offset, edit_token_idx + len(edited_suffix))

        # Reference ground-truth: fresh model cache prefilled with original_tokens[:5] + edited_suffix
        fresh_timeline = HybridTimelineCache(model=self.model, prefill_step_size=512)
        _ = fresh_timeline.prefill_with_snapshots(original_tokens[:edit_token_idx])
        ref_logits = fresh_timeline.prefill_with_snapshots(edited_suffix)
        mx.eval(ref_logits)

        max_logit_diff = float(
            mx.max(
                mx.abs(
                    forked_logits.astype(mx.float32) - ref_logits.astype(mx.float32)
                )
            ).item()
        )
        self.assertLess(
            max_logit_diff,
            1e-4,
            f"Fork-and-edit logits diverged from clean prefill (max_diff={max_logit_diff})",
        )

        # Verify all 48 DeltaNet states and 16 KV caches match between forked and reference timelines
        for l_idx in timeline.deltanet_layer_indices:
            s_fork = timeline.cache[l_idx][1]
            s_ref = fresh_timeline.cache[l_idx][1]
            diff = float(mx.max(mx.abs(s_fork - s_ref)).item())
            self.assertLess(diff, 1e-5)

        for l_idx in timeline.kv_layer_indices:
            k_fork, v_fork = timeline.cache[l_idx].keys_and_values()
            k_ref, v_ref = fresh_timeline.cache[l_idx].keys_and_values()
            self.assertEqual(k_fork.shape, k_ref.shape)
            k_diff = float(
                mx.max(mx.abs(k_fork.astype(mx.float32) - k_ref.astype(mx.float32))).item()
            )
            self.assertLess(k_diff, 1e-5)

    def test_04_attention_probe_and_end_to_end_experiment_harness(self):
        """
        Verify:
          1. `capture_attention_maps` records post-softmax A^{(l)} for all 16 full-attention
             layers (24 Q heads, 4 KV heads, head_dim 256, partial RoPE 64) and produces
             `< 1e-5` identical layer outputs to the un-wrapped `qwen3_5` Gated Full-Attention
             layer (verifying `q_proj` -> `(queries, gate)` split, `q_norm`/`k_norm`, partial
             RoPE, and post-SDPA `mx.sigmoid(gate)` multiplication).
          2. `resolve_decoder_layers(model)` resolves `model.language_model.model.layers`,
             `model.model.layers`, and `model.layers` while keeping `training=False` (`model.eval()`).
          3. `run_experiment_mode` executes the mid-`<think>` intervention (Clause A prefill,
             rewind/restore to `think_start_idx`, warm-prefill Clause B under alpha=0.0 and
             alpha=1.0 on layers 10..58), highlights routing layers `31, 35, 39, 43, 47`,
             and saves `./outputs/attention_jsd_heatmap.png`.
          4. `download_assets.py` smoke-checks local vector directories and verifies existing
             pinned `Qwen/Qwen3.8-27B` (`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`) directories.
        """
        # 1. Verify exact `< 1e-5` layer output equivalence on real `mlx_lm.models.qwen3_5.DecoderLayer`
        from mlx_lm.models.qwen3_5 import DecoderLayer as Qwen35DecoderLayer
        from mlx_lm.models.qwen3_5 import TextModelArgs as Qwen35TextModelArgs

        qwen_args = Qwen35TextModelArgs(
            model_type="qwen3_5_text",
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=4,
            num_attention_heads=24,
            num_key_value_heads=4,
            head_dim=256,
            full_attention_interval=4,
            rms_norm_eps=1e-6,
            vocab_size=256,
            max_position_embeddings=4096,
            rope_parameters={
                "type": "default",
                "rope_theta": 100000.0,
                "partial_rotary_factor": 0.25,
            },
        )
        # Layer 3 is a Gated Full-Attention layer ((3 + 1) % 4 == 0)
        real_fa_layer = Qwen35DecoderLayer(qwen_args, layer_idx=3)
        real_fa_layer.eval()
        mx.eval(real_fa_layer.parameters())

        class MiniGatedAttnHost(nn.Module):
            def __init__(self, layer):
                super().__init__()
                self.layers = [layer]

        host = MiniGatedAttnHost(real_fa_layer)
        host.eval()
        x_test = mx.random.normal((1, 6, 512), dtype=mx.float32)
        mask_test = create_attention_mask(x_test, None)
        mx.eval(x_test)

        kv_fused = KVCache()
        out_fused = real_fa_layer(x_test, mask=mask_test, cache=kv_fused)
        mx.eval(out_fused)

        kv_unfused = KVCache()
        with capture_attention_maps(host, layers=[0]) as mini_probe:
            out_unfused = real_fa_layer(x_test, mask=mask_test, cache=kv_unfused)
            mx.eval(out_unfused)

        layer_max_diff = float(mx.max(mx.abs(out_fused - out_unfused)).item())
        self.assertLess(
            layer_max_diff,
            1e-5,
            f"UnfusedAttentionProbeWrapper output diverged from un-wrapped qwen3_5 Gated Full-Attention layer: {layer_max_diff}",
        )
        self.assertEqual(mini_probe[0].shape, (1, 24, 6, 6))

        # 2. Check full 64-layer synthetic model attention capture
        tokens = mx.array([[11, 22, 33, 44, 55, 66]], dtype=mx.int32)
        with capture_attention_maps(self.model) as probe:
            logits_unfused = self.model(tokens)
            mx.eval(logits_unfused)

        self.assertEqual(
            sorted(probe.attention_maps.keys()),
            list(FULL_ATTENTION_LAYERS_DEFAULT),
        )
        for l_idx, attn_mat in probe.attention_maps.items():
            # Shape: (B=1, H_q=24, Q_len=6, K_len=6)
            self.assertEqual(attn_mat.shape, (1, 24, 6, 6))
            row_sums = mx.sum(attn_mat, axis=-1)
            max_row_err = float(mx.max(mx.abs(row_sums - 1.0)).item())
            self.assertLess(max_row_err, 1e-4)

        # 3. Verify `resolve_decoder_layers` on nested VLM/MTP (`model.language_model.model.layers`)
        #    and verify `training=False` (`model.eval()`) enforcement for GatedDeltaNet Metal kernel path
        class NestedVLMWrapper(nn.Module):
            def __init__(self, inner_lm):
                super().__init__()
                self.language_model = inner_lm

            def __call__(self, inputs: mx.array, cache: Optional[Any] = None):
                return self.language_model(inputs, cache=cache)

        vlm_wrapped = NestedVLMWrapper(self.model)
        vlm_wrapped.train()  # Intentionally set training=True to test automatic .eval() enforcement
        resolved = resolve_decoder_layers(vlm_wrapped)
        self.assertEqual(len(resolved), 64)
        self.assertFalse(vlm_wrapped.training)
        self.assertTrue(all(not l.training for l in resolved))

        # 4. Verify mid-`<think>` experiment mode (Clause A prefill, fork at `think_start_idx`,
        #    Branch 1 alpha=0.0 vs Branch 2 alpha=1.0 on layers 10..58, routing highlights, and heatmap)
        vectors = {
            l: normalize_unit_vector(
                mx.random.normal((HIDDEN_SIZE_DEFAULT,)), dtype=mx.bfloat16
            )
            for l in range(1, NUM_LAYERS_DEFAULT)
        }
        projector = SubspaceProjector(
            vectors=vectors,
            alpha=1.0,
            active_layers=range(10, 59),
            enabled=True,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            heatmap_file = tmp_path / "outputs" / "attention_jsd_heatmap.png"
            routing_plot_file = tmp_path / "outputs" / "top_heads_token_routing.png"
            think_2x2_plot_file = tmp_path / "outputs" / "think_2x2_lookback_mass.png"

            # Prefix up to `<think>\n` (6 tokens), Clause A (8 tokens), Edited Clause B (8 tokens)
            prefix_think_tokens = [1, 2, 3, 4, 150, 151]
            clause_a_tokens = [10, 11, 12, 13, 14, 15, 16, 17]
            edited_clause_b_tokens = [20, 21, 22, 23, 24, 25, 26, 27]

            exp_result = run_experiment_mode(
                model=vlm_wrapped,
                projector=projector,
                prefix_tokens=prefix_think_tokens,
                edited_tokens=edited_clause_b_tokens,
                clause_a_tokens=clause_a_tokens,
                active_layers=range(10, 59),
                prefill_step_size=512,
                top_k_heads=12,
                heatmap_path=heatmap_file,
                token_routing_plot_path=routing_plot_file,
                think_2x2_plot_path=think_2x2_plot_file,
                top_n_routing_heads=5,
                continuation_steps=4,
            )
            self.assertEqual(exp_result["think_start_idx"], len(prefix_think_tokens))
            self.assertEqual(exp_result["clause_a_len"], len(clause_a_tokens))
            self.assertEqual(exp_result["edited_len"], len(edited_clause_b_tokens))
            self.assertEqual(exp_result["active_layers"], list(range(10, 59)))
            self.assertEqual(len(exp_result["ranked_heads"]), 16 * 24)
            self.assertEqual(len(exp_result["token_attributions"]), 5)
            self.assertEqual(len(exp_result["layer_band_ablations"]), 4)
            self.assertEqual(
                [r["num_layers"] for r in exp_result["layer_band_ablations"]],
                [49, 20, 18, 5],
            )
            self.assertIn("Full GLP-49 Band", exp_result["layer_band_ablation_table"])
            self.assertIn("Sparse Full-Attention Only", exp_result["layer_band_ablation_table"])
            self.assertEqual(len(exp_result["component_writer_ablations"]), 5)
            self.assertEqual(
                [r["matrices_orthogonalized"] for r in exp_result["component_writer_ablations"]],
                ["49 (hook)", 98, 49, 86, 12],
            )
            self.assertIn("Weight-Space All Writers", exp_result["component_writer_ablation_table"])
            self.assertIn("Weight-Space Gated DeltaNet + MLP Only", exp_result["component_writer_ablation_table"])
            self.assertEqual(
                [b["key"] for b in exp_result["think_2x2_branches"]],
                ["A0", "A1", "B0", "B1_pulse"],
            )
            for b in exp_result["think_2x2_branches"]:
                total_mass = (
                    b["lookback_mass"]["mean_prompt_mass"]
                    + b["lookback_mass"]["mean_local_think_mass"]
                )
                self.assertAlmostEqual(total_mass, 1.0, places=3)
                self.assertEqual(len(b["continuation_token_ids"]), 4)
                self.assertAlmostEqual(b["cosine_similarities"][b["key"]], 1.0, places=4)
            self.assertIn("[A0]", exp_result["think_2x2_summary_table"])
            self.assertIn("[B1_pulse]", exp_result["think_2x2_summary_table"])
            self.assertTrue(think_2x2_plot_file.exists())
            self.assertGreater(think_2x2_plot_file.stat().st_size, 1000)
            self.assertGreater(exp_result["mean_jsd"], 0.0)
            self.assertGreater(exp_result["mean_l1"], 0.0)
            self.assertGreater(exp_result["clause_a_vs_b_mean_jsd"], 0.0)
            self.assertEqual(
                sorted(exp_result["routing_layer_summary"].keys()),
                list(ROUTING_HIGHLIGHT_LAYERS),
            )
            self.assertIn("Routing Band", exp_result["table"])
            self.assertIn("Peak Query q*", exp_result["token_routing_table"])
            self.assertTrue(heatmap_file.exists())
            self.assertGreater(heatmap_file.stat().st_size, 1000)
            self.assertTrue(routing_plot_file.exists())
            self.assertGreater(routing_plot_file.stat().st_size, 1000)

            # 5. Verify `download_assets.py` vector smoke check and pinned model verification
            vec_dir = tmp_path / "vectors"
            vec_dir.mkdir(parents=True, exist_ok=True)
            mx.save_safetensors(
                str(vec_dir / "directions.safetensors"),
                {f"direction.{l}": v for l, v in vectors.items()},
            )
            smoke_info = smoke_check_vectors(vec_dir, hidden_size=HIDDEN_SIZE_DEFAULT)
            self.assertEqual(smoke_info["active_layers"], list(range(1, 64)))
            self.assertEqual(smoke_info["shape"], (HIDDEN_SIZE_DEFAULT,))

            mock_model_dir = tmp_path / "Qwen3.8-27B-bf16"
            mock_model_dir.mkdir(parents=True, exist_ok=True)
            (mock_model_dir / "config.json").write_text('{"model_type": "qwen3_5"}')
            mx.save_safetensors(
                str(mock_model_dir / "model-00001-of-00001.safetensors"),
                {"dummy": mx.zeros((4,), dtype=mx.bfloat16)},
            )
            model_check = pull_or_verify_model(
                model_dir=mock_model_dir,
                verify_only_if_exists=True,
            )
            self.assertEqual(model_check["status"], "verified_existing")
            self.assertEqual(PINNED_QWEN38_REPO_ID, "Qwen/Qwen3.8-27B")
            self.assertEqual(
                PINNED_QWEN38_REVISION, "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
            )
            self.assertEqual(
                MODEL_ALLOW_PATTERNS,
                ["*.json", "*.safetensors", "tokenizer*", "merges.txt", "vocab.json"],
            )

        projector.detach(vlm_wrapped)

        # Verify benchmark mode on the synthetic model (with warmup and multi-length prefill)
        bench_result = run_benchmark_mode(
            model=self.model,
            prompt_tokens=[1, 2, 3, 4, 5, 6, 7, 8],
            edit_token_idx=4,
            edited_suffix_tokens=[9, 10, 11],
            prefill_step_size=512,
            decode_steps=4,
            prefill_lengths=(8, 16),
        )
        self.assertGreater(bench_result["warmup_page_in_ms"], 0.0)
        self.assertGreater(bench_result["cold_prefill_tok_s"], 0.0)
        self.assertIn("len_8", bench_result["cold_prefill_by_length"])
        self.assertIn("len_16", bench_result["cold_prefill_by_length"])
        self.assertGreater(bench_result["warm_edit_refeed_ms"], 0.0)
        self.assertGreater(bench_result["decode_tok_s"], 0.0)
        self.assertEqual(len(bench_result["decoded_token_ids"]), 4)
        self.assertEqual(
            SAFE_ALLOW_PATTERNS, ["*.safetensors", "*.gguf", "*.json"]
        )

    def test_05_harness_server_fastapi_endpoints_and_stride8_fork_exactness(self):
        """
        Verify `harness_server.py` FastAPI endpoints via `TestClient`:
          1. `GET /` serves the dark-mode Single-Page Workbench UI (`200 OK`).
          2. `GET /v1/models` and `POST /v1/chat/completions` (including `think_gated=True`
             auto-releasing `alpha` to `0.0` when `</think>` is emitted).
          3. `POST /api/prefill_and_probe` with `snapshot_stride=8` records dense Gated DeltaNet
             snapshots and returns per-token 5-Head Prompt-Lookback Scores + `peak_spike_token`.
          4. `POST /api/fork_and_edit` at an arbitrary non-snapshot index (`fork_token_idx=11`,
             nearest stride-8 snapshot `6` + `5`-token delta replay) matches a clean full
             prefill of `prefix[:11] + edited_tokens` to `< 1e-5`!
        """
        from fastapi.testclient import TestClient
        from attention_probe import compute_preplan_and_anchor_rhythm
        from harness_server import (
            SandboxEngine,
            SimpleFallbackTokenizer,
            create_harness_app,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            spi_log_file = Path(tmp_dir) / "spi_traces.jsonl"

            vectors = {
                l: normalize_unit_vector(
                    mx.random.normal((HIDDEN_SIZE_DEFAULT,)), dtype=mx.bfloat16
                )
                for l in range(1, NUM_LAYERS_DEFAULT)
            }
            projector = SubspaceProjector(
                vectors=vectors,
                alpha=0.0,
                active_layers=range(10, 59),
                enabled=False,
            )
            tokenizer = SimpleFallbackTokenizer(vocab_size=256)
            engine = SandboxEngine(
                model=self.model,
                tokenizer=tokenizer,
                projector=projector,
                model_id="Qwen3.8-27B-synthetic-test",
                prefill_step_size=512,
                snapshot_stride=8,
                spi_log_path=spi_log_file,
            )
            app = create_harness_app(engine)
            client = TestClient(app)

            # 1. Verify GET / Single-Page Workbench UI (including 4-mode Telemetry Heatmap bar & SPI Gold button)
            r_root = client.get("/")
            self.assertEqual(r_root.status_code, 200)
            self.assertIn("Qwen3.8-27B Stateful", r_root.text)
            self.assertIn("/api/prefill_and_probe", r_root.text)
            self.assertIn("/api/fork_and_edit", r_root.text)
            self.assertIn("Preplan Rhythm (WAAD Onsets)", r_root.text)
            self.assertIn("Anchor Influence (FAI Hubs)", r_root.text)
            self.assertIn("Mark as Gold SPI Pair", r_root.text)

            # 2. Verify GET /v1/models
            r_models = client.get("/v1/models")
            self.assertEqual(r_models.status_code, 200)
            self.assertEqual(r_models.json()["data"][0]["id"], "Qwen3.8-27B-synthetic-test")

            # 3. Verify POST /api/prefill_and_probe with stride-8 dense snapshotting + rhythm telemetry
            prompt_ids = [1, 2, 3, 4, 150, 151]  # think_start_idx = 6
            think_ids = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]  # 18 tokens -> total 24
            r_probe = client.post(
                "/api/prefill_and_probe",
                json={
                    "prompt_token_ids": prompt_ids,
                    "think_token_ids": think_ids,
                    "alpha": 0.0,
                    "snapshot_stride": 8,
                },
            )
            self.assertEqual(r_probe.status_code, 200)
            probe_data = r_probe.json()
            self.assertEqual(probe_data["think_start_idx"], 6)
            self.assertEqual(probe_data["total_tokens"], 24)
            self.assertEqual(probe_data["snapshots"], [6, 14, 22, 24])
            self.assertEqual(len(probe_data["tokens"]), 24)
            self.assertIn("peak_spike_token", probe_data)
            self.assertIn("peak_lookback_token_idx", probe_data)
            self.assertIn("peak_preplan_token_idx", probe_data)
            self.assertIn("peak_anchor_token_idx", probe_data)
            self.assertIn("coupled_junctures", probe_data)
            self.assertEqual(probe_data["rhythm_summary"]["num_total_heads"], 384)
            self.assertEqual(probe_data["rhythm_summary"]["num_local_heads"], 192)
            self.assertEqual(probe_data["rhythm_summary"]["num_global_heads"], 192)
            for tok_obj in probe_data["tokens"]:
                for req_field in (
                    "lookback_score",
                    "preplan_score",
                    "waad_raw",
                    "anchor_score",
                    "fai_raw",
                    "is_coupled_juncture",
                ):
                    self.assertIn(req_field, tok_obj)
            self.assertTrue(probe_data["peak_spike_token"]["is_think"])
            self.assertEqual(
                sorted(probe_data["tail_16_summary"]["per_head"].keys()),
                ["L35.H21", "L39.H0", "L39.H1", "L43.H1", "L43.H7"],
            )

            # 4. Verify arbitrary-index stride-8 snapshot + delta replay exactness (< 1e-5)
            # Fork at arbitrary token index 11 (between snapshot 6 and snapshot 14 -> 5 delta tokens replayed)
            fork_idx = 11
            edited_ids = [40, 41, 42, 43, 44, 45]
            r_fork = client.post(
                "/api/fork_and_edit",
                json={
                    "fork_token_idx": fork_idx,
                    "edited_token_ids": edited_ids,
                    "alpha": 0.0,
                    "think_gated": True,
                    "max_new_tokens": 0,
                    "capture_probe": True,
                },
            )
            self.assertEqual(r_fork.status_code, 200)
            fork_data = r_fork.json()
            self.assertEqual(fork_data["fork_token_idx"], 11)
            self.assertEqual(fork_data["snapshot_restored_idx"], 6)
            self.assertEqual(fork_data["delta_tokens_replayed"], 5)
            self.assertEqual(fork_data["edited_tokens_count"], 6)
            self.assertEqual(len(fork_data["edited_tokens"]), 6)
            self.assertEqual(len(fork_data["forked_tokens"]), 6)
            self.assertIn("spi_trace_id", fork_data)
            self.assertEqual(fork_data["spi_traces_count"], 1)

            # Compare engine's hybrid cache state after stride-8 snapshot (at 6) + 5-token delta replay (to 11) + 6 edited tokens
            # against a reference timeline with a direct snapshot at fork_idx=11
            projector.detach(self.model)
            clean_timeline = HybridTimelineCache(model=self.model, prefill_step_size=512)
            _ = clean_timeline.prefill_with_snapshots(prompt_ids, snapshot_indices=[6])
            _ = clean_timeline.prefill_with_snapshots(think_ids[:5], snapshot_indices=[fork_idx])
            with capture_attention_maps(self.model):
                _ = clean_timeline.fork_and_edit(edit_token_idx=fork_idx, new_token_ids=edited_ids)

            for l_idx in engine.timeline.deltanet_layer_indices:
                s_fork = engine.timeline.cache[l_idx][1]
                s_clean = clean_timeline.cache[l_idx][1]
                diff = float(mx.max(mx.abs(s_fork - s_clean)).item())
                self.assertLess(
                    diff,
                    1e-5,
                    f"Stride-8 snapshot + delta replay DeltaNet state diverged at layer {l_idx}: {diff}",
                )

            for l_idx in engine.timeline.kv_layer_indices:
                k_fork, v_fork = engine.timeline.cache[l_idx].keys_and_values()
                k_clean, v_clean = clean_timeline.cache[l_idx].keys_and_values()
                k_diff = float(
                    mx.max(mx.abs(k_fork.astype(mx.float32) - k_clean.astype(mx.float32))).item()
                )
                self.assertLess(k_diff, 1e-5)

            # Re-attach projector for organic prefill, chained continuation fork, SPI logger, and chat completion tests
            projector.attach(self.model)

            # 5. Verify POST /api/prefill_and_probe in Organic Generation Mode (`clause_a_text=""`, `generate_organic_tokens=12`)
            r_organic = client.post(
                "/api/prefill_and_probe",
                json={
                    "prompt_token_ids": prompt_ids,
                    "clause_a_text": "",
                    "generate_organic_tokens": 12,
                    "organic_alpha": 0.0,
                    "snapshot_stride": 8,
                },
            )
            self.assertEqual(r_organic.status_code, 200)
            org_data = r_organic.json()
            self.assertTrue(org_data["organic_generated"])
            self.assertEqual(org_data["think_start_idx"], 6)
            self.assertEqual(org_data["think_tokens"], 12)
            self.assertEqual(org_data["total_tokens"], 18)
            # Stride-8 snapshots from think_start_idx=6 over 12 generated tokens -> [6, 14, 18]
            self.assertEqual(org_data["snapshots"], [6, 14, 18])
            self.assertEqual(len(org_data["tokens"]), 18)
            for idx_i, tok_rec in enumerate(org_data["tokens"]):
                self.assertEqual(tok_rec["token_idx"], idx_i)
                self.assertIn("mean_lookback_score", tok_rec)
                self.assertIn("preplan_score", tok_rec)
                self.assertIn("anchor_score", tok_rec)

            # 6. Verify POST /api/fork_and_edit returns clickable `continuation_tokens` with valid sequential `token_idx`
            #    and rhythm scores, AND supports a second chained `POST /api/fork_and_edit` inside `continuation_tokens`!
            r_fork_cont = client.post(
                "/api/fork_and_edit",
                json={
                    "fork_token_idx": 11,
                    "edited_token_ids": [40, 41, 42, 43],
                    "alpha": 1.0,
                    "think_gated": True,
                    "max_new_tokens": 6,
                    "capture_probe": True,
                },
            )
            self.assertEqual(r_fork_cont.status_code, 200)
            fc_data = r_fork_cont.json()
            self.assertEqual(fc_data["fork_token_idx"], 11)
            self.assertEqual(fc_data["edited_tokens_count"], 4)
            self.assertEqual(fc_data["continuation_tokens_count"], 6)
            self.assertIn("peak_lookback_token_idx", fc_data)
            self.assertIn("peak_preplan_token_idx", fc_data)
            self.assertIn("peak_anchor_token_idx", fc_data)
            self.assertIn("coupled_junctures", fc_data)
            cont_tokens = fc_data["continuation_tokens"]
            self.assertEqual(len(cont_tokens), 6)
            expected_cont_indices = [15, 16, 17, 18, 19, 20]
            self.assertEqual([ct["token_idx"] for ct in cont_tokens], expected_cont_indices)
            for ct in cont_tokens:
                for req_field in (
                    "lookback_score",
                    "mean_lookback_score",
                    "preplan_score",
                    "waad_raw",
                    "anchor_score",
                    "fai_raw",
                    "is_coupled_juncture",
                    "head_scores",
                    "is_think_end",
                ):
                    self.assertIn(req_field, ct)

            # Chained (multi-hop) fork directly at `token_idx=18` INSIDE the decoded continuation stream!
            chained_fork_idx = cont_tokens[3]["token_idx"]  # 18
            self.assertEqual(chained_fork_idx, 18)
            r_chained = client.post(
                "/api/fork_and_edit",
                json={
                    "fork_token_idx": chained_fork_idx,
                    "edited_token_ids": [90, 91, 92],
                    "alpha": 0.5,
                    "think_gated": True,
                    "max_new_tokens": 4,
                    "capture_probe": True,
                },
            )
            self.assertEqual(r_chained.status_code, 200)
            ch_data = r_chained.json()
            self.assertEqual(ch_data["fork_token_idx"], 18)
            self.assertEqual(ch_data["edited_tokens_count"], 3)
            self.assertEqual(
                [et["token_idx"] for et in ch_data["edited_tokens"]],
                [18, 19, 20],
            )
            self.assertEqual(
                [ct["token_idx"] for ct in ch_data["continuation_tokens"]],
                [21, 22, 23, 24],
            )

            # 7. Verify Automatic Counterfactual Trajectory Logger (`spi_traces.jsonl`) & `/api/spi_traces` + `/api/spi_traces/label`
            self.assertTrue(spi_log_file.exists())
            r_spi_list = client.get("/api/spi_traces")
            self.assertEqual(r_spi_list.status_code, 200)
            spi_list_data = r_spi_list.json()
            # 3 fork_and_edit calls were executed above -> 3 counterfactual traces logged
            self.assertEqual(spi_list_data["count"], 3)
            latest_trace = spi_list_data["traces"][0]
            self.assertEqual(latest_trace["trace_id"], ch_data["spi_trace_id"])
            self.assertEqual(latest_trace["fork_token_idx"], 18)
            self.assertEqual(latest_trace["label"], "candidate")
            for k in (
                "attention_jsd_shift",
                "lookback_mass_delta",
                "preplan_mean_delta",
                "anchor_mean_delta",
                "fork_edit_ms",
            ):
                self.assertIn(k, latest_trace["causal_deltas"])
            self.assertIn("telemetry_at_fork", latest_trace["rejected_branch"])
            self.assertIn("telemetry_summary", latest_trace["repaired_branch"])

            # Mark latest trace as Gold SPI Pair ("accepted") via POST /api/spi_traces/label
            r_spi_label = client.post(
                "/api/spi_traces/label",
                json={"trace_id": ch_data["spi_trace_id"], "label": "accepted"},
            )
            self.assertEqual(r_spi_label.status_code, 200)
            lbl_data = r_spi_label.json()
            self.assertTrue(lbl_data["updated"])
            self.assertEqual(lbl_data["label"], "accepted")
            r_spi_after = client.get("/api/spi_traces").json()
            self.assertEqual(r_spi_after["traces"][0]["label"], "accepted")

            # Also verify POST /api/promote_to_baseline promotes the chained stream (25 tokens) to Left Panel baseline
            r_promote = client.post("/api/promote_to_baseline")
            self.assertEqual(r_promote.status_code, 200)
            prom_data = r_promote.json()
            self.assertTrue(prom_data["promoted"])
            self.assertEqual(prom_data["total_tokens"], 25)
            self.assertEqual(len(prom_data["tokens"]), 25)

            # 8. Verify POST /v1/chat/completions with `think_gated=True` auto-release on `</think>`
            orig_check = engine._check_think_closed
            engine._check_think_closed = lambda gen_ids, txt: len(gen_ids) >= 2
            try:
                r_chat = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "Qwen3.8-27B-synthetic-test",
                        "messages": [{"role": "user", "content": "Derive Gated DeltaNet state."}],
                        "max_tokens": 5,
                        "alpha": 1.0,
                        "think_gated": True,
                    },
                )
                self.assertEqual(r_chat.status_code, 200)
                chat_data = r_chat.json()
                self.assertEqual(chat_data["object"], "chat.completion")
                telem = chat_data["sandbox_telemetry"]
                self.assertTrue(telem["think_gated"])
                self.assertEqual(telem["think_gated_released_at_step"], 2)
                # Steps 0 and 1 run at alpha=1.0; steps 2, 3, 4 run at alpha=0.0 after </think> release!
                self.assertEqual(telem["alpha_schedule"], [1.0, 1.0, 0.0, 0.0, 0.0])
            finally:
                engine._check_think_closed = orig_check
                projector.detach(self.model)

    def test_06_preplan_and_anchor_rhythm_telemetry(self):
        """
        Verify `compute_preplan_and_anchor_rhythm` in `attention_probe.py`:
          1. Sink-free `<think>` submatrix normalization over `j <= i`.
          2. Automatic Local vs. Global head partitioning (192 local vs 192 global heads across 384 heads).
          3. Preplan (`WAAD_t` + `Delta WAAD_t`) peak detection, Anchor (`FAI_s`) hub detection,
             and Coupled `Preplan -> Anchor` juncture identification (`1..6` tokens apart).
        """
        from attention_probe import (
            FULL_ATTENTION_LAYERS_DEFAULT,
            compute_preplan_and_anchor_rhythm,
        )

        think_start_idx = 6
        m_len = 24
        total_len = think_start_idx + m_len
        n_heads_per_layer = 24

        # Construct synthetic 16-layer x 24-head attention maps with:
        #   - First 8 layers (192 heads) acting as Locally Focused Heads (attending mostly to i or i-1,
        #     except at relative token p_rel=8 where they reach back to j=0 -> sharp WAAD_t preplan spike!)
        #   - Last 8 layers (192 heads) acting as Globally Focused Heads (attending mostly to long-range
        #     tokens, with future tokens i in [12..22] strongly attending to anchor token a_rel=10!)
        synth_attn: Dict[int, mx.array] = {}
        for idx_l, l_idx in enumerate(FULL_ATTENTION_LAYERS_DEFAULT):
            mat_np = np.zeros((1, n_heads_per_layer, m_len, total_len), dtype=np.float32)
            is_local_layer = idx_l < 8
            for i in range(m_len):
                abs_i = think_start_idx + i
                if is_local_layer:
                    # Local head: normally attends to self (distance 0), but at i == 8 reaches back to j == 0 (distance 8)
                    if i == 8:
                        mat_np[:, :, i, think_start_idx + 0] = 0.95
                        mat_np[:, :, i, abs_i] = 0.05
                    else:
                        mat_np[:, :, i, abs_i] = 0.90
                        if i > 0:
                            mat_np[:, :, i, abs_i - 1] = 0.10
                else:
                    # Global head: future tokens i >= 12 strongly attend to anchor a_rel = 10
                    if i >= 12:
                        mat_np[:, :, i, think_start_idx + 10] = 0.80
                        mat_np[:, :, i, think_start_idx + 0] = 0.20
                    else:
                        mat_np[:, :, i, think_start_idx + 0] = 0.85
                        mat_np[:, :, i, abs_i] = 0.15
            synth_attn[l_idx] = mx.array(mat_np)

        rhythm = compute_preplan_and_anchor_rhythm(
            attn_maps=synth_attn,
            think_start_idx=think_start_idx,
            query_offset=think_start_idx,
        )

        self.assertEqual(rhythm["num_total_heads"], 384)
        self.assertEqual(rhythm["num_local_heads"], 192)
        self.assertEqual(rhythm["num_global_heads"], 192)
        self.assertEqual(len(rhythm["preplan_scores"]), m_len)
        self.assertEqual(len(rhythm["anchor_scores"]), m_len)

        # Verify all normalized scores are in [0, 1]
        for s in rhythm["preplan_scores"]:
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)
        for s in rhythm["anchor_scores"]:
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)

        # Peak preplan token must be at think_start_idx + 8 = 14
        self.assertEqual(rhythm["peak_preplan_token_idx"], think_start_idx + 8)
        # Peak anchor token must be at think_start_idx + 10 = 16
        self.assertEqual(rhythm["peak_anchor_token_idx"], think_start_idx + 10)

        # Verify coupled Preplan (14) -> Anchor (16) juncture (distance = 2 tokens in [1..6]) is detected!
        juncture_pairs = [
            (j["preplan_idx"], j["anchor_idx"]) for j in rhythm["coupled_junctures"]
        ]
        self.assertIn((14, 16), juncture_pairs)
        self.assertIn(14, rhythm["coupled_token_indices"])
        self.assertIn(16, rhythm["coupled_token_indices"])

        # 2. Regression Test: 120-token sequence with planted Preplan->Anchor transitions at
        #    early (`i=15`), mid (`i=55`), and late (`i=90`) positions, plus recency noise at the tail (`i=115..119`).
        m_long = 120
        total_long = think_start_idx + m_long
        planted_preplans = (15, 55, 90)
        planted_anchors = (17, 57, 92)
        long_attn: Dict[int, mx.array] = {}

        for idx_l, l_idx in enumerate(FULL_ATTENTION_LAYERS_DEFAULT):
            mat_np = np.zeros((1, n_heads_per_layer, m_long, total_long), dtype=np.float32)
            is_local_layer = idx_l < 8
            for i in range(m_long):
                abs_i = think_start_idx + i
                if is_local_layer:
                    # Local head: at planted preplan onsets (15, 55, 90), reach back 14 tokens;
                    # otherwise attend to self/immediate neighbor.
                    if i in planted_preplans:
                        back_j = max(0, i - 14)
                        mat_np[:, :, i, think_start_idx + back_j] = 0.92
                        mat_np[:, :, i, abs_i] = 0.08
                    else:
                        mat_np[:, :, i, abs_i] = 0.88
                        if i > 0:
                            mat_np[:, :, i, abs_i - 1] = 0.12
                else:
                    # Global head: simulate 1/(i+1) diluted causal baseline + strong future routing
                    # to planted anchors (17, 57, 92) over their future windows, PLUS strong short-range
                    # recency attention on the final tail tokens (115..119) to test the tail-horizon guard!
                    mat_np[:, :, i, think_start_idx : abs_i + 1] = 0.25 / float(i + 1)
                    if 21 <= i <= 38:
                        mat_np[:, :, i, think_start_idx + 17] += 0.55
                    elif 61 <= i <= 78:
                        mat_np[:, :, i, think_start_idx + 57] += 0.55
                    elif 96 <= i <= 112:
                        mat_np[:, :, i, think_start_idx + 92] += 0.55
                    elif i >= 115:
                        # Tail tokens attending heavily to 2-3 tokens prior (recency trap)
                        mat_np[:, :, i, think_start_idx + max(0, i - 2)] += 0.70
            long_attn[l_idx] = mx.array(mat_np)

        rhythm_long = compute_preplan_and_anchor_rhythm(
            attn_maps=long_attn,
            think_start_idx=think_start_idx,
            query_offset=think_start_idx,
        )
        long_pairs = [
            (j["preplan_idx"] - think_start_idx, j["anchor_idx"] - think_start_idx)
            for j in rhythm_long["coupled_junctures"]
        ]
        # Verify early (15->17), mid (55->57), and late (90->92) junctures are detected!
        self.assertIn((15, 17), long_pairs)
        self.assertIn((55, 57), long_pairs)
        self.assertIn((90, 92), long_pairs)

        # Verify the final 5 tail tokens (`i=115..119`) are NOT artificially inflated in `anchor_scores`
        # and do NOT produce any false coupled junctures at the tail!
        tail_5_anchor_scores = rhythm_long["anchor_scores"][115:120]
        for tail_s in tail_5_anchor_scores:
            self.assertLess(
                tail_s,
                0.25,
                f"Tail token anchor_score was artificially inflated: {tail_5_anchor_scores}",
            )
        for p_rel, a_rel in long_pairs:
            self.assertLess(p_rel, 115)
            self.assertLess(a_rel, 115)


if __name__ == "__main__":
    unittest.main(verbosity=2)
