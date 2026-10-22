"""
subspace_projector.py — Rank-1 Orthogonal Projection Hook for Qwen3.8-27B (MLX)

Supports loading per-layer unit steering/ablation vectors from local .safetensors
or .gguf files (including `direction.{l}` keys and PEFT `lora_B` tensors), normalizing
to unit L2 norm in mx.bfloat16, and dynamically applying:
    h <- h - alpha * (h . u_hat_l) * u_hat_l
on residual stream activations of layers 1..63.
"""

from __future__ import annotations

import os
import re
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np


NUM_LAYERS_DEFAULT = 64
HIDDEN_SIZE_DEFAULT = 5120


def _extract_layer_index(key: str) -> Optional[int]:
    """
    Extract layer index from tensor keys such as:
      - "direction.15" or "directions.15" or "layer.15.direction"
      - "model.layers.15.self_attn.o_proj.lora_B.weight"
      - "model.layers.15.linear_attn.out_proj.lora_B.weight"
      - "layers.15.mlp.down_proj.lora_B"
      - "blk.15.direction"
    """
    patterns = [
        r"(?:^|\.)direction(?:s)?\.(\d+)(?:\.|$)",
        r"(?:^|\.)layers\.(\d+)\.",
        r"(?:^|\.)blk\.(\d+)\.",
        r"(?:^|\.)layer_(\d+)(?:\.|$)",
        r"^(\d+)$",
    ]
    for pat in patterns:
        m = re.search(pat, key)
        if m is not None:
            return int(m.group(1))
    return None


def _coerce_to_1d_vector(
    arr: mx.array,
    hidden_size: int = HIDDEN_SIZE_DEFAULT,
    key_name: str = "",
) -> mx.array:
    """
    Convert a loaded tensor (either 1D direction or 2D rank-1/low-rank lora_B tensor)
    into a 1D vector of shape (hidden_size,).
    """
    if arr.ndim == 1:
        if arr.shape[0] != hidden_size:
            raise ValueError(
                f"Tensor '{key_name}' has 1D shape {arr.shape}, expected ({hidden_size},)."
            )
        return arr

    if arr.ndim == 2:
        # Handle rank-1 (or multi-rank where we take the principal column/row) lora_B tensors
        if arr.shape == (hidden_size, 1):
            return arr[:, 0]
        if arr.shape == (1, hidden_size):
            return arr[0, :]
        if arr.shape[0] == hidden_size and arr.shape[1] > 1:
            # Extract leading singular vector or first rank column
            return arr[:, 0]
        if arr.shape[1] == hidden_size and arr.shape[0] > 1:
            return arr[0, :]

    # Squeeze singleton dimensions if possible
    squeezed = mx.squeeze(arr)
    if squeezed.ndim == 1 and squeezed.shape[0] == hidden_size:
        return squeezed

    raise ValueError(
        f"Cannot coerce tensor '{key_name}' with shape {arr.shape} to ({hidden_size},)."
    )


def normalize_unit_vector(
    v: mx.array,
    dtype: mx.Dtype = mx.bfloat16,
    eps: float = 1e-12,
) -> mx.array:
    """
    Normalize a 1D vector to unit L2 length in FP32 for numerical stability,
    then cast to target activation dtype (default: mx.bfloat16).
    """
    v_fp32 = v.astype(mx.float32)
    norm = mx.sqrt(mx.sum(v_fp32 * v_fp32))
    norm_val = float(norm.item())
    if norm_val < eps:
        raise ValueError(f"Cannot normalize near-zero vector (L2 norm = {norm_val:.3e}).")
    u_fp32 = v_fp32 / norm
    u_cast = u_fp32.astype(dtype)
    mx.eval(u_cast)
    return u_cast


def load_subspace_vectors(
    path: Union[str, Path],
    hidden_size: int = HIDDEN_SIZE_DEFAULT,
    dtype: mx.Dtype = mx.bfloat16,
) -> Dict[int, mx.array]:
    """
    Load per-layer steering/projection vectors from a local .safetensors or .gguf file.

    Supports:
      1. Standard direction keys (`direction.{l}`, `layers.{l}.direction`, etc.)
      2. PEFT LoRA B matrices (`*layers.{l}*lora_B*`)

    Returns:
      Dict[layer_idx, unit_vector_bf16] where each vector has shape (hidden_size,)
      and unit L2 norm.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Vector file not found: {path}")

    raw_tensors: Dict[str, mx.array] = {}
    suffix = path.suffix.lower()

    if suffix == ".safetensors":
        loaded = mx.load(str(path))
        if isinstance(loaded, dict):
            raw_tensors = loaded
        else:
            raise ValueError(f"Unexpected return format from mx.load on {path}")
    elif suffix == ".gguf":
        try:
            loaded = mx.load(str(path))
            if isinstance(loaded, tuple) and len(loaded) >= 1 and isinstance(loaded[0], dict):
                raw_tensors = loaded[0]
            elif isinstance(loaded, dict):
                raw_tensors = loaded
        except Exception:
            # Fallback to python `gguf` reader if needed
            import gguf

            reader = gguf.GGUFReader(str(path))
            for tensor in reader.tensors:
                np_arr = np.array(tensor.data, copy=True)
                raw_tensors[tensor.name] = mx.array(np_arr)
    else:
        raise ValueError(
            f"Unsupported vector file extension '{suffix}'. Only '.safetensors' and '.gguf' are permitted."
        )

    # First pass: prefer explicit `direction.{l}` keys, or `lora_B` keys
    layer_vectors: Dict[int, mx.array] = {}
    for key, tensor in raw_tensors.items():
        # If this is a PEFT checkpoint containing both lora_A and lora_B, skip lora_A
        if "lora_a" in key.lower():
            continue
        layer_idx = _extract_layer_index(key)
        if layer_idx is None:
            continue
        try:
            vec = _coerce_to_1d_vector(tensor, hidden_size=hidden_size, key_name=key)
            layer_vectors[layer_idx] = normalize_unit_vector(vec, dtype=dtype)
        except ValueError:
            # Skip non-matching shapes (e.g. intermediate MLP projections if not hidden_size)
            continue

    if not layer_vectors:
        raise ValueError(
            f"No valid per-layer vectors of hidden_size={hidden_size} found in {path}. "
            f"Available keys: {list(raw_tensors.keys())[:10]}"
        )

    return layer_vectors


def resolve_decoder_layers(model: nn.Module) -> List[nn.Module]:
    """
    Resolve the 64-layer decoder list across Qwen3.8-27B (`qwen3_5` / `qwen3_next`)
    unified VLM/MTP wrappers or standalone text models:
      1. `model.language_model.model.layers`
      2. `model.model.layers`
      3. `model.layers`
    Also ensures `model.eval()` (`training=False`) is active on all resolved layers
    so `GatedDeltaNet` always dispatches to the compiled Metal kernel (`use_kernel=True`).
    """
    if hasattr(model, "training") and model.training:
        model.eval()

    layers: Optional[List[nn.Module]] = None
    if (
        hasattr(model, "language_model")
        and hasattr(model.language_model, "model")
        and hasattr(model.language_model.model, "layers")
    ):
        layers = model.language_model.model.layers
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers") and isinstance(model.layers, list):
        layers = model.layers

    if layers is None:
        raise AttributeError(
            "Could not resolve decoder layers from `model.language_model.model.layers`, "
            "`model.model.layers`, or `model.layers`."
        )

    for layer in layers:
        if hasattr(layer, "training") and layer.training:
            layer.eval()
    return layers


@mx.compile
def _project_subspace(h: mx.array, scaled_u_col: mx.array, u_row: mx.array) -> mx.array:
    return h - mx.matmul(h, scaled_u_col) * u_row


class ProjectedLayerWrapper(nn.Module):
    """
    Lightweight layer wrapper that delegates to the underlying decoder layer
    (either Gated DeltaNet linear-attention layer or full-attention layer)
    and optionally projects out a rank-1 subspace from the output hidden state:
        h <- h - alpha * (h . u_hat_l) * u_hat_l
    """

    def __init__(
        self,
        base_layer: nn.Module,
        layer_idx: int,
        unit_vector: Optional[mx.array] = None,
        controller: Optional["SubspaceProjector"] = None,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.layer_idx = layer_idx
        self.unit_vector = unit_vector
        self.u_hat = unit_vector
        self._controller = controller
        if hasattr(base_layer, "is_linear"):
            self.is_linear = base_layer.is_linear

        self.enabled: bool = True if controller is None else bool(controller.enabled)
        self.alpha: float = 1.0 if controller is None else float(controller.alpha)
        self.token_slice: Optional[Tuple[Optional[int], Optional[int]]] = (
            None if controller is None else controller.token_slice
        )
        self.capture_last_hidden: bool = False
        self.last_token_hidden: Optional[mx.array] = None

        if unit_vector is not None:
            self.u_hat_col: Optional[mx.array] = mx.contiguous(unit_vector[:, None])
            self.u_hat_row: Optional[mx.array] = mx.contiguous(unit_vector[None, None, :])
            self.scaled_u_hat_col: Optional[mx.array] = mx.contiguous(
                (self.alpha * self.u_hat_col).astype(unit_vector.dtype)
            )
            mx.eval(self.u_hat_col, self.u_hat_row, self.scaled_u_hat_col)
        else:
            self.u_hat_col = None
            self.u_hat_row = None
            self.scaled_u_hat_col = None

    def sync_projection_buffers(
        self,
        unit_vector: Optional[mx.array],
        alpha: float,
        is_layer_active: bool,
        token_slice: Optional[Tuple[Optional[int], Optional[int]]] = None,
        capture_last_hidden: bool = False,
    ) -> None:
        """Pre-reshape and pre-scale `scaled_u_hat_col` and `u_hat_row` outside the hot loop."""
        self.unit_vector = unit_vector
        self.u_hat = unit_vector
        self.alpha = float(alpha)
        self.token_slice = token_slice
        self.capture_last_hidden = bool(capture_last_hidden)
        self.enabled = bool(is_layer_active and unit_vector is not None and self.alpha != 0.0)
        if unit_vector is not None:
            if self.u_hat_col is None or self.u_hat_col.dtype != unit_vector.dtype:
                self.u_hat_col = mx.contiguous(unit_vector[:, None])
                self.u_hat_row = mx.contiguous(unit_vector[None, None, :])
            self.scaled_u_hat_col = mx.contiguous(
                (self.alpha * self.u_hat_col).astype(unit_vector.dtype)
            )
            mx.eval(self.u_hat_col, self.u_hat_row, self.scaled_u_hat_col)
        else:
            self.u_hat_col = None
            self.u_hat_row = None
            self.scaled_u_hat_col = None

    def __getattr__(self, name: str):
        # Delegate attribute lookups (such as .is_linear, .self_attn, .linear_attn, .mlp)
        # transparently to `base_layer` so mlx_lm cache construction and introspection work seamlessly.
        try:
            return super().__getattr__(name)
        except AttributeError:
            base = self.get("base_layer", None)
            if base is not None:
                return getattr(base, name)
            raise

    def _apply_projection(self, h: mx.array) -> mx.array:
        if self.scaled_u_hat_col is None or self.u_hat_row is None:
            return h
        scaled_col = self.scaled_u_hat_col
        u_row = self.u_hat_row
        if scaled_col.dtype != h.dtype:
            scaled_col = scaled_col.astype(h.dtype)
            u_row = u_row.astype(h.dtype)
        if self.token_slice is None:
            return _project_subspace(h, scaled_col, u_row)

        start_pos, end_pos = self.token_slice
        seq_len = int(h.shape[1])
        s_idx = 0 if start_pos is None else max(0, min(seq_len, int(start_pos)))
        e_idx = seq_len if end_pos is None else max(s_idx, min(seq_len, int(end_pos)))
        if s_idx >= e_idx:
            return h
        if s_idx == 0 and e_idx == seq_len:
            return _project_subspace(h, scaled_col, u_row)

        projected_slice = _project_subspace(h[:, s_idx:e_idx, :], scaled_col, u_row)
        parts: List[mx.array] = []
        if s_idx > 0:
            parts.append(h[:, :s_idx, :])
        parts.append(projected_slice)
        if e_idx < seq_len:
            parts.append(h[:, e_idx:, :])
        return mx.concatenate(parts, axis=1)

    def __call__(self, x: mx.array, *args, **kwargs):
        if self._controller is not None:
            is_active = (
                self._controller.enabled
                and self._controller.alpha != 0.0
                and self.layer_idx in self._controller.active_layers
                and self.scaled_u_hat_col is not None
            )
            capture_last = self.capture_last_hidden or (
                self.layer_idx in self._controller.capture_hidden_layers
            )
        else:
            is_active = self.enabled and self.alpha != 0.0 and self.scaled_u_hat_col is not None
            capture_last = self.capture_last_hidden

        if not is_active and not capture_last:
            return self.base_layer(x, *args, **kwargs)

        out = self.base_layer(x, *args, **kwargs)
        if isinstance(out, tuple):
            h = self._apply_projection(out[0]) if is_active else out[0]
            if capture_last:
                self.last_token_hidden = h[:, -1, :]
            return (h, *out[1:])
        h = self._apply_projection(out) if is_active else out
        if capture_last:
            self.last_token_hidden = h[:, -1, :]
        return h


class SubspaceProjector:
    """
    Controller managing rank-1 orthogonal projection hooks across `model.model.layers[1..63]`.

    Invariants enforced:
      1. Layer 0 is NEVER projected (`assert 0 not in active_layers`).
      2. Emits a UserWarning if `alpha > 1.0` (since alpha = 1.0 is an exact orthogonal
         projection Pi_{u^perp}(h) = h - (h . u) u).
    """

    def __init__(
        self,
        vectors: Dict[int, mx.array],
        alpha: float = 1.0,
        active_layers: Optional[Iterable[int]] = None,
        enabled: bool = True,
        dtype: mx.Dtype = mx.bfloat16,
    ):
        self.dtype = dtype
        self.vectors: Dict[int, mx.array] = {}
        self.u_hat_cols: Dict[int, mx.array] = {}
        self.u_hat_rows: Dict[int, mx.array] = {}
        self.scaled_u_hat_cols: Dict[int, mx.array] = {}

        for l_idx, vec in vectors.items():
            idx_int = int(l_idx)
            u = normalize_unit_vector(vec, dtype=dtype)
            self.vectors[idx_int] = u
            self.u_hat_cols[idx_int] = mx.contiguous(u[:, None])
            self.u_hat_rows[idx_int] = mx.contiguous(u[None, None, :])

        if active_layers is None:
            # Default to all available layers except layer 0
            default_layers = [l for l in sorted(self.vectors.keys()) if l != 0]
            if not default_layers:
                default_layers = list(range(1, NUM_LAYERS_DEFAULT))
            active_layers = default_layers

        self._active_layers: Set[int] = set()
        self._alpha: float = 1.0
        self.enabled: bool = enabled
        self.token_slice: Optional[Tuple[Optional[int], Optional[int]]] = None
        self.capture_hidden_layers: Set[int] = set()
        self._attached_model = None

        self.set_active_layers(active_layers)
        self.set_alpha(alpha)

    def _sync_attached_wrappers(self) -> None:
        if self._attached_model is None:
            return
        layers = self._get_layers_list(self._attached_model)
        for idx in range(1, len(layers)):
            layer = layers[idx]
            if isinstance(layer, ProjectedLayerWrapper):
                u = self.vectors.get(idx, layer.unit_vector)
                is_active = self.enabled and (idx in self._active_layers) and (self._alpha != 0.0)
                layer.sync_projection_buffers(
                    unit_vector=u,
                    alpha=self._alpha,
                    is_layer_active=is_active,
                    token_slice=self.token_slice,
                    capture_last_hidden=(idx in self.capture_hidden_layers),
                )

    @property
    def active_layers(self) -> Set[int]:
        return self._active_layers

    @active_layers.setter
    def active_layers(self, layers: Iterable[int]) -> None:
        self.set_active_layers(layers)

    def set_active_layers(self, layers: Iterable[int]) -> None:
        layer_set = {int(l) for l in layers}
        assert 0 not in layer_set, (
            "Invariant violation: Layer 0 must NEVER be projected (`0 not in active_layers`)."
        )
        self._active_layers = layer_set
        self._sync_attached_wrappers()

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        self.set_alpha(value)

    def set_alpha(self, value: float) -> None:
        val = float(value)
        if val > 1.0:
            warnings.warn(
                f"SubspaceProjector alpha={val} > 1.0 exceeds exact orthogonal projection "
                f"(alpha=1.0 corresponds to Pi_{{u^perp}}(h) = h - (h . u)u and over-projects/reflects the component).",
                UserWarning,
                stacklevel=2,
            )
        self._alpha = val
        for idx_int, u_col in self.u_hat_cols.items():
            self.scaled_u_hat_cols[idx_int] = mx.contiguous((val * u_col).astype(self.dtype))
        if self.scaled_u_hat_cols:
            mx.eval(*self.scaled_u_hat_cols.values())
        self._sync_attached_wrappers()

    def configure(
        self,
        *,
        alpha: Optional[float] = None,
        active_layers: Optional[Iterable[int]] = None,
        enabled: Optional[bool] = None,
        token_slice: Optional[Tuple[Optional[int], Optional[int]]] = None,
        clear_token_slice: bool = False,
        capture_hidden_layers: Optional[Iterable[int]] = None,
    ) -> "SubspaceProjector":
        """Dynamically reconfigure alpha, active_layers, enabled, or token_slice between forward passes."""
        if enabled is not None:
            self.enabled = bool(enabled)
        if clear_token_slice:
            self.token_slice = None
        if token_slice is not None:
            self.token_slice = (token_slice[0], token_slice[1])
        if capture_hidden_layers is not None:
            self.capture_hidden_layers = {int(l) for l in capture_hidden_layers}
        if active_layers is not None:
            layer_set = {int(l) for l in active_layers}
            assert 0 not in layer_set, (
                "Invariant violation: Layer 0 must NEVER be projected (`0 not in active_layers`)."
            )
            self._active_layers = layer_set
        if alpha is not None:
            self.set_alpha(alpha)
        else:
            self._sync_attached_wrappers()
        return self

    @contextmanager
    def pulse(
        self,
        alpha: float = 1.0,
        active_layers: Optional[Iterable[int]] = None,
        enabled: bool = True,
        token_slice: Optional[Tuple[Optional[int], Optional[int]]] = None,
    ):
        """
        Context manager for Token-Windowed Pulse Steering:
        Temporarily applies `alpha`, `active_layers`, `enabled`, and optional intra-chunk
        `token_slice = (start_pos, end_pos)` inside the `with` block, and automatically
        restores the exact prior `SubspaceProjector` state on exit.
        """
        prev_alpha = self._alpha
        prev_layers = set(self._active_layers)
        prev_enabled = self.enabled
        prev_slice = self.token_slice
        try:
            self.configure(
                alpha=alpha,
                active_layers=active_layers if active_layers is not None else prev_layers,
                enabled=enabled,
                token_slice=token_slice,
                clear_token_slice=(token_slice is None),
            )
            yield self
        finally:
            self.configure(
                alpha=prev_alpha,
                active_layers=prev_layers,
                enabled=prev_enabled,
                token_slice=prev_slice,
                clear_token_slice=(prev_slice is None),
            )

    def get_last_hidden(self, layer_idx: int = 63) -> Optional[mx.array]:
        """
        Return the most recently captured post-layer residual hidden state `(B, D)`
        for `layer_idx` (when `layer_idx` is in `capture_hidden_layers`).
        """
        if self._attached_model is None:
            return None
        layers = self._get_layers_list(self._attached_model)
        if 0 <= layer_idx < len(layers):
            layer = layers[layer_idx]
            if isinstance(layer, ProjectedLayerWrapper):
                return layer.last_token_hidden
        return None

    @staticmethod
    def _get_layers_list(model: nn.Module) -> List[nn.Module]:
        return resolve_decoder_layers(model)

    def attach(self, model: nn.Module) -> "SubspaceProjector":
        """
        Wrap resolved decoder layers `[1..N-1]` in-place with `ProjectedLayerWrapper`
        and enforce `model.eval()` (`training=False`) so GatedDeltaNet uses the Metal kernel.
        Layer 0 is intentionally left untouched.
        """
        if hasattr(model, "eval"):
            model.eval()
        layers = self._get_layers_list(model)
        for idx in range(1, len(layers)):
            layer = layers[idx]
            u = self.vectors.get(idx, None)
            if isinstance(layer, ProjectedLayerWrapper):
                layer._controller = self
                layer.sync_projection_buffers(
                    unit_vector=u,
                    alpha=self._alpha,
                    is_layer_active=self.enabled and (idx in self._active_layers),
                    token_slice=self.token_slice,
                    capture_last_hidden=(idx in self.capture_hidden_layers),
                )
                layer.eval()
            else:
                wrapped = ProjectedLayerWrapper(
                    base_layer=layer,
                    layer_idx=idx,
                    unit_vector=u,
                    controller=self,
                )
                wrapped.sync_projection_buffers(
                    unit_vector=u,
                    alpha=self._alpha,
                    is_layer_active=self.enabled and (idx in self._active_layers),
                    token_slice=self.token_slice,
                    capture_last_hidden=(idx in self.capture_hidden_layers),
                )
                wrapped.eval()
                layers[idx] = wrapped
        self._attached_model = model
        return self

    def detach(self, model: Optional[nn.Module] = None) -> None:
        """Unwrap `ProjectedLayerWrapper` instances and restore original layers."""
        target = model or self._attached_model
        if target is None:
            return
        layers = self._get_layers_list(target)
        for idx in range(len(layers)):
            if isinstance(layers[idx], ProjectedLayerWrapper):
                layers[idx] = layers[idx].base_layer
        if target is self._attached_model:
            self._attached_model = None


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Ensure `model` is restored to 100% stock `mlx_lm` decoder layers with zero
    Python wrapper overhead (unwrapping any `ProjectedLayerWrapper` instances).
    """
    layers = resolve_decoder_layers(model)
    for idx in range(len(layers)):
        while isinstance(layers[idx], ProjectedLayerWrapper):
            layers[idx] = layers[idx].base_layer
        if hasattr(layers[idx], "eval"):
            layers[idx].eval()
    if hasattr(model, "eval"):
        model.eval()
    return model


@mx.compile
def _fused_quant_lora_rank1(
    y: mx.array,
    x: mx.array,
    v_w_col: mx.array,
    scaled_u_row: mx.array,
) -> mx.array:
    return y - mx.matmul(x, v_w_col) * scaled_u_row


class QuantizedRank1WriterWrapper(nn.Module):
    """
    Lightweight rank-1 LoRA-style fused additive hook for `nn.QuantizedLinear`
    when running weight-space projection without dequantizing/requantizing 8-bit weights.
    """

    def __init__(
        self,
        base_quant_linear: nn.Module,
        v_w_col: mx.array,
        scaled_u_row: mx.array,
    ):
        super().__init__()
        self.base_quant_linear = base_quant_linear
        self.v_w_col = v_w_col
        self.scaled_u_row = scaled_u_row

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            base = self.get("base_quant_linear", None)
            if base is not None:
                return getattr(base, name)
            raise

    def __call__(self, x: mx.array) -> mx.array:
        y = self.base_quant_linear(x)
        v_col = self.v_w_col
        u_row = self.scaled_u_row
        if v_col.dtype != x.dtype:
            v_col = v_col.astype(x.dtype)
            u_row = u_row.astype(x.dtype)
        return _fused_quant_lora_rank1(y, x, v_col, u_row)


VALID_WRITER_COMPONENTS: Set[str] = {"deltanet", "full_attn", "mlp"}


class WeightSpaceProjector:
    """
    Reversible In-Place Weight-Space Orthogonalization for Qwen3.8-27B (`qwen3_5`).

    In a `qwen3_5` decoder block, each layer writes to the residual stream via two
    projection matrices:
      1. Attention/Recurrent writer:
         - `layer.linear_attn.out_proj` (on the 48 Gated DeltaNet layers, component `"deltanet"`)
         - `layer.self_attn.o_proj` (on the 16 Full-Attention layers `3, 7, ..., 63`, component `"full_attn"`)
      2. MLP writer:
         - `layer.mlp.down_proj` (on all 64 layers, component `"mlp"`)

    For any linear writer `y = x W^T` where `W in R^{5120 x d_in}`, projecting the output
    via `Pi_{u_l^perp}(y) = y - alpha (y . u_l) u_l` is achieved in weight space by
    orthogonalizing the output rows of `W`:
        W_new = W - alpha * u_l (u_l^T W) = W - alpha * u_l v_W

    Pre-caches the compact row vectors `v_W = u_l^T W in R^{1 x d_in}` (< 10 MB total across
    all 63 layers) and tracks the currently applied `alpha` per `(layer_idx, component)`
    so `apply_in_place(...)` and `restore_baseline()` can switch configurations in-place
    in `< 20 ms` without reloading the 55.6 GB checkpoint.
    """

    def __init__(
        self,
        model: nn.Module,
        vectors: Dict[int, mx.array],
        dtype: mx.Dtype = mx.bfloat16,
        quantized_mode: str = "dequant_requant",
        target_layers: Optional[Iterable[int]] = None,
    ):
        """
        Args:
            model: The language model (automatically unwrapped via `unwrap_model(model)`).
            vectors: Mapping `layer_idx -> unit_vector` (shape `(5120,)`).
            dtype: Target dtype for unit vectors (default `mx.bfloat16`).
            quantized_mode: How to handle `nn.QuantizedLinear` modules (8-bit):
                - `"dequant_requant"` (default): Dequantizes baseline `W`, applies exact
                  rank-1 row orthogonalization, and re-quantizes in-place so the model
                  remains 100% native `nn.QuantizedLinear` with zero wrapper overhead.
                - `"lora_hook"`: Wraps active `nn.QuantizedLinear` modules with
                  `QuantizedRank1WriterWrapper`.
            target_layers: Optional iterable of layer indices `[1..63]` to pre-cache
                `v_W = u_l^T W` for (defaults to all layers present in `vectors`).
        """
        if quantized_mode not in ("dequant_requant", "lora_hook"):
            raise ValueError(
                f"quantized_mode must be 'dequant_requant' or 'lora_hook' (got {quantized_mode})."
            )
        self.model = unwrap_model(model)
        self.dtype = dtype
        self.quantized_mode = quantized_mode

        allowed_layers: Optional[Set[int]] = None
        if target_layers is not None:
            allowed_layers = {int(l) for l in target_layers}
            assert 0 not in allowed_layers, (
                "Invariant violation: Layer 0 must NEVER be projected (`0 not in target_layers`)."
            )

        self.vectors_f32: Dict[int, mx.array] = {}
        self.vectors_bf16: Dict[int, mx.array] = {}
        for l_idx, vec in vectors.items():
            idx_int = int(l_idx)
            if idx_int == 0:
                continue
            if allowed_layers is not None and idx_int not in allowed_layers:
                continue
            u_bf16 = normalize_unit_vector(vec, dtype=dtype)
            u_f32 = normalize_unit_vector(vec, dtype=mx.float32)
            self.vectors_bf16[idx_int] = u_bf16
            self.vectors_f32[idx_int] = u_f32

        # Pre-compute compact v_W = u_l^T W in FP32 for each (layer_idx, component)
        # Keys: (layer_idx, comp_name) where comp_name in {"deltanet", "full_attn", "mlp"}
        self._v_w_cache: Dict[Tuple[int, str], mx.array] = {}
        self._applied_alpha: Dict[Tuple[int, str], float] = {}
        # For quantized layers when using dequant_requant, cache baseline quantized tuple (w_q, scales, biases)
        self._quant_baseline: Dict[Tuple[int, str], Tuple[mx.array, mx.array, Optional[mx.array]]] = {}

        self._precompute_projection_rows()

    def _iter_layer_writers(self) -> List[Tuple[int, str, nn.Module, str, nn.Module]]:
        """
        Return list of `(layer_idx, comp_name, parent_module, attr_name, writer_module)`
        for layers `1..N-1`.
        """
        layers = resolve_decoder_layers(self.model)
        entries: List[Tuple[int, str, nn.Module, str, nn.Module]] = []
        for l_idx in range(1, len(layers)):
            if l_idx not in self.vectors_f32:
                continue
            layer = layers[l_idx]
            is_lin = getattr(layer, "is_linear", False)
            if is_lin and hasattr(layer, "linear_attn") and hasattr(layer.linear_attn, "out_proj"):
                entries.append(
                    (l_idx, "deltanet", layer.linear_attn, "out_proj", layer.linear_attn.out_proj)
                )
            elif not is_lin and hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
                entries.append(
                    (l_idx, "full_attn", layer.self_attn, "o_proj", layer.self_attn.o_proj)
                )

            if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
                entries.append(
                    (l_idx, "mlp", layer.mlp, "down_proj", layer.mlp.down_proj)
                )
        return entries

    def _precompute_projection_rows(self) -> None:
        to_eval: List[mx.array] = []
        for l_idx, comp_name, _, _, mod in self._iter_layer_writers():
            key = (l_idx, comp_name)
            u_row_f32 = self.vectors_f32[l_idx][None, :]  # (1, 5120)

            base_mod = mod.base_quant_linear if isinstance(mod, QuantizedRank1WriterWrapper) else mod
            if isinstance(base_mod, nn.QuantizedLinear):
                w_deq = mx.dequantize(
                    base_mod.weight,
                    base_mod.scales,
                    base_mod.get("biases", None),
                    base_mod.group_size,
                    base_mod.bits,
                ).astype(mx.float32)
                v_w = mx.contiguous(mx.matmul(u_row_f32, w_deq))  # (1, d_in)
                self._quant_baseline[key] = (
                    mx.array(base_mod.weight),
                    mx.array(base_mod.scales),
                    mx.array(base_mod.biases) if base_mod.get("biases", None) is not None else None,
                )
            else:
                w_f32 = base_mod.weight.astype(mx.float32)
                v_w = mx.contiguous(mx.matmul(u_row_f32, w_f32))  # (1, d_in)

            self._v_w_cache[key] = v_w
            self._applied_alpha[key] = 0.0
            to_eval.append(v_w)

        if to_eval:
            mx.eval(*to_eval)

    @property
    def cached_v_w_nbytes(self) -> int:
        return sum(int(v.nbytes) for v in self._v_w_cache.values())

    def apply_in_place(
        self,
        alpha: float = 1.0,
        active_layers: Optional[Iterable[int]] = None,
        target_components: Iterable[str] = ("deltanet", "full_attn", "mlp"),
    ) -> int:
        """
        In-place orthogonalize the specified writer matrices (`W <- W - alpha * u_l v_W`)
        across `active_layers` and `target_components`, while automatically restoring any
        previously modified matrices that are not in the new target set back to baseline (`alpha=0.0`).

        Returns:
            Total number of weight matrices actively orthogonalized (`alpha != 0.0`).
        """
        val = float(alpha)
        if val > 1.0:
            warnings.warn(
                f"WeightSpaceProjector alpha={val} > 1.0 exceeds exact orthogonal projection.",
                UserWarning,
                stacklevel=2,
            )

        if active_layers is None:
            layer_set = set(self.vectors_f32.keys())
        else:
            layer_set = {int(l) for l in active_layers}
        assert 0 not in layer_set, (
            "Invariant violation: Layer 0 must NEVER be projected (`0 not in active_layers`)."
        )

        comp_set = {str(c) for c in target_components}
        invalid_comps = comp_set - VALID_WRITER_COMPONENTS
        if invalid_comps:
            raise ValueError(
                f"Invalid target_components {invalid_comps}. Valid options: {VALID_WRITER_COMPONENTS}"
            )

        unwrap_model(self.model)
        updated_tensors: List[mx.array] = []
        active_matrix_count = 0

        for l_idx, comp_name, parent_mod, attr_name, mod in self._iter_layer_writers():
            key = (l_idx, comp_name)
            desired_alpha = val if (l_idx in layer_set and comp_name in comp_set) else 0.0
            if desired_alpha != 0.0:
                active_matrix_count += 1

            current_alpha = self._applied_alpha.get(key, 0.0)
            delta_alpha = desired_alpha - current_alpha
            if abs(delta_alpha) < 1e-9:
                continue

            v_w_f32 = self._v_w_cache[key]  # (1, d_in)
            u_col_f32 = self.vectors_f32[l_idx][:, None]  # (5120, 1)

            base_mod = mod.base_quant_linear if isinstance(mod, QuantizedRank1WriterWrapper) else mod
            if isinstance(base_mod, nn.QuantizedLinear):
                if self.quantized_mode == "lora_hook":
                    if desired_alpha == 0.0:
                        setattr(parent_mod, attr_name, base_mod)
                    else:
                        v_w_col = mx.contiguous(v_w_f32.T.astype(self.dtype))  # (d_in, 1)
                        scaled_u_row = mx.contiguous(
                            (desired_alpha * self.vectors_f32[l_idx][None, :]).astype(self.dtype)
                        )
                        updated_tensors.extend([v_w_col, scaled_u_row])
                        wrapped_q = QuantizedRank1WriterWrapper(
                            base_quant_linear=base_mod,
                            v_w_col=v_w_col,
                            scaled_u_row=scaled_u_row,
                        )
                        wrapped_q.eval()
                        setattr(parent_mod, attr_name, wrapped_q)
                else:
                    # Exact dequantize -> orthogonalize -> requantize from clean baseline
                    orig_wq, orig_scales, orig_biases = self._quant_baseline[key]
                    if desired_alpha == 0.0:
                        base_mod.weight = mx.array(orig_wq)
                        base_mod.scales = mx.array(orig_scales)
                        if orig_biases is not None:
                            base_mod.biases = mx.array(orig_biases)
                        updated_tensors.extend([base_mod.weight, base_mod.scales])
                    else:
                        w_deq = mx.dequantize(
                            orig_wq,
                            orig_scales,
                            orig_biases,
                            base_mod.group_size,
                            base_mod.bits,
                        ).astype(mx.float32)
                        w_proj = (w_deq - (desired_alpha * u_col_f32) * v_w_f32).astype(
                            orig_scales.dtype
                        )
                        q_out = mx.quantize(
                            w_proj,
                            group_size=base_mod.group_size,
                            bits=base_mod.bits,
                        )
                        base_mod.weight = q_out[0]
                        base_mod.scales = q_out[1]
                        if len(q_out) > 2 and orig_biases is not None:
                            base_mod.biases = q_out[2]
                        updated_tensors.extend([base_mod.weight, base_mod.scales])
            else:
                # Standard BF16 / FP16 / FP32 Linear writer: update in-place via delta_alpha
                w_orig_dtype = base_mod.weight.dtype
                w_updated = (
                    base_mod.weight.astype(mx.float32)
                    - (delta_alpha * u_col_f32) * v_w_f32
                ).astype(w_orig_dtype)
                base_mod.weight = w_updated
                updated_tensors.append(base_mod.weight)

            self._applied_alpha[key] = desired_alpha

        if updated_tensors:
            mx.eval(*updated_tensors)
        return active_matrix_count

    def restore_baseline(self) -> None:
        """Restore all modified writer matrices back to exact unprojected baseline (`alpha = 0.0`)."""
        self.apply_in_place(alpha=0.0, active_layers=set(), target_components=VALID_WRITER_COMPONENTS)
