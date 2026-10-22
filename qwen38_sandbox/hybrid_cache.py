"""
hybrid_cache.py — Post-Hoc Slicing & Forking for Hybrid KV + DeltaNet State

Manages the 64-layer hybrid cache of Qwen3.8-27B (`qwen3_5` 3:1 architecture):
  - 16 Full-Attention Layers (`KVCache`, at indices 3, 7, 11, ..., 63):
      Supports exact post-hoc sequence rewinding via tensor slicing
      (`keys[..., :token_idx, :]`, `values[..., :token_idx, :]`) and updating `cache.offset = token_idx`.
  - 48 Gated DeltaNet Layers (`ArraysCache`, at all other indices):
      Because recurrent fast-weight state S_t in R^{d_k x d_v} (FP32) and Conv1d buffer
      form a contracting semigroup (~192 MB total across 48 layers), supports `snapshot(token_idx)`
      which deep-copies and explicitly calls `mx.eval(cache.state)` into unified memory,
      and `restore(token_idx)` to restore the recurrent state at a designated edit point.
  - `fork_and_edit(cache, edit_token_idx, new_token_ids)`:
      Rewinds the 16 KV caches to `edit_token_idx`, restores the 48 DeltaNet states to
      `edit_token_idx`, and executes a chunked warm prefill ONLY on `new_token_ids`
      while evaluating `ArraysCache.state` at every step to prevent lazy graph buildup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import ArraysCache, KVCache, make_prompt_cache

from subspace_projector import resolve_decoder_layers


DEFAULT_PREFILL_STEP_SIZE = 512


@dataclass
class CacheInspectionSummary:
    total_layers: int
    kv_layer_indices: List[int]
    deltanet_layer_indices: List[int]
    kv_offset: int
    kv_nbytes: int
    deltanet_nbytes: int

    @property
    def total_nbytes(self) -> int:
        return self.kv_nbytes + self.deltanet_nbytes


def _clone_array(arr: Optional[mx.array]) -> Optional[mx.array]:
    if arr is None:
        return None
    # Create an independent contiguous copy in unified memory
    return mx.array(arr)


def eval_hybrid_cache_state(cache_list: Sequence[Any]) -> None:
    """
    Explicitly evaluate `cache.state` across all cache entries (the 48 Gated DeltaNet
    `ArraysCache` instances and 16 `KVCache` instances) in a single batched `mx.eval` call
    so lazy evaluation graphs do not accumulate in unified memory.
    """
    to_eval: List[mx.array] = []
    for c in cache_list:
        if isinstance(c, ArraysCache):
            cache_arrs, left_pad, lengths = c.state
            for arr in cache_arrs:
                if isinstance(arr, mx.array):
                    to_eval.append(arr)
            if isinstance(left_pad, mx.array):
                to_eval.append(left_pad)
            if isinstance(lengths, mx.array):
                to_eval.append(lengths)
        elif isinstance(c, KVCache):
            if c.keys is not None:
                to_eval.append(c.keys)
            if c.values is not None:
                to_eval.append(c.values)
        elif hasattr(c, "state"):
            st = c.state
            if isinstance(st, mx.array):
                to_eval.append(st)
            elif isinstance(st, (list, tuple)):
                for item in st:
                    if isinstance(item, mx.array):
                        to_eval.append(item)
    if to_eval:
        mx.eval(*to_eval)


class HybridTimelineCache:
    """
    Timeline & Branching Manager for Qwen3.8-27B's 64-layer hybrid cache
    (48 Gated DeltaNet `ArraysCache` + 16 Full-Attention `KVCache`).
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        cache: Optional[List[Any]] = None,
        prefill_step_size: int = DEFAULT_PREFILL_STEP_SIZE,
    ):
        if prefill_step_size not in (512, 1024, 2048):
            raise ValueError(
                f"Hardware constraint: prefill_step_size must be 512, 1024, or 2048 (got {prefill_step_size})."
            )
        self.model = model
        self.prefill_step_size = prefill_step_size

        if model is not None:
            # Ensure model and all 48 GatedDeltaNet layers are in eval mode (training=False)
            # so mlx-lm always dispatches to the compiled Metal kernel (`use_kernel=True`).
            if hasattr(model, "eval"):
                model.eval()
            resolve_decoder_layers(model)

        if cache is not None:
            self.cache: List[Any] = cache
        elif model is not None:
            if hasattr(model, "make_cache"):
                self.cache = model.make_cache()
            elif hasattr(model, "language_model") and hasattr(model.language_model, "make_cache"):
                self.cache = model.language_model.make_cache()
            else:
                layers = resolve_decoder_layers(model)
                if any(hasattr(l, "is_linear") for l in layers):
                    self.cache = [
                        ArraysCache(size=2) if getattr(l, "is_linear", False) else KVCache()
                        for l in layers
                    ]
                else:
                    self.cache = make_prompt_cache(model)
        else:
            raise ValueError("Either `model` or `cache` must be provided to HybridTimelineCache.")

        self.kv_layer_indices: List[int] = []
        self.deltanet_layer_indices: List[int] = []
        for idx, c in enumerate(self.cache):
            if isinstance(c, KVCache):
                self.kv_layer_indices.append(idx)
            elif isinstance(c, ArraysCache):
                self.deltanet_layer_indices.append(idx)
            else:
                # Fallback duck-typing check
                if hasattr(c, "keys") and hasattr(c, "values") and hasattr(c, "offset"):
                    self.kv_layer_indices.append(idx)
                else:
                    self.deltanet_layer_indices.append(idx)

        # Snapshots keyed by token_idx -> {layer_idx: (conv_state_copy, ssm_state_copy), left_padding, lengths}
        self._snapshots: Dict[int, Dict[int, Tuple[List[Optional[mx.array]], Optional[mx.array], Optional[mx.array]]]] = {}
        self.token_history: List[int] = []

    def eval_all(self) -> None:
        """Evaluate all 64 hybrid cache states in a single batched `mx.eval` call."""
        eval_hybrid_cache_state(self.cache)

    def inspect(self) -> CacheInspectionSummary:
        """Inspect layer distribution, current token offset, and memory footprint of the hybrid cache."""
        kv_offset = 0
        if self.kv_layer_indices:
            first_kv = self.cache[self.kv_layer_indices[0]]
            kv_offset = int(first_kv.offset)

        kv_nbytes = sum(int(self.cache[i].nbytes) for i in self.kv_layer_indices)
        deltanet_nbytes = sum(int(self.cache[i].nbytes) for i in self.deltanet_layer_indices)
        return CacheInspectionSummary(
            total_layers=len(self.cache),
            kv_layer_indices=list(self.kv_layer_indices),
            deltanet_layer_indices=list(self.deltanet_layer_indices),
            kv_offset=kv_offset,
            kv_nbytes=kv_nbytes,
            deltanet_nbytes=deltanet_nbytes,
        )

    @property
    def current_offset(self) -> int:
        if not self.kv_layer_indices:
            return 0
        return int(self.cache[self.kv_layer_indices[0]].offset)

    def rewind_to(self, token_idx: int) -> None:
        """
        Rewind the 16 Full-Attention `KVCache` layers to `token_idx` by slicing
        the sequence axis (`keys[..., :token_idx, :]`, `values[..., :token_idx, :]`)
        and setting `cache.offset = token_idx`.
        """
        if token_idx < 0:
            raise ValueError(f"token_idx must be non-negative (got {token_idx}).")

        to_eval: List[mx.array] = []
        for idx in self.kv_layer_indices:
            kv_cache: KVCache = self.cache[idx]
            if kv_cache.keys is not None and kv_cache.values is not None:
                if token_idx > kv_cache.offset:
                    raise ValueError(
                        f"Cannot rewind layer {idx} KVCache from offset {kv_cache.offset} "
                        f"to future token_idx {token_idx}."
                    )
                if token_idx == 0:
                    kv_cache.keys = kv_cache.keys[..., :0, :]
                    kv_cache.values = kv_cache.values[..., :0, :]
                else:
                    kv_cache.keys = mx.contiguous(kv_cache.keys[..., :token_idx, :])
                    kv_cache.values = mx.contiguous(kv_cache.values[..., :token_idx, :])
                to_eval.extend([kv_cache.keys, kv_cache.values])
            kv_cache.offset = int(token_idx)

        if to_eval:
            mx.eval(*to_eval)

    def snapshot(self, token_idx: Optional[int] = None) -> int:
        """
        Deep-copy and `mx.eval` the 48 Gated DeltaNet `ArraysCache.state` tensors into
        unified memory at `token_idx` (defaults to `current_offset`).
        """
        if token_idx is None:
            token_idx = self.current_offset
        token_idx = int(token_idx)

        layer_states: Dict[int, Tuple[List[Optional[mx.array]], Optional[mx.array], Optional[mx.array]]] = {}
        for idx in self.deltanet_layer_indices:
            arr_cache: ArraysCache = self.cache[idx]
            # First evaluate live state tuple to prevent lazy graph accumulation
            mx.eval(arr_cache.state)

            raw_cache_list, left_padding, lengths = arr_cache.state
            cloned_list: List[Optional[mx.array]] = [
                _clone_array(t) for t in raw_cache_list
            ]
            cloned_left_pad = _clone_array(left_padding)
            cloned_lengths = _clone_array(lengths)

            # Explicitly materialize the cloned snapshot state tuple
            snapshot_tuple = (cloned_list, cloned_left_pad, cloned_lengths)
            mx.eval(snapshot_tuple)
            layer_states[idx] = snapshot_tuple

        self._snapshots[token_idx] = layer_states
        return token_idx

    def has_snapshot(self, token_idx: int) -> bool:
        return int(token_idx) in self._snapshots

    def list_snapshots(self) -> List[int]:
        return sorted(self._snapshots.keys())

    def nearest_snapshot_le(self, target_token_idx: int) -> int:
        """
        Return the largest recorded snapshot index `<= target_token_idx` (or `0` if none).
        """
        target = int(target_token_idx)
        if target < 0:
            raise ValueError(f"target_token_idx must be >= 0 (got {target}).")
        candidates = [s for s in self._snapshots.keys() if s <= target]
        return max(candidates) if candidates else 0

    def restore_to_arbitrary_index(
        self,
        target_token_idx: int,
        token_history: Optional[Sequence[int]] = None,
        model: Optional[nn.Module] = None,
        cache_snapshot_at_target: bool = True,
    ) -> Tuple[int, int]:
        """
        Rewind/restore the 64-layer hybrid cache to ANY arbitrary `target_token_idx`:
          1. If `target_token_idx` is already a recorded snapshot, rewinds the 16 `KVCache`
             views and restores the 48 `ArraysCache` states directly (`delta_replayed = 0`).
          2. Otherwise, finds `snap_idx = nearest_snapshot_le(target_token_idx)`, rewinds
             `KVCache` to `snap_idx`, restores `ArraysCache` to `snap_idx`, and replays the
             short token delta `history[snap_idx : target_token_idx]` (`0..stride-1` tokens)
             to reach `target_token_idx` in `< 50 ms`.

        Returns:
            `(snap_idx_used, delta_tokens_replayed)`
        """
        target = int(target_token_idx)
        history = list(token_history) if token_history is not None else list(self.token_history)
        if target in self._snapshots or target == 0:
            self.rewind_to(target)
            self.restore(target)
            self.token_history = history[:target]
            return target, 0

        snap_idx = self.nearest_snapshot_le(target)
        delta_len = target - snap_idx
        if len(history) < target:
            raise ValueError(
                f"Cannot replay delta from snapshot {snap_idx} to target_token_idx={target}: "
                f"token_history only has {len(history)} tokens."
            )
        delta_tokens = history[snap_idx:target]
        self.rewind_to(snap_idx)
        self.restore(snap_idx)

        target_model = model or self.model
        if target_model is None:
            raise ValueError("A `model` is required to replay delta tokens in `restore_to_arbitrary_index`.")
        delta_arr = mx.array(delta_tokens, dtype=mx.int32)[None, :]
        logits = target_model(delta_arr, cache=self.cache)
        mx.eval(logits)
        self.eval_all()
        self.token_history = history[:target]
        if cache_snapshot_at_target:
            self.snapshot(target)
        return snap_idx, delta_len

    def restore(self, token_idx: int) -> None:
        """
        Restore the 48 Gated DeltaNet `ArraysCache` layers to the state previously
        captured at `token_idx`, explicitly calling `mx.eval(cache.state)`.
        """
        token_idx = int(token_idx)
        if token_idx == 0 and token_idx not in self._snapshots:
            # Reset to clean initial state at token 0
            for idx in self.deltanet_layer_indices:
                arr_cache: ArraysCache = self.cache[idx]
                arr_cache.state = ([None] * len(arr_cache.cache), None, None)
                mx.eval(arr_cache.state)
            return

        if token_idx not in self._snapshots:
            raise KeyError(
                f"No Gated DeltaNet snapshot recorded for token_idx={token_idx}. "
                f"Available snapshots: {self.list_snapshots()}"
            )

        layer_states = self._snapshots[token_idx]
        for idx in self.deltanet_layer_indices:
            arr_cache: ArraysCache = self.cache[idx]
            saved_list, saved_left_pad, saved_lengths = layer_states[idx]
            # Clone on restore so the stored snapshot remains immutable for repeated branches
            restored_list = [_clone_array(t) for t in saved_list]
            restored_left_pad = _clone_array(saved_left_pad)
            restored_lengths = _clone_array(saved_lengths)
            arr_cache.state = (restored_list, restored_left_pad, restored_lengths)
            mx.eval(arr_cache.state)

    def prefill_with_snapshots(
        self,
        token_ids: Union[mx.array, Sequence[int]],
        snapshot_indices: Optional[Sequence[int]] = None,
        model: Optional[nn.Module] = None,
    ) -> mx.array:
        """
        Run a chunked prefill over `token_ids` using `self.prefill_step_size` (512 or 1024),
        automatically pausing at any requested `snapshot_indices` to capture exact
        Gated DeltaNet `ArraysCache` snapshots and evaluating `cache.state` after each chunk.
        """
        target_model = model or self.model
        if target_model is None:
            raise ValueError("A `model` is required to run `prefill_with_snapshots`.")
        if hasattr(target_model, "eval") and getattr(target_model, "training", False):
            target_model.eval()
        resolve_decoder_layers(target_model)

        if not isinstance(token_ids, mx.array):
            tokens_list = [int(t) for t in token_ids]
            tokens = mx.array(tokens_list, dtype=mx.int32)
        else:
            tokens = token_ids
            tokens_list = [int(t) for t in tokens.reshape(-1).tolist()]
        if tokens.ndim == 1:
            tokens = tokens[None, :]  # (1, L)

        total_len = int(tokens.shape[1])
        start_offset = self.current_offset
        self.token_history = self.token_history[:start_offset] + tokens_list

        # Build sorted cut points combining snapshot boundaries and prefill_step_size chunks
        snap_set = sorted(
            {
                int(s) - start_offset
                for s in (snapshot_indices or [])
                if start_offset <= int(s) <= start_offset + total_len
            }
        )
        if 0 in snap_set:
            self.snapshot(start_offset)
            snap_set.remove(0)

        boundaries: List[int] = []
        cursor = 0
        for snap_rel in snap_set:
            while cursor + self.prefill_step_size < snap_rel:
                cursor += self.prefill_step_size
                boundaries.append(cursor)
            if snap_rel > cursor:
                cursor = snap_rel
                boundaries.append(cursor)
        while cursor + self.prefill_step_size < total_len:
            cursor += self.prefill_step_size
            boundaries.append(cursor)
        if cursor < total_len:
            boundaries.append(total_len)

        last_logits: Optional[mx.array] = None
        prev = 0
        snap_rel_lookup = set(snap_set)
        for end in boundaries:
            chunk = tokens[:, prev:end]
            last_logits = target_model(chunk, cache=self.cache)
            if end in snap_rel_lookup or end == boundaries[-1]:
                mx.eval(last_logits)
                self.eval_all()
            if end in snap_rel_lookup:
                self.snapshot(start_offset + end)
            prev = end

        assert last_logits is not None
        return last_logits

    def fork_and_edit(
        self,
        edit_token_idx: int,
        new_token_ids: Union[mx.array, Sequence[int]],
        model: Optional[nn.Module] = None,
    ) -> mx.array:
        """
        Rewind the 16 Full-Attention KV caches to `edit_token_idx`, restore the 48 Gated DeltaNet
        states to `edit_token_idx`, and execute a warm prefill ONLY on `new_token_ids`.
        """
        return fork_and_edit(
            cache=self,
            edit_token_idx=edit_token_idx,
            new_token_ids=new_token_ids,
            model=model or self.model,
            prefill_step_size=self.prefill_step_size,
        )


def fork_and_edit(
    cache: Union[HybridTimelineCache, List[Any]],
    edit_token_idx: int,
    new_token_ids: Union[mx.array, Sequence[int]],
    model: Optional[nn.Module] = None,
    prefill_step_size: int = DEFAULT_PREFILL_STEP_SIZE,
) -> mx.array:
    """
    Top-level helper that rewinds the 16 KV caches to `edit_token_idx`, restores the 48
    Gated DeltaNet `ArraysCache` states to `edit_token_idx`, and runs a warm chunked prefill
    ONLY on `new_token_ids`.

    Args:
        cache: A `HybridTimelineCache` manager (or raw cache list if `edit_token_idx == 0`).
        edit_token_idx: Token index where the timeline branches.
        new_token_ids: 1D sequence or (B, S) `mx.array` of replacement/continuation token IDs.
        model: The language model to run warm prefill on (uses `cache.model` if omitted).
        prefill_step_size: Chunk size for prefill (must be 512, 1024, or 2048).

    Returns:
        Logits `mx.array` from the final warm-prefill chunk.
    """
    if prefill_step_size not in (512, 1024, 2048):
        raise ValueError(
            f"Hardware constraint: prefill_step_size must be 512, 1024, or 2048 (got {prefill_step_size})."
        )

    if isinstance(cache, HybridTimelineCache):
        timeline = cache
        target_model = model or timeline.model
    else:
        timeline = HybridTimelineCache(
            model=model, cache=cache, prefill_step_size=prefill_step_size
        )
        target_model = model

    if target_model is None:
        raise ValueError("`model` must be supplied to `fork_and_edit` or attached to `HybridTimelineCache`.")
    if hasattr(target_model, "eval") and getattr(target_model, "training", False):
        target_model.eval()
    resolve_decoder_layers(target_model)

    # 1. Rewind 16 Full-Attention KVCache layers by sequence slicing
    timeline.rewind_to(edit_token_idx)

    # 2. Restore 48 Gated DeltaNet ArraysCache layers from snapshot at edit_token_idx
    timeline.restore(edit_token_idx)

    # 3. Warm prefill ONLY on `new_token_ids` in chunks of `prefill_step_size`
    if not isinstance(new_token_ids, mx.array):
        new_list = [int(t) for t in new_token_ids]
        tokens = mx.array(new_list, dtype=mx.int32)
    else:
        tokens = new_token_ids
        new_list = [int(t) for t in tokens.reshape(-1).tolist()]
    if tokens.ndim == 1:
        tokens = tokens[None, :]

    num_new = int(tokens.shape[1])
    if num_new == 0:
        raise ValueError("`new_token_ids` must contain at least 1 token.")

    timeline.token_history = timeline.token_history[:edit_token_idx] + new_list

    last_logits: Optional[mx.array] = None
    for start in range(0, num_new, prefill_step_size):
        chunk = tokens[:, start : start + prefill_step_size]
        last_logits = target_model(chunk, cache=timeline.cache)
    mx.eval(last_logits)
    timeline.eval_all()

    assert last_logits is not None
    return last_logits
