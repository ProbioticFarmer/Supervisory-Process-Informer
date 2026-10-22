"""
attention_probe.py — Unfused QK^T Attention Matrix Inspector for Qwen3.8-27B (MLX)

Provides:
  - `capture_attention_maps(model, layers=None)`:
      Context manager that temporarily wraps the 16 full-attention layers
      (every 4th layer: `3, 7, 11, ..., 63`, with 24 Q heads, 4 KV heads,
      head_dim 256, partial RoPE dim 64) to explicitly compute and record
      the post-softmax attention weight matrices:
          A^{(l)} = softmax((Q K^T) / sqrt(d_k) + M)
  - `compute_attention_shift(attn_a, attn_b)`:
      Calculates per-layer, per-head Jensen-Shannon Divergence (JSD) and
      L1 distance between two forward/prefill passes (e.g., alpha=0.0 vs alpha=1.0,
      or pre-edit vs post-edit tokens).
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn

from subspace_projector import resolve_decoder_layers


FULL_ATTENTION_LAYERS_DEFAULT: Tuple[int, ...] = tuple(range(3, 64, 4))
ROUTING_HIGHLIGHT_LAYERS: Tuple[int, ...] = (31, 35, 39, 43, 47)


@dataclass
class HeadShiftMetric:
    layer_idx: int
    head_idx: int
    jsd_mean: float
    jsd_max: float
    l1_mean: float
    l1_max: float


@dataclass
class KeyTokenAttention:
    key_idx: int
    token_id: int
    token_str: str
    weight: float


@dataclass
class HeadTokenRoutingAttribution:
    layer_idx: int
    head_idx: int
    jsd_max: float
    jsd_mean: float
    l1_at_peak: float
    peak_query_rel_idx: int
    peak_query_abs_idx: int
    peak_query_token_id: int
    peak_query_token_str: str
    top_keys_alpha0: List[KeyTokenAttention]
    top_keys_alpha1: List[KeyTokenAttention]
    dist_alpha0: List[float]
    dist_alpha1: List[float]


class AttentionProbeSession:
    """Stores captured post-softmax attention matrices A^{(l)} keyed by layer index."""

    def __init__(self, target_layers: Sequence[int]):
        self.target_layers: List[int] = list(target_layers)
        self.attention_maps: Dict[int, mx.array] = {}

    def clear(self) -> None:
        self.attention_maps.clear()

    def record(self, layer_idx: int, attn_weights: mx.array) -> None:
        # Materialize post-softmax attention matrix in FP32 so graphs don't accumulate
        weights_fp32 = attn_weights.astype(mx.float32)
        mx.eval(weights_fp32)
        self.attention_maps[layer_idx] = weights_fp32

    def __getitem__(self, layer_idx: int) -> mx.array:
        return self.attention_maps[layer_idx]

    def keys(self):
        return self.attention_maps.keys()


def _unwrap_decoder_layer(layer: nn.Module) -> nn.Module:
    """Unwrap any ProjectedLayerWrapper to access the underlying DecoderLayer."""
    curr = layer
    while hasattr(curr, "base_layer") and isinstance(getattr(curr, "base_layer"), nn.Module):
        curr = curr.base_layer
    return curr


def _build_explicit_causal_additive_mask(
    q_len: int,
    k_len: int,
    mask: Optional[Any],
    dtype: mx.Dtype = mx.float32,
) -> Optional[mx.array]:
    """
    Convert an MLX attention mask (`"causal"`, boolean mask, or additive float mask)
    into an explicit additive mask M of shape broadcastable to `(B, H_q, q_len, k_len)`.
    """
    neg_inf = mx.array(-1e9 if dtype == mx.float16 else -1e30, dtype=dtype)
    zero = mx.array(0.0, dtype=dtype)

    if isinstance(mask, str):
        if mask == "causal":
            if q_len == 1:
                return None
            offset = k_len - q_len
            q_idx = mx.arange(offset, offset + q_len)[:, None]
            k_idx = mx.arange(k_len)[None, :]
            allowed = q_idx >= k_idx
            return mx.where(allowed, zero, neg_inf)[None, None, :, :]
        raise ValueError(f"Unsupported string mask mode: {mask}")

    if mask is None:
        if q_len > 1:
            offset = k_len - q_len
            q_idx = mx.arange(offset, offset + q_len)[:, None]
            k_idx = mx.arange(k_len)[None, :]
            allowed = q_idx >= k_idx
            return mx.where(allowed, zero, neg_inf)[None, None, :, :]
        return None

    if isinstance(mask, mx.array):
        if mask.dtype == mx.bool_:
            m = mx.where(mask, zero, neg_inf)
        else:
            m = mask.astype(dtype)
        while m.ndim < 4:
            m = mx.expand_dims(m, axis=0)
        return m

    return None


class UnfusedAttentionProbeWrapper(nn.Module):
    """
    Wraps a Qwen3.8-27B `Qwen3NextAttention` (`self_attn`) module to explicitly compute
    unfused post-softmax attention weights:
        A^{(l)} = softmax((Q K^T) / sqrt(d_k) + M)
    with Grouped-Query Attention (GQA: e.g. 24 Q heads, 4 KV heads -> 6x KV head repeat),
    q_norm, k_norm, partial RoPE (dim 64 out of 256), and output sigmoid gating.
    """

    def __init__(
        self,
        base_attn: nn.Module,
        layer_idx: int,
        session: AttentionProbeSession,
    ):
        super().__init__()
        self.base_attn = base_attn
        self.layer_idx = layer_idx
        self.session = session

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            base = self.get("base_attn", None)
            if base is not None:
                return getattr(base, name)
            raise

    def __call__(
        self,
        x: mx.array,
        mask: Optional[Any] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        attn = self.base_attn
        B, L, _ = x.shape

        # 1. q_proj projects to 2 * num_attention_heads * head_dim and splits into (queries, gate)
        q_proj_output = attn.q_proj(x)
        queries, gate = mx.split(
            q_proj_output.reshape(B, L, attn.num_attention_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(B, L, -1)

        # 2. Project K and V
        keys = attn.k_proj(x)
        values = attn.v_proj(x)

        # 3. Apply per-head q_norm and k_norm before partial RoPE (partial_rotary_factor = 0.25 -> 64 dims)
        queries = attn.q_norm(queries).transpose(0, 2, 1, 3)
        keys = attn.k_norm(keys.reshape(B, L, attn.num_key_value_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, attn.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        # 4. Apply partial RoPE and update KVCache if present
        if cache is not None:
            queries = attn.rope(queries, offset=cache.offset)
            keys = attn.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = attn.rope(queries)
            keys = attn.rope(keys)

        # 5. GQA repetition: 24 Q heads / 4 KV heads = 6 repeats per KV head
        n_q_heads = attn.num_attention_heads
        n_kv_heads = attn.num_key_value_heads
        if n_q_heads < n_kv_heads or n_q_heads % n_kv_heads != 0:
            raise ValueError(
                f"Invalid GQA head configuration at layer {self.layer_idx}: "
                f"n_q_heads={n_q_heads}, n_kv_heads={n_kv_heads}"
            )
        n_repeats = n_q_heads // n_kv_heads

        if n_repeats > 1:
            keys_expanded = mx.repeat(keys, n_repeats, axis=1)
            values_expanded = mx.repeat(values, n_repeats, axis=1)
        else:
            keys_expanded = keys
            values_expanded = values

        # 6. Compute explicit unfused softmax((Q K^T) / sqrt(d_k) + M) in FP32
        q_fp32 = queries.astype(mx.float32)
        k_fp32 = keys_expanded.astype(mx.float32)
        scores = mx.matmul(q_fp32, k_fp32.transpose(0, 1, 3, 2)) * float(attn.scale)

        q_len = int(scores.shape[-2])
        k_len = int(scores.shape[-1])
        additive_mask = _build_explicit_causal_additive_mask(
            q_len=q_len, k_len=k_len, mask=mask, dtype=mx.float32
        )
        if additive_mask is not None:
            scores = scores + additive_mask

        attn_weights = mx.softmax(scores, axis=-1, precise=True)
        self.session.record(self.layer_idx, attn_weights)

        # 7. Multiply attention weights by V, transpose, and multiply elementwise by mx.sigmoid(gate) before o_proj
        output = mx.matmul(attn_weights.astype(values_expanded.dtype), values_expanded)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return attn.o_proj(output * mx.sigmoid(gate))


def _get_model_layers(model: nn.Module) -> List[nn.Module]:
    return resolve_decoder_layers(model)


@contextmanager
def capture_attention_maps(
    model: nn.Module,
    layers: Optional[Iterable[int]] = None,
):
    """
    Context manager that temporarily wraps the 16 full-attention layers
    (indices `3, 7, 11, ..., 63` by default) with `UnfusedAttentionProbeWrapper`
    to compute and record post-softmax attention matrices `A^{(l)}`.
    """
    model_layers = _get_model_layers(model)

    if layers is None:
        target_indices = [
            idx
            for idx in range(len(model_layers))
            if not getattr(_unwrap_decoder_layer(model_layers[idx]), "is_linear", True)
        ]
        if not target_indices:
            target_indices = [
                i for i in FULL_ATTENTION_LAYERS_DEFAULT if i < len(model_layers)
            ]
    else:
        target_indices = [int(i) for i in layers]

    session = AttentionProbeSession(target_layers=target_indices)
    saved_modules: List[Tuple[nn.Module, nn.Module]] = []

    try:
        for idx in target_indices:
            dec_layer = _unwrap_decoder_layer(model_layers[idx])
            if getattr(dec_layer, "is_linear", False) or not hasattr(dec_layer, "self_attn"):
                raise ValueError(
                    f"Layer {idx} is a Gated DeltaNet linear-attention layer and does not have `self_attn`."
                )
            orig_attn = dec_layer.self_attn
            saved_modules.append((dec_layer, orig_attn))
            wrapped_attn = UnfusedAttentionProbeWrapper(
                base_attn=orig_attn,
                layer_idx=idx,
                session=session,
            )
            wrapped_attn.eval()
            dec_layer.self_attn = wrapped_attn
        yield session
    finally:
        for dec_layer, orig_attn in saved_modules:
            dec_layer.self_attn = orig_attn


def compute_attention_shift(
    attn_a: Dict[int, mx.array],
    attn_b: Dict[int, mx.array],
    eps: float = 1e-8,
) -> Dict[str, Any]:
    """
    Calculate per-layer, per-head Jensen-Shannon Divergence (JSD, in nats and bits)
    and L1 distance between two sets of captured post-softmax attention matrices
    (e.g., alpha=0.0 vs alpha=1.0, or pre-edit vs post-edit tokens).

    Args:
        attn_a: Mapping `layer_idx -> mx.array` of shape `(B, H_q, Q_len, K_len)`.
        attn_b: Mapping `layer_idx -> mx.array` of shape `(B, H_q, Q_len, K_len)`.
        eps: Numerical floor for log probabilities.

    Returns:
        Dictionary containing:
          - `per_layer`: Dict[layer_idx, {"jsd_per_head": ..., "l1_per_head": ..., ...}]
          - `ranked_heads`: List[HeadShiftMetric] sorted descending by `jsd_mean`
          - `mean_jsd`: float across all compared layers and heads
          - `mean_l1`: float across all compared layers and heads
    """
    common_layers = sorted(set(attn_a.keys()) & set(attn_b.keys()))
    if not common_layers:
        raise ValueError("`attn_a` and `attn_b` have no overlapping layer indices.")

    per_layer: Dict[int, Dict[str, Any]] = {}
    ranked_heads: List[HeadShiftMetric] = []

    for layer_idx in common_layers:
        p = attn_a[layer_idx].astype(mx.float32)
        q = attn_b[layer_idx].astype(mx.float32)

        # If comparing two runs with different query/key lengths, align on the suffix tokens
        q_len = min(int(p.shape[-2]), int(q.shape[-2]))
        k_len = min(int(p.shape[-1]), int(q.shape[-1]))
        p = p[..., -q_len:, :k_len]
        q = q[..., -q_len:, :k_len]

        # Re-normalize rows if k_len was truncated
        p = p / mx.maximum(mx.sum(p, axis=-1, keepdims=True), eps)
        q = q / mx.maximum(mx.sum(q, axis=-1, keepdims=True), eps)

        m = 0.5 * (p + q)

        p_safe = mx.maximum(p, eps)
        q_safe = mx.maximum(q, eps)
        m_safe = mx.maximum(m, eps)

        kl_pm = mx.sum(p * (mx.log(p_safe) - mx.log(m_safe)), axis=-1)
        kl_qm = mx.sum(q * (mx.log(q_safe) - mx.log(m_safe)), axis=-1)
        jsd_rows = mx.maximum(0.5 * (kl_pm + kl_qm), 0.0)  # (B, H_q, Q_len)

        l1_rows = mx.sum(mx.abs(p - q), axis=-1)  # (B, H_q, Q_len)

        # Aggregate over batch and query positions -> (H_q,)
        jsd_mean_heads = mx.mean(jsd_rows, axis=(0, 2))
        jsd_max_heads = mx.max(jsd_rows, axis=(0, 2))
        l1_mean_heads = mx.mean(l1_rows, axis=(0, 2))
        l1_max_heads = mx.max(l1_rows, axis=(0, 2))
        mx.eval(jsd_mean_heads, jsd_max_heads, l1_mean_heads, l1_max_heads)

        jsd_mean_list = [float(x) for x in jsd_mean_heads.tolist()]
        jsd_max_list = [float(x) for x in jsd_max_heads.tolist()]
        l1_mean_list = [float(x) for x in l1_mean_heads.tolist()]
        l1_max_list = [float(x) for x in l1_max_heads.tolist()]

        per_layer[layer_idx] = {
            "jsd_per_head": jsd_mean_list,
            "jsd_max_per_head": jsd_max_list,
            "l1_per_head": l1_mean_list,
            "l1_max_per_head": l1_max_list,
            "layer_jsd_mean": float(sum(jsd_mean_list) / max(len(jsd_mean_list), 1)),
            "layer_l1_mean": float(sum(l1_mean_list) / max(len(l1_mean_list), 1)),
        }

        for h_idx in range(len(jsd_mean_list)):
            ranked_heads.append(
                HeadShiftMetric(
                    layer_idx=layer_idx,
                    head_idx=h_idx,
                    jsd_mean=jsd_mean_list[h_idx],
                    jsd_max=jsd_max_list[h_idx],
                    l1_mean=l1_mean_list[h_idx],
                    l1_max=l1_max_list[h_idx],
                )
            )

    ranked_heads.sort(key=lambda item: item.jsd_mean, reverse=True)
    ranked_heads_by_max = sorted(ranked_heads, key=lambda item: item.jsd_max, reverse=True)
    mean_jsd = float(
        sum(h.jsd_mean for h in ranked_heads) / max(len(ranked_heads), 1)
    )
    mean_l1 = float(
        sum(h.l1_mean for h in ranked_heads) / max(len(ranked_heads), 1)
    )

    return {
        "per_layer": per_layer,
        "ranked_heads": ranked_heads,
        "ranked_heads_by_max": ranked_heads_by_max,
        "mean_jsd": mean_jsd,
        "mean_l1": mean_l1,
    }


def _decode_single_token(tokenizer: Optional[Any], token_id: int) -> str:
    if tokenizer is None:
        return f"<tok:{token_id}>"
    try:
        s = tokenizer.decode([int(token_id)])
        return repr(s)
    except Exception:
        return f"<tok:{token_id}>"


def attribute_top_heads_token_routing(
    attn_alpha0: Dict[int, mx.array],
    attn_alpha1: Dict[int, mx.array],
    full_sequence_token_ids: Sequence[int],
    query_start_idx: int,
    tokenizer: Optional[Any] = None,
    top_n_heads: int = 5,
    top_k_keys: int = 3,
    eps: float = 1e-8,
) -> List[HeadTokenRoutingAttribution]:
    """
    For the top `top_n_heads` full-attention heads ranked by Max JSD:
      1. Identify the exact query token index `q*` (both relative within the edited `<think>`
         clause and absolute across the full sequence) where `JSD(A_alpha0[q*], A_alpha1[q*])` peaks.
      2. Extract the top-`top_k_keys` key tokens `(k1, k2, k3)` (token index, attention weight,
         and decoded token string) attended to under `alpha = 0.0` vs `alpha = 1.0`.
      3. Record the full sequence attention distributions `A_alpha0[q*, :k_valid]` and
         `A_alpha1[q*, :k_valid]` for plotting.
    """
    shift = compute_attention_shift(attn_alpha0, attn_alpha1, eps=eps)
    top_heads_by_max: List[HeadShiftMetric] = shift["ranked_heads_by_max"][:top_n_heads]

    attributions: List[HeadTokenRoutingAttribution] = []
    seq_ids = [int(t) for t in full_sequence_token_ids]

    for head_metric in top_heads_by_max:
        layer_idx = head_metric.layer_idx
        head_idx = head_metric.head_idx

        p_full = attn_alpha0[layer_idx].astype(mx.float32)
        q_full = attn_alpha1[layer_idx].astype(mx.float32)

        q_len = min(int(p_full.shape[-2]), int(q_full.shape[-2]))
        k_len = min(int(p_full.shape[-1]), int(q_full.shape[-1]))

        # Slice batch 0, target head, and aligned suffix/prefix dimensions -> (Q_len, K_len)
        p_head = p_full[0, head_idx, -q_len:, :k_len]
        q_head = q_full[0, head_idx, -q_len:, :k_len]

        p_head = p_head / mx.maximum(mx.sum(p_head, axis=-1, keepdims=True), eps)
        q_head = q_head / mx.maximum(mx.sum(q_head, axis=-1, keepdims=True), eps)
        m_head = 0.5 * (p_head + q_head)

        p_safe = mx.maximum(p_head, eps)
        q_safe = mx.maximum(q_head, eps)
        m_safe = mx.maximum(m_head, eps)

        kl_pm = mx.sum(p_head * (mx.log(p_safe) - mx.log(m_safe)), axis=-1)
        kl_qm = mx.sum(q_head * (mx.log(q_safe) - mx.log(m_safe)), axis=-1)
        jsd_per_q = mx.maximum(0.5 * (kl_pm + kl_qm), 0.0)  # (Q_len,)
        l1_per_q = mx.sum(mx.abs(p_head - q_head), axis=-1)  # (Q_len,)

        peak_q_rel = int(mx.argmax(jsd_per_q).item())
        peak_jsd_val = float(jsd_per_q[peak_q_rel].item())
        peak_l1_val = float(l1_per_q[peak_q_rel].item())

        peak_q_abs = int(query_start_idx + peak_q_rel)
        q_tok_id = seq_ids[peak_q_abs] if 0 <= peak_q_abs < len(seq_ids) else -1
        q_tok_str = _decode_single_token(tokenizer, q_tok_id)

        # Causal valid keys up to peak_q_abs + 1 (or k_len)
        valid_k_len = min(k_len, peak_q_abs + 1) if peak_q_abs >= 0 else k_len
        p_row = p_head[peak_q_rel, :valid_k_len]
        q_row = q_head[peak_q_rel, :valid_k_len]
        mx.eval(p_row, q_row)

        p_list = [float(x) for x in p_row.tolist()]
        q_list = [float(x) for x in q_row.tolist()]

        p_sorted_idx = sorted(range(len(p_list)), key=lambda i: p_list[i], reverse=True)[:top_k_keys]
        q_sorted_idx = sorted(range(len(q_list)), key=lambda i: q_list[i], reverse=True)[:top_k_keys]

        top_keys_a0 = [
            KeyTokenAttention(
                key_idx=k_i,
                token_id=seq_ids[k_i] if k_i < len(seq_ids) else -1,
                token_str=_decode_single_token(tokenizer, seq_ids[k_i] if k_i < len(seq_ids) else -1),
                weight=p_list[k_i],
            )
            for k_i in p_sorted_idx
        ]
        top_keys_a1 = [
            KeyTokenAttention(
                key_idx=k_i,
                token_id=seq_ids[k_i] if k_i < len(seq_ids) else -1,
                token_str=_decode_single_token(tokenizer, seq_ids[k_i] if k_i < len(seq_ids) else -1),
                weight=q_list[k_i],
            )
            for k_i in q_sorted_idx
        ]

        attributions.append(
            HeadTokenRoutingAttribution(
                layer_idx=layer_idx,
                head_idx=head_idx,
                jsd_max=peak_jsd_val,
                jsd_mean=head_metric.jsd_mean,
                l1_at_peak=peak_l1_val,
                peak_query_rel_idx=peak_q_rel,
                peak_query_abs_idx=peak_q_abs,
                peak_query_token_id=q_tok_id,
                peak_query_token_str=q_tok_str,
                top_keys_alpha0=top_keys_a0,
                top_keys_alpha1=top_keys_a1,
                dist_alpha0=p_list,
                dist_alpha1=q_list,
            )
        )

    return attributions


def _robust_normalize_01(values: List[float], eps: float = 1e-9) -> List[float]:
    """
    Normalize a 1D non-negative sequence into `[0.0, 1.0]` while preserving strict
    monotonic ordering so the true peak uniquely reaches `1.0` and top-quartile chunk
    onsets / anchors are not crushed by a single outlier.
    """
    if not values:
        return []
    n = len(values)
    vmin = min(values)
    vmax = max(values)
    span = vmax - vmin
    if span <= eps:
        return [0.0 for _ in values]
    sorted_vals = sorted(values)
    p95_idx = min(n - 1, max(0, int(0.95 * (n - 1))))
    p95_span = max(sorted_vals[p95_idx] - vmin, span * 0.5, eps)
    out: List[float] = []
    for v in values:
        dv = max(0.0, v - vmin)
        if dv <= p95_span or vmax <= sorted_vals[p95_idx] + eps:
            # Linear up to 0.92 at p95 (or 1.0 if p95 == vmax)
            top_cap = 1.0 if vmax <= sorted_vals[p95_idx] + eps else 0.92
            out.append(min(1.0, (dv / p95_span) * top_cap))
        else:
            # Strictly monotonic tail from 0.92 -> 1.0 between p95 and vmax
            tail_frac = (dv - p95_span) / max(span - p95_span, eps)
            out.append(min(1.0, 0.92 + 0.08 * tail_frac))
    max_out = max(out)
    if max_out > eps:
        return [min(1.0, max(0.0, x / max_out)) for x in out]
    return out


def compute_preplan_and_anchor_rhythm(
    attn_maps: Dict[int, mx.array],
    think_start_idx: int,
    query_offset: Optional[int] = None,
    window_w: int = 32,
    horizon_lo: int = 2,
    horizon_hi: int = 48,
    eps: float = 1e-9,
) -> Dict[str, Any]:
    """
    Compute task-agnostic Preplan (`WAAD_t`) and Anchor (`FAI_s`) structural rhythm telemetry
    across the `<think>` trace `[think_start_idx .. N-1]` from the captured 16-layer
    full-attention maps (`L = 16` layers, `H = 24` heads, total `384` heads), following
    Li et al. ("Attention Illuminates LLM Reasoning: The Preplan-and-Anchor Rhythm").

    Steps:
      1. Sink-Free `<think>` Submatrix Normalization:
         Extract the causal `<think>` submatrix `A_think^{(l,h)} in R^{M x M}` and
         row-normalize over the causal lower triangle `j <= i` (`+ 1e-9` for stability).
      2. Automatic Local vs. Global Head Classification:
         Compute mean backward attention distance `d^{(l,h)}` for all `L * H` heads and
         partition at the median into Locally Focused Heads (`H_loc`, bottom 50%) and
         Globally Focused Heads (`H_glob`, top 50%).
      3. Per-Token Preplan (`WAAD_i` & `Delta WAAD_i`) and Anchor (`FAI_j`) Scores:
         - `WAAD_i = sum_{j=0}^i A_loc[i, j] * min(i - j, W)` (with `W = 32`)
         - `Delta WAAD_i = |WAAD_i - WAAD_{min(i+1, M-1)}|`
         - `FAI_j = (1 / |T(j)|) * sum_{i in T(j)} A_glob[i, j]` for `T(j) = [j + H_lo .. min(M-1, j + H_hi)]`
         - Identify `peak_preplan_token_idx`, `peak_anchor_token_idx`, and coupled
           `Preplan -> Anchor` junctures (top-quartile `preplan_score` followed within
           `1..6` tokens by a top-quartile `anchor_score`).
    """
    if not attn_maps:
        return {
            "span_start_idx": int(think_start_idx),
            "num_think_tokens": 0,
            "num_total_heads": 0,
            "num_local_heads": 0,
            "num_global_heads": 0,
            "median_head_distance": 0.0,
            "local_heads": [],
            "global_heads": [],
            "waad_raw": [],
            "delta_waad_raw": [],
            "preplan_scores": [],
            "fai_raw": [],
            "anchor_scores": [],
            "peak_preplan_token_idx": int(think_start_idx),
            "peak_anchor_token_idx": int(think_start_idx),
            "coupled_junctures": [],
            "coupled_token_indices": [],
            "juncture_roles": {},
        }

    sorted_layers = sorted(attn_maps.keys())
    sample_mat = attn_maps[sorted_layers[0]]
    q_len_raw = int(sample_mat.shape[-2])
    k_len_raw = int(sample_mat.shape[-1])

    inferred_q_offset = max(0, k_len_raw - q_len_raw) if query_offset is None else int(query_offset)
    span_start = max(int(think_start_idx), inferred_q_offset)
    q_skip = max(0, span_start - inferred_q_offset)
    m_len = max(0, min(q_len_raw - q_skip, k_len_raw - span_start))

    if m_len <= 0:
        return {
            "span_start_idx": int(span_start),
            "num_think_tokens": 0,
            "num_total_heads": 0,
            "num_local_heads": 0,
            "num_global_heads": 0,
            "median_head_distance": 0.0,
            "local_heads": [],
            "global_heads": [],
            "waad_raw": [],
            "delta_waad_raw": [],
            "preplan_scores": [],
            "fai_raw": [],
            "anchor_scores": [],
            "peak_preplan_token_idx": int(span_start),
            "peak_anchor_token_idx": int(span_start),
            "coupled_junctures": [],
            "coupled_token_indices": [],
            "juncture_roles": {},
        }

    # 1. Extract M x M causal <think> submatrix across all (layer, head) pairs
    submatrices: List[mx.array] = []
    head_coords: List[Tuple[int, int]] = []
    for l_idx in sorted_layers:
        mat = attn_maps[l_idx].astype(mx.float32)  # (B, H, Q, K)
        n_heads = int(mat.shape[1])
        sub = mat[0, :, q_skip : q_skip + m_len, span_start : span_start + m_len]  # (H, M, M)
        submatrices.append(sub)
        for h_idx in range(n_heads):
            head_coords.append((int(l_idx), int(h_idx)))

    all_sub = mx.concatenate(submatrices, axis=0)  # (N_heads, M, M)
    n_total_heads = int(all_sub.shape[0])

    # Causal lower-triangular mask j <= i
    i_idx = mx.arange(m_len, dtype=mx.float32)[:, None]
    j_idx = mx.arange(m_len, dtype=mx.float32)[None, :]
    causal_mask = (j_idx <= i_idx).astype(mx.float32)  # (M, M)

    masked_sub = all_sub * causal_mask[None, :, :]
    row_sums = mx.sum(masked_sub, axis=-1, keepdims=True) + float(eps)
    a_tilde = masked_sub / row_sums  # (N_heads, M, M)

    # 2. Automatic Local vs. Global Head Classification via mean backward distance d^{(l,h)}
    dist_mat = mx.maximum(i_idx - j_idx, 0.0)  # (M, M)
    # d^{(u)} = (1 / M) * sum_{i=0}^{M-1} sum_{j=0}^i a_tilde[u, i, j] * (i - j)
    head_distances_arr = mx.mean(mx.sum(a_tilde * dist_mat[None, :, :], axis=-1), axis=-1)  # (N_heads,)
    mx.eval(head_distances_arr)
    head_distances = [float(x) for x in head_distances_arr.tolist()]

    ranked_head_indices = sorted(range(n_total_heads), key=lambda u: (head_distances[u], u))
    half_heads = max(1, n_total_heads // 2)
    local_head_indices = ranked_head_indices[:half_heads]
    global_head_indices = (
        ranked_head_indices[half_heads:] if n_total_heads > 1 else ranked_head_indices
    )
    median_dist = float(head_distances[ranked_head_indices[half_heads - 1]])

    a_loc = mx.mean(a_tilde[mx.array(local_head_indices, dtype=mx.int32)], axis=0)  # (M, M)
    a_glob = mx.mean(a_tilde[mx.array(global_head_indices, dtype=mx.int32)], axis=0)  # (M, M)

    # 3A. Compute Windowed Average Attention Distance (WAAD_i) & step-transition spike (Delta WAAD_i)
    win_dist_mat = mx.minimum(dist_mat, float(window_w))  # min(i - j, W)
    waad_arr = mx.sum(a_loc * win_dist_mat, axis=-1)  # (M,)
    mx.eval(waad_arr, a_glob)
    waad_raw = [float(x) for x in waad_arr.tolist()]

    delta_waad_raw: List[float] = []
    for i in range(m_len):
        next_i = min(i + 1, m_len - 1)
        delta_waad_raw.append(abs(waad_raw[i] - waad_raw[next_i]))

    waad_norm = _robust_normalize_01(waad_raw, eps=eps)
    delta_norm = _robust_normalize_01(delta_waad_raw, eps=eps)
    combined_preplan = [0.65 * w_n + 0.35 * d_n for w_n, d_n in zip(waad_norm, delta_norm)]
    preplan_scores = _robust_normalize_01(combined_preplan, eps=eps)

    # 3B. Compute Future Attention Influence (FAI_j — Anchor Score)
    a_glob_list: List[List[float]] = a_glob.tolist()
    fai_raw: List[float] = []
    h_lo = max(1, int(horizon_lo))
    h_hi = max(h_lo, int(horizon_hi))

    for j in range(m_len):
        start_i = j + h_lo
        end_i = min(m_len - 1, j + h_hi)
        if start_i <= end_i:
            t_count = end_i - start_i + 1
            col_sum = sum(a_glob_list[i][j] for i in range(start_i, end_i + 1))
            fai_raw.append(float(col_sum / t_count))
        else:
            # Fallback for final tail tokens where [j + H_lo .. M-1] is empty
            if j + 1 < m_len:
                fai_raw.append(float(a_glob_list[j + 1][j]) * 0.25)
            else:
                fai_raw.append(0.0)

    anchor_scores = _robust_normalize_01(fai_raw, eps=eps)

    # 4. Identify Peak Preplan Token, Peak Anchor Token, and Coupled Preplan -> Anchor Junctures
    peak_preplan_rel = max(range(m_len), key=lambda i: preplan_scores[i])
    peak_anchor_rel = max(range(m_len), key=lambda j: anchor_scores[j])
    peak_preplan_token_idx = int(span_start + peak_preplan_rel)
    peak_anchor_token_idx = int(span_start + peak_anchor_rel)

    sorted_pre = sorted(preplan_scores)
    sorted_anc = sorted(anchor_scores)
    q75_idx = min(m_len - 1, max(0, int(0.75 * (m_len - 1))))
    pre_q75 = max(sorted_pre[q75_idx], 0.50) if m_len >= 4 else 0.50
    anc_q75 = max(sorted_anc[q75_idx], 0.50) if m_len >= 4 else 0.50

    coupled_junctures: List[Dict[str, Any]] = []
    coupled_set = set()
    juncture_roles: Dict[int, str] = {}

    for p_rel in range(m_len):
        if preplan_scores[p_rel] < pre_q75:
            continue
        # Check if p_rel is a local maximum or top-quartile onset followed within 1..6 tokens by an anchor peak
        search_hi = min(m_len - 1, p_rel + 6)
        if p_rel + 1 > search_hi:
            continue
        best_a_rel: Optional[int] = None
        best_a_score = -1.0
        for a_rel in range(p_rel + 1, search_hi + 1):
            if anchor_scores[a_rel] >= anc_q75 and anchor_scores[a_rel] > best_a_score:
                best_a_score = anchor_scores[a_rel]
                best_a_rel = a_rel
        if best_a_rel is not None:
            p_abs = int(span_start + p_rel)
            a_abs = int(span_start + best_a_rel)
            coupled_junctures.append(
                {
                    "preplan_idx": p_abs,
                    "anchor_idx": a_abs,
                    "distance": int(best_a_rel - p_rel),
                    "preplan_score": round(float(preplan_scores[p_rel]), 4),
                    "anchor_score": round(float(anchor_scores[best_a_rel]), 4),
                }
            )
            coupled_set.add(p_abs)
            coupled_set.add(a_abs)
            juncture_roles[p_abs] = (
                "both" if juncture_roles.get(p_abs) == "anchor" else "preplan"
            )
            juncture_roles[a_abs] = (
                "both" if juncture_roles.get(a_abs) == "preplan" else "anchor"
            )

    return {
        "span_start_idx": int(span_start),
        "num_think_tokens": int(m_len),
        "num_total_heads": int(n_total_heads),
        "num_local_heads": len(local_head_indices),
        "num_global_heads": len(global_head_indices),
        "median_head_distance": round(median_dist, 4),
        "local_heads": [
            {"layer": head_coords[u][0], "head": head_coords[u][1], "mean_distance": round(head_distances[u], 4)}
            for u in local_head_indices[:10]
        ],
        "global_heads": [
            {"layer": head_coords[u][0], "head": head_coords[u][1], "mean_distance": round(head_distances[u], 4)}
            for u in reversed(global_head_indices[-10:])
        ],
        "waad_raw": [round(float(x), 4) for x in waad_raw],
        "delta_waad_raw": [round(float(x), 4) for x in delta_waad_raw],
        "preplan_scores": [round(float(x), 4) for x in preplan_scores],
        "fai_raw": [round(float(x), 6) for x in fai_raw],
        "anchor_scores": [round(float(x), 4) for x in anchor_scores],
        "peak_preplan_token_idx": peak_preplan_token_idx,
        "peak_anchor_token_idx": peak_anchor_token_idx,
        "coupled_junctures": coupled_junctures,
        "coupled_token_indices": sorted(coupled_set),
        "juncture_roles": juncture_roles,
    }

