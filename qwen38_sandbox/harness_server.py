"""
harness_server.py — Persistent Stateful Server & Interactive <think> Fork-Edit Workbench
for Qwen3.8-27B on Apple Silicon (M4 Max, 128 GB Unified Memory)

Keeps a singleton `SandboxEngine` hot in unified memory (`HybridTimelineCache` + `SubspaceProjector`
+ `capture_attention_maps`) and exposes:
  1. `GET  /v1/models` & `POST /v1/chat/completions` (OpenAI-compatible streaming SSE & JSON API
     with `alpha`, `active_layers`, and `think_gated` auto-release at `</think>`).
  2. `POST /api/prefill_and_probe`: Supports both explicit `clause_a_text` prefill and **Organic
     `<think>` Generation Mode** (`clause_a_text=""`, `generate_organic_tokens=48`, `organic_alpha`),
     automatically snapshotting the 48 Gated DeltaNet `ArraysCache` states at `think_start_idx`
     and every `snapshot_stride=8` tokens inside `<think>`, and capturing unfused attention maps
     across all 16 full-attention layers (`384` heads) to return:
       - Per-token 5-Head Prompt-Lookback Scores (`L35.H21`, `L39.H1`, `L43.H7`, `L43.H1`, `L39.H0`)
       - Task-Agnostic Preplan Rhythm (`preplan_score` / `waad_raw`) & Anchor Influence (`anchor_score` / `fai_raw`)
       - Coupled `Preplan -> Anchor` junctures (`coupled_junctures`, `is_coupled_juncture`)
  3. `POST /api/fork_and_edit`: Rewinds to any arbitrary `fork_token_idx` inside `<think>` (or
     inside a previously forked continuation stream!) in `< 50 ms`, warm-prefills `edited_text`
     under `alpha` / `think_gated`, decodes `max_new_tokens` continuation tokens with stride-8
     snapshots and unfused 384-head rhythm + 5-head lookback probing, and automatically logs a
     counterfactual trajectory record to `outputs/spi_traces.jsonl`.
  4. `POST /api/promote_to_baseline`: Promotes the active forked + continuation stream to the
     baseline state for infinite multi-hop reasoning tree surgery.
  5. `GET  /api/spi_traces` & `POST /api/spi_traces/label`: Inspects and labels logged SPI
     counterfactual trajectories (`"accepted"`, `"candidate"`, `"rejected"`).
  6. `GET  /`: Self-contained dark-mode Single-Page Workbench UI with live 4-mode Telemetry
     Heatmap switcher (`Prompt Lookback`, `Preplan Rhythm (WAAD)`, `Anchor Influence (FAI)`,
     `Coupled Preplan->Anchor Junctures`), Organic `<think>` Generation, recursive continuation
     forking, and one-click **"★ Mark as Gold SPI Pair"** labeling.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

from attention_probe import (
    capture_attention_maps,
    compute_attention_shift,
    compute_preplan_and_anchor_rhythm,
)
from benchmark_and_experiment import (
    DEFAULT_8BIT_LOCAL_PATH,
    DEFAULT_ACTIVE_LAYERS,
    LOOKBACK_HEADS_5,
    enforce_metal_command_buffer_invariant,
    safe_load_model_and_tokenizer,
    warmup_model_and_kernels,
)
from hybrid_cache import (
    DEFAULT_PREFILL_STEP_SIZE,
    HybridTimelineCache,
)
from subspace_projector import (
    HIDDEN_SIZE_DEFAULT,
    SubspaceProjector,
    load_subspace_vectors,
    resolve_decoder_layers,
)


DEFAULT_VECTOR_PATH = "./vectors/Qwen3.8-27B-abliterated-cyber-GLP-49-L10-58-a1.gguf"
DEFAULT_SNAPSHOT_STRIDE = 8
DEFAULT_SPI_LOG_PATH = Path(__file__).resolve().parent / "outputs" / "spi_traces.jsonl"


def _parse_active_layers(
    spec: Optional[Union[Sequence[int], str]],
    default_layers: Sequence[int] = DEFAULT_ACTIVE_LAYERS,
) -> List[int]:
    if spec is None:
        return list(default_layers)
    if isinstance(spec, str):
        s = spec.strip()
        if not s:
            return list(default_layers)
        if "-" in s and "," not in s:
            a, b = s.split("-", 1)
            layers = list(range(int(a.strip()), int(b.strip())))
        elif ".." in s:
            a, b = s.split("..", 1)
            layers = list(range(int(a.strip()), int(b.strip()) + 1))
        else:
            layers = [int(x.strip()) for x in s.split(",") if x.strip()]
    else:
        layers = [int(x) for x in spec]
    assert 0 not in layers, "Invariant violation: Layer 0 must NEVER be projected (`0 not in active_layers`)."
    return layers


class SimpleFallbackTokenizer:
    """
    Deterministic fallback tokenizer for synthetic unit tests when no HuggingFace
    tokenizer is attached. Maps special tags (`<think>`, `</think>`) to fixed IDs.
    """

    THINK_OPEN_ID = 150
    THINK_CLOSE_ID = 152
    NEWLINE_ID = 151

    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self._id_to_piece: Dict[int, str] = {
            self.THINK_OPEN_ID: "<think>",
            self.NEWLINE_ID: "\n",
            self.THINK_CLOSE_ID: "</think>",
        }

    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        ids: List[int] = []
        remaining = text
        while remaining:
            if remaining.startswith("<think>\n"):
                ids.extend([self.THINK_OPEN_ID, self.NEWLINE_ID])
                remaining = remaining[len("<think>\n") :]
            elif remaining.startswith("<think>"):
                ids.append(self.THINK_OPEN_ID)
                remaining = remaining[len("<think>") :]
            elif remaining.startswith("</think>"):
                ids.append(self.THINK_CLOSE_ID)
                remaining = remaining[len("</think>") :]
            else:
                next_tag = min(
                    [
                        idx
                        for idx in (
                            remaining.find("<think>"),
                            remaining.find("</think>"),
                        )
                        if idx != -1
                    ]
                    or [len(remaining)]
                )
                chunk = remaining[:next_tag]
                remaining = remaining[next_tag:]
                for word in chunk.split(" "):
                    if not word:
                        continue
                    tid = (sum(ord(c) for c in word) % (self.vocab_size - 20)) + 10
                    if tid in (self.THINK_OPEN_ID, self.THINK_CLOSE_ID, self.NEWLINE_ID):
                        tid = 19
                    self._id_to_piece[tid] = word + " "
                    ids.append(tid)
        return ids

    def decode(self, token_ids: Sequence[int]) -> str:
        pieces: List[str] = []
        for tid in token_ids:
            t_int = int(tid)
            if t_int in self._id_to_piece:
                pieces.append(self._id_to_piece[t_int])
            else:
                pieces.append(f"tok_{t_int} ")
        return "".join(pieces)


def _decode_single_token(tokenizer: Any, token_id: int) -> str:
    if tokenizer is not None and hasattr(tokenizer, "decode"):
        try:
            return str(tokenizer.decode([int(token_id)]))
        except Exception:
            return f"<{token_id}>"
    return f"<{token_id}>"


class SPITrajectoryLogger:
    """
    Automatic Counterfactual Trajectory Logger (`spi_traces.jsonl`).
    Appends a structured JSON Lines record on every `/api/fork_and_edit` intervention
    and supports listing and re-labeling traces (`"accepted"`, `"candidate"`, `"rejected"`).
    """

    def __init__(self, log_path: Union[str, Path] = DEFAULT_SPI_LOG_PATH):
        self.log_path = Path(log_path)
        self._lock = threading.Lock()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def append_trace(self, record: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def read_all(self) -> List[Dict[str, Any]]:
        with self._lock:
            if not self.log_path.exists():
                return []
            traces: List[Dict[str, Any]] = []
            with self.log_path.open("r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        traces.append(json.loads(s))
                    except Exception:
                        continue
            return traces

    def count(self) -> int:
        return len(self.read_all())

    def list_recent(self, limit: int = 25) -> Dict[str, Any]:
        all_traces = self.read_all()
        recent = all_traces[-max(1, int(limit)) :]
        summaries = [
            {
                "trace_id": t.get("trace_id"),
                "timestamp": t.get("timestamp"),
                "model_path": t.get("model_path"),
                "think_start_idx": t.get("think_start_idx"),
                "fork_token_idx": t.get("fork_token_idx"),
                "label": t.get("label", "candidate"),
                "causal_deltas": t.get("causal_deltas", {}),
                "rejected_branch": t.get("rejected_branch", {}),
                "repaired_branch": t.get("repaired_branch", {}),
            }
            for t in reversed(recent)
        ]
        return {
            "count": len(all_traces),
            "log_path": str(self.log_path),
            "traces": summaries,
        }

    def update_label(
        self,
        label: str = "accepted",
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if label not in ("accepted", "candidate", "rejected"):
            raise ValueError(
                f"Invalid label {label!r}; must be 'accepted', 'candidate', or 'rejected'."
            )
        with self._lock:
            if not self.log_path.exists():
                raise ValueError("No SPI traces logged yet.")
            traces: List[Dict[str, Any]] = []
            with self.log_path.open("r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        traces.append(json.loads(s))
            if not traces:
                raise ValueError("No SPI traces logged yet.")

            target_idx: Optional[int] = None
            if trace_id is None or not str(trace_id).strip():
                target_idx = len(traces) - 1
            else:
                for i in range(len(traces) - 1, -1, -1):
                    if traces[i].get("trace_id") == trace_id:
                        target_idx = i
                        break
            if target_idx is None:
                raise ValueError(f"trace_id={trace_id!r} not found in {self.log_path}.")

            traces[target_idx]["label"] = label
            traces[target_idx]["labeled_at"] = datetime.now(timezone.utc).isoformat()
            updated_trace_id = traces[target_idx].get("trace_id")

            tmp_path = self.log_path.with_suffix(".jsonl.tmp")
            with tmp_path.open("w", encoding="utf-8") as f:
                for t in traces:
                    f.write(json.dumps(t, ensure_ascii=False) + "\n")
            tmp_path.replace(self.log_path)

            return {
                "updated": True,
                "trace_id": updated_trace_id,
                "label": label,
                "count": len(traces),
                "log_path": str(self.log_path),
            }


class SandboxEngine:
    """
    Stateful singleton holding the Qwen3.8-27B model, `SubspaceProjector`,
    `HybridTimelineCache`, and `SPITrajectoryLogger` hot in Apple Silicon Unified Memory.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        projector: SubspaceProjector,
        model_id: str = "Qwen3.8-27B-8bit-local",
        prefill_step_size: int = DEFAULT_PREFILL_STEP_SIZE,
        snapshot_stride: int = DEFAULT_SNAPSHOT_STRIDE,
        monitor_heads: Sequence[Tuple[int, int]] = LOOKBACK_HEADS_5,
        spi_log_path: Optional[Union[str, Path]] = None,
    ):
        enforce_metal_command_buffer_invariant()
        self.lock = threading.Lock()
        self.model = model
        self.tokenizer = tokenizer if tokenizer is not None else SimpleFallbackTokenizer()
        self.projector = projector
        self.model_id = model_id
        self.prefill_step_size = prefill_step_size
        self.snapshot_stride = max(1, int(snapshot_stride))
        self.monitor_heads: Tuple[Tuple[int, int], ...] = tuple(monitor_heads)
        self.spi_logger = SPITrajectoryLogger(
            log_path=spi_log_path if spi_log_path is not None else DEFAULT_SPI_LOG_PATH
        )

        if hasattr(self.model, "eval"):
            self.model.eval()
        resolve_decoder_layers(self.model)
        self.projector.attach(self.model)
        self.projector.configure(alpha=0.0, enabled=False, clear_token_slice=True)

        self.timeline = HybridTimelineCache(
            model=self.model, prefill_step_size=self.prefill_step_size
        )
        self.think_start_idx: int = 0
        self.baseline_token_ids: List[int] = []
        self.active_token_ids: List[int] = []
        self.baseline_token_records: List[Dict[str, Any]] = []
        self.active_token_records: List[Dict[str, Any]] = []
        self.baseline_think_attn: Dict[int, mx.array] = {}
        self.active_think_attn: Dict[int, mx.array] = {}
        self.last_rhythm_info: Dict[str, Any] = {}

        # Identify `</think>` token IDs if present in tokenizer
        self.think_close_token_ids: List[int] = []
        try:
            enc_close = self.tokenizer.encode("</think>")
            if enc_close:
                self.think_close_token_ids = [int(x) for x in enc_close]
        except Exception:
            self.think_close_token_ids = []

    def _compute_per_token_lookback_records(
        self,
        all_token_ids: Sequence[int],
        think_start_idx: int,
        attn_maps: Dict[int, mx.array],
        query_offset: int,
        think_end_token_idx: Optional[int] = None,
        rhythm_info: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Compute per-token telemetry across `all_token_ids`:
          1. 5-Head Prompt-Lookback Scores (`L35.H21`, `L39.H1`, `L43.H7`, `L43.H1`, `L39.H0`)
          2. Task-Agnostic Preplan Rhythm (`preplan_score` / `waad_raw`)
          3. Task-Agnostic Anchor Influence (`anchor_score` / `fai_raw`)
          4. Coupled Preplan -> Anchor Juncture flags (`is_coupled_juncture`, `juncture_role`)
        """
        if rhythm_info is None:
            rhythm_info = compute_preplan_and_anchor_rhythm(
                attn_maps=attn_maps,
                think_start_idx=think_start_idx,
                query_offset=query_offset,
            )

        rhythm_span_start = int(rhythm_info.get("span_start_idx", max(think_start_idx, query_offset)))
        waad_raw_list: List[float] = rhythm_info.get("waad_raw", [])
        preplan_list: List[float] = rhythm_info.get("preplan_scores", [])
        fai_raw_list: List[float] = rhythm_info.get("fai_raw", [])
        anchor_list: List[float] = rhythm_info.get("anchor_scores", [])
        coupled_indices = set(rhythm_info.get("coupled_token_indices", []))
        juncture_roles: Dict[int, str] = rhythm_info.get("juncture_roles", {})

        head_keys = [f"L{l}.H{h}" for l, h in self.monitor_heads]
        per_head_prompt_series: Dict[str, List[float]] = {}
        q_len = 0
        for (l_idx, h_idx), hk in zip(self.monitor_heads, head_keys):
            if l_idx not in attn_maps:
                continue
            mat = attn_maps[l_idx]  # (1, H_q, Q_len, K_len)
            q_len = int(mat.shape[2])
            k_len = int(mat.shape[3])
            p_end = min(max(1, think_start_idx), k_len)
            sub = mat[0, h_idx, :, :p_end].astype(mx.float32)  # (Q_len, p_end)
            mass_vec = mx.sum(sub, axis=-1)
            mx.eval(mass_vec)
            per_head_prompt_series[hk] = [float(v) for v in mass_vec.tolist()]

        records: List[Dict[str, Any]] = []
        for idx, tid in enumerate(all_token_ids):
            tok_text = _decode_single_token(self.tokenizer, int(tid))
            is_think = idx >= think_start_idx
            q_rel = idx - query_offset
            if 0 <= q_rel < q_len and per_head_prompt_series:
                head_scores = {
                    hk: round(per_head_prompt_series[hk][q_rel], 6)
                    for hk in head_keys
                    if hk in per_head_prompt_series
                }
                mean_score = float(
                    sum(head_scores.values()) / max(len(head_scores), 1)
                )
                local_mass = max(0.0, 1.0 - mean_score) if is_think else 0.0
            else:
                head_scores = {hk: 0.0 for hk in head_keys}
                mean_score = 0.0
                local_mass = 0.0

            r_rel = idx - rhythm_span_start
            if is_think and 0 <= r_rel < len(preplan_list):
                preplan_score = float(preplan_list[r_rel])
                waad_raw = float(waad_raw_list[r_rel])
                anchor_score = float(anchor_list[r_rel])
                fai_raw = float(fai_raw_list[r_rel])
                is_coupled = idx in coupled_indices
                j_role = juncture_roles.get(idx, None)
            else:
                preplan_score = 0.0
                waad_raw = 0.0
                anchor_score = 0.0
                fai_raw = 0.0
                is_coupled = False
                j_role = None

            is_think_end = (
                (think_end_token_idx is not None and idx == think_end_token_idx)
                or ("</think>" in tok_text)
            )
            rounded_score = round(mean_score, 6)
            records.append(
                {
                    "token_idx": idx,
                    "token_id": int(tid),
                    "text": tok_text,
                    "is_think": is_think,
                    "is_think_end": bool(is_think_end),
                    "head_scores": head_scores,
                    "lookback_score": rounded_score,
                    "mean_lookback_score": rounded_score,
                    "local_think_mass": round(local_mass, 6),
                    "preplan_score": round(preplan_score, 4),
                    "waad_raw": round(waad_raw, 4),
                    "anchor_score": round(anchor_score, 4),
                    "fai_raw": round(fai_raw, 6),
                    "is_coupled_juncture": bool(is_coupled),
                    "juncture_role": j_role,
                }
            )
        return records

    def _check_think_closed(self, generated_ids: Sequence[int], decoded_so_far: str) -> bool:
        if "</think>" in decoded_so_far:
            return True
        if self.think_close_token_ids:
            n_c = len(self.think_close_token_ids)
            if len(generated_ids) >= n_c and list(generated_ids[-n_c:]) == self.think_close_token_ids:
                return True
        return False

    def _prefill_segment_with_stride_snapshots_and_probe(
        self,
        seg_token_ids: Sequence[int],
        stride: int,
        alpha: float,
        active_layers: Sequence[int],
        token_slice: Optional[Tuple[Optional[int], Optional[int]]] = None,
        capture_probe: bool = True,
    ) -> Tuple[mx.array, Dict[int, mx.array]]:
        """
        Prefill `seg_token_ids` starting from `self.timeline.current_offset`, pausing at
        every stride-8 boundary `(abs_pos - start_offset) % stride == 0` (and at the
        segment end) to take a materialized Gated DeltaNet `ArraysCache` snapshot while
        stitching the unfused attention maps across the sub-chunks.
        """
        seg_list = [int(t) for t in seg_token_ids]
        if not seg_list:
            raise ValueError("`seg_token_ids` must be non-empty.")

        start_offset = self.timeline.current_offset
        end_offset = start_offset + len(seg_list)

        # Build relative cut points aligned to `start_offset + k * stride`
        cut_points: List[int] = []
        for abs_pos in range(start_offset + 1, end_offset + 1):
            if (abs_pos - start_offset) % stride == 0 or abs_pos == end_offset:
                cut_points.append(abs_pos - start_offset)

        chunk_attns: List[Dict[int, mx.array]] = []
        cursor = 0
        last_logits: Optional[mx.array] = None

        for end_rel in cut_points:
            sub_ids = seg_list[cursor:end_rel]
            if not sub_ids:
                continue

            # Map overall segment `token_slice` into this sub-chunk `[cursor : end_rel]`
            sub_slice: Optional[Tuple[Optional[int], Optional[int]]] = None
            sub_active = float(alpha) != 0.0
            if token_slice is not None and sub_active:
                s_start, s_end = token_slice
                g_s = 0 if s_start is None else int(s_start)
                g_e = len(seg_list) if s_end is None else int(s_end)
                ov_s = max(cursor, g_s) - cursor
                ov_e = min(end_rel, g_e) - cursor
                if ov_s >= ov_e:
                    sub_active = False
                else:
                    sub_slice = (ov_s, ov_e)

            with self.projector.pulse(
                alpha=float(alpha) if sub_active else 0.0,
                active_layers=active_layers,
                enabled=sub_active,
                token_slice=sub_slice,
            ):
                sub_arr = mx.array(sub_ids, dtype=mx.int32)[None, :]
                if capture_probe:
                    with capture_attention_maps(self.model) as sub_probe:
                        last_logits = self.model(sub_arr, cache=self.timeline.cache)
                        mx.eval(last_logits)
                        self.timeline.eval_all()
                    chunk_attns.append(sub_probe.attention_maps)
                else:
                    last_logits = self.model(sub_arr, cache=self.timeline.cache)
                    mx.eval(last_logits)
                    self.timeline.eval_all()

            abs_pos = start_offset + end_rel
            if (abs_pos - self.think_start_idx) % stride == 0 or abs_pos == end_offset:
                self.timeline.snapshot(abs_pos)
            cursor = end_rel

        self.timeline.token_history = self.timeline.token_history[:start_offset] + seg_list
        stitched = self._stitch_attention_sequence(chunk_attns) if capture_probe else {}
        assert last_logits is not None
        return last_logits, stitched

    def prefill_and_probe(
        self,
        prompt: Optional[str] = None,
        clause_a_text: Optional[str] = None,
        initial_think_text: Optional[str] = None,
        prompt_token_ids: Optional[Sequence[int]] = None,
        think_token_ids: Optional[Sequence[int]] = None,
        generate_organic_tokens: Optional[int] = None,
        max_think_tokens: int = 0,
        alpha: float = 0.0,
        organic_alpha: Optional[float] = None,
        active_layers: Optional[Union[Sequence[int], str]] = None,
        snapshot_stride: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Prefill user prompt + initial `<think>` trace OR organically generate the model's
        natural `<think>` trace (`generate_organic_tokens`, default 48 when `clause_a_text` is empty),
        automatically snapshotting the 48 Gated DeltaNet `ArraysCache` states at `think_start_idx`
        and every `snapshot_stride=8` tokens inside `<think>`, and capturing unfused attention
        maps across all 16 full-attention layers (`384` heads) for both 5-Head Prompt-Lookback
        and Preplan (`WAAD_t`) + Anchor (`FAI_s`) rhythm telemetry.
        """
        with self.lock:
            stride = max(1, int(snapshot_stride or self.snapshot_stride))
            self.snapshot_stride = stride
            layers_list = _parse_active_layers(active_layers)

            if prompt_token_ids is not None:
                p_ids = [int(t) for t in prompt_token_ids]
            else:
                raw_prompt = prompt or (
                    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                    "<|im_start|>user\nExplain the mathematical relationship between Gated DeltaNet fast-weight "
                    "linear attention states and exact orthogonal subspace projections in deep transformers.<|im_end|>\n"
                    "<|im_start|>assistant\n<think>\n"
                )
                if "<think>" not in raw_prompt:
                    raw_prompt = (
                        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                        f"<|im_start|>user\n{raw_prompt}<|im_end|>\n"
                        f"<|im_start|>assistant\n<think>\n"
                    )
                p_ids = [int(t) for t in self.tokenizer.encode(raw_prompt)]

            # Resolve explicit Clause A text vs Organic Generation mode
            effective_clause_a = (
                clause_a_text if clause_a_text is not None else initial_think_text
            )
            if think_token_ids is not None:
                t_ids = [int(t) for t in think_token_ids]
            elif effective_clause_a is not None and len(effective_clause_a.strip()) > 0:
                t_ids = [int(t) for t in self.tokenizer.encode(effective_clause_a)]
            else:
                t_ids = []

            eff_alpha = float(organic_alpha) if organic_alpha is not None else float(alpha)
            if not t_ids:
                if generate_organic_tokens is not None and generate_organic_tokens > 0:
                    num_organic = int(generate_organic_tokens)
                elif max_think_tokens > 0:
                    num_organic = int(max_think_tokens)
                else:
                    num_organic = 48
            else:
                num_organic = 0

            think_start_idx = len(p_ids)
            self.think_start_idx = think_start_idx

            # Reset fresh HybridTimelineCache
            self.timeline = HybridTimelineCache(
                model=self.model, prefill_step_size=self.prefill_step_size
            )

            t0 = time.perf_counter()
            # 1. Prefill prompt prefix [0..think_start_idx-1] at alpha=0.0 and snapshot at think_start_idx
            self.projector.configure(alpha=0.0, enabled=False, clear_token_slice=True)
            prefix_logits = self.timeline.prefill_with_snapshots(
                p_ids, snapshot_indices=[think_start_idx]
            )
            mx.eval(prefix_logits)

            # 2. Organic <think> Generation Mode: autoregressively generate `num_organic` tokens,
            #    then rewind to `think_start_idx` and run a single fast unfused probe pass with
            #    stride-8 snapshots!
            organic_generated = False
            if not t_ids and num_organic > 0:
                organic_generated = True
                self.projector.configure(
                    alpha=eff_alpha,
                    active_layers=layers_list,
                    enabled=(eff_alpha != 0.0),
                    clear_token_slice=True,
                )
                cur_logits = prefix_logits
                gen_ids: List[int] = []
                for step_i in range(num_organic):
                    next_tok = mx.argmax(cur_logits[:, -1, :], axis=-1, keepdims=True)
                    mx.eval(next_tok)
                    tid = int(next_tok.item())
                    gen_ids.append(tid)
                    if step_i < num_organic - 1:
                        cur_logits = self.model(next_tok, cache=self.timeline.cache)
                        mx.eval(cur_logits)
                        abs_pos = think_start_idx + step_i + 1
                        if (abs_pos - think_start_idx) % stride == 0:
                            self.timeline.eval_all()
                t_ids = gen_ids
                # Rewind to `think_start_idx` so step 3 runs the single fast unfused probe pass
                # and populates all stride-8 snapshots across the organically generated trace
                self.timeline.rewind_to(think_start_idx)
                self.timeline.restore(think_start_idx)
                self.timeline.token_history = list(p_ids)

            if not t_ids:
                next_tok = mx.argmax(prefix_logits[:, -1, :], axis=-1)
                mx.eval(next_tok)
                t_ids = [int(next_tok.item())]

            # 3. Run unfused probe pass over `t_ids` with dense stride-8 snapshots
            _, stitched_attn = self._prefill_segment_with_stride_snapshots_and_probe(
                seg_token_ids=t_ids,
                stride=stride,
                alpha=eff_alpha,
                active_layers=layers_list,
                token_slice=None,
                capture_probe=True,
            )
            self.projector.configure(alpha=0.0, enabled=False, clear_token_slice=True)
            prefill_ms = (time.perf_counter() - t0) * 1000.0

            all_ids = list(p_ids) + list(t_ids)
            self.timeline.token_history = list(all_ids)
            self.baseline_token_ids = list(all_ids)
            self.active_token_ids = list(all_ids)
            self.baseline_think_attn = stitched_attn
            self.active_think_attn = dict(stitched_attn)

            rhythm_info = compute_preplan_and_anchor_rhythm(
                attn_maps=stitched_attn,
                think_start_idx=think_start_idx,
                query_offset=think_start_idx,
            )
            self.last_rhythm_info = rhythm_info

            records = self._compute_per_token_lookback_records(
                all_token_ids=all_ids,
                think_start_idx=think_start_idx,
                attn_maps=stitched_attn,
                query_offset=think_start_idx,
                rhythm_info=rhythm_info,
            )
            self.baseline_token_records = list(records)
            self.active_token_records = list(records)

            think_records = [r for r in records if r["is_think"]]
            if think_records:
                peak_tok = max(think_records, key=lambda r: r["mean_lookback_score"])
            else:
                peak_tok = records[-1]

            try:
                decoded_clause_a = str(self.tokenizer.decode(t_ids))
            except Exception:
                decoded_clause_a = "".join(_decode_single_token(self.tokenizer, tid) for tid in t_ids)

            tail_summary = self._summarize_tail_records(think_records, tail_window=16)

            return {
                "model_id": self.model_id,
                "organic_generated": organic_generated,
                "clause_a_text": decoded_clause_a,
                "think_start_idx": think_start_idx,
                "total_tokens": len(all_ids),
                "think_tokens": len(t_ids),
                "snapshot_stride": stride,
                "snapshots": self.timeline.list_snapshots(),
                "prefill_ms": round(prefill_ms, 2),
                "alpha": eff_alpha,
                "active_layers": [layers_list[0], layers_list[-1]] if layers_list else [],
                "peak_spike_token": peak_tok,
                "peak_lookback_token_idx": int(peak_tok["token_idx"]),
                "peak_preplan_token_idx": int(rhythm_info["peak_preplan_token_idx"]),
                "peak_anchor_token_idx": int(rhythm_info["peak_anchor_token_idx"]),
                "coupled_junctures": rhythm_info["coupled_junctures"],
                "rhythm_summary": {
                    "num_total_heads": rhythm_info["num_total_heads"],
                    "num_local_heads": rhythm_info["num_local_heads"],
                    "num_global_heads": rhythm_info["num_global_heads"],
                    "median_head_distance": rhythm_info["median_head_distance"],
                    "peak_preplan_token_idx": rhythm_info["peak_preplan_token_idx"],
                    "peak_anchor_token_idx": rhythm_info["peak_anchor_token_idx"],
                    "coupled_junctures": rhythm_info["coupled_junctures"],
                },
                "tail_16_summary": tail_summary,
                "spi_traces_count": self.spi_logger.count(),
                "tokens": records,
            }

    def _summarize_tail_records(
        self,
        records_subset: Sequence[Dict[str, Any]],
        tail_window: int = 16,
    ) -> Dict[str, Any]:
        tail_w = min(tail_window, len(records_subset))
        tail_recs = records_subset[-tail_w:] if tail_w > 0 else []
        head_keys = [f"L{l}.H{h}" for l, h in self.monitor_heads]
        per_head_summary: Dict[str, Dict[str, float]] = {}
        for hk in head_keys:
            p_m = (
                sum(r["head_scores"].get(hk, 0.0) for r in tail_recs) / len(tail_recs)
                if tail_recs
                else 0.0
            )
            per_head_summary[hk] = {
                "prompt_mass": round(p_m, 4),
                "local_think_mass": round(max(0.0, 1.0 - p_m), 4),
            }
        mean_prompt_tail = (
            sum(r["mean_lookback_score"] for r in tail_recs) / len(tail_recs)
            if tail_recs
            else 0.0
        )
        mean_preplan_tail = (
            sum(r.get("preplan_score", 0.0) for r in tail_recs) / len(tail_recs)
            if tail_recs
            else 0.0
        )
        mean_anchor_tail = (
            sum(r.get("anchor_score", 0.0) for r in tail_recs) / len(tail_recs)
            if tail_recs
            else 0.0
        )
        return {
            "mean_prompt_mass": round(mean_prompt_tail, 4),
            "mean_local_think_mass": round(max(0.0, 1.0 - mean_prompt_tail), 4),
            "mean_preplan_score": round(mean_preplan_tail, 4),
            "mean_anchor_score": round(mean_anchor_tail, 4),
            "per_head": per_head_summary,
        }

    @staticmethod
    def _stitch_attention_sequence(
        chunk_attns: Sequence[Dict[int, mx.array]],
    ) -> Dict[int, mx.array]:
        if not chunk_attns:
            return {}
        if len(chunk_attns) == 1:
            return dict(chunk_attns[0])
        sample_layer = next(iter(chunk_attns[-1].keys()))
        final_k = int(chunk_attns[-1][sample_layer].shape[-1])
        stitched: Dict[int, mx.array] = {}
        for l_idx in chunk_attns[0].keys():
            padded_chunks: List[mx.array] = []
            for ch in chunk_attns:
                if l_idx not in ch:
                    continue
                a = ch[l_idx]
                B, H, q_len, k_len = a.shape
                pad_k = final_k - k_len
                if pad_k > 0:
                    z = mx.zeros((B, H, q_len, pad_k), dtype=a.dtype)
                    a = mx.concatenate([a, z], axis=-1)
                padded_chunks.append(a)
            comb = mx.concatenate(padded_chunks, axis=2)
            mx.eval(comb)
            stitched[l_idx] = comb
        return stitched

    def fork_and_edit_stateful(
        self,
        fork_token_idx: int,
        edited_text: Optional[str] = None,
        edited_token_ids: Optional[Sequence[int]] = None,
        alpha: float = 1.0,
        active_layers: Optional[Union[Sequence[int], str]] = None,
        think_gated: bool = True,
        max_new_tokens: int = 24,
        capture_probe: bool = True,
        label: str = "candidate",
    ) -> Dict[str, Any]:
        """
        Rewind the 16 `KVCache` views and restore the nearest `ArraysCache` snapshot
        `<= fork_token_idx` (supporting both baseline token indices AND any token index
        inside a previously forked / decoded continuation stream!), warm-prefill `edited_text`
        under `alpha` / `think_gated`, decode `max_new_tokens` continuation tokens with
        stride-8 snapshots and unfused 5-head lookback + 384-head rhythm probing, and
        automatically log a counterfactual record to `spi_traces.jsonl`.
        """
        with self.lock:
            reference_history = (
                list(self.active_token_ids)
                if len(self.active_token_ids) >= len(self.baseline_token_ids)
                else list(self.baseline_token_ids)
            )
            reference_records = (
                list(self.active_token_records)
                if len(self.active_token_records) >= len(self.baseline_token_records)
                else list(self.baseline_token_records)
            )
            if not reference_history:
                raise ValueError(
                    "No timeline initialized. Call `/api/prefill_and_probe` first."
                )
            fork_idx = int(fork_token_idx)
            if fork_idx < 0 or fork_idx > len(reference_history):
                raise ValueError(
                    f"fork_token_idx={fork_idx} out of range [0, {len(reference_history)}]."
                )

            stride = self.snapshot_stride
            layers_list = _parse_active_layers(active_layers)
            if edited_token_ids is not None:
                new_ids = [int(t) for t in edited_token_ids]
            elif edited_text is not None and len(edited_text) > 0:
                new_ids = [int(t) for t in self.tokenizer.encode(edited_text)]
            else:
                raise ValueError("Either `edited_text` or `edited_token_ids` must be non-empty.")

            t0 = time.perf_counter()
            # 1. Restore to nearest snapshot <= fork_idx and replay delta (0..7 tokens) at alpha=0.0
            self.projector.configure(alpha=0.0, enabled=False, clear_token_slice=True)
            snap_used, delta_replayed = self.timeline.restore_to_arbitrary_index(
                target_token_idx=fork_idx,
                token_history=reference_history,
                model=self.model,
                cache_snapshot_at_target=True,
            )

            # 2. Warm-prefill `new_ids` with stride-8 snapshots & unfused attention capture
            token_slice: Optional[Tuple[Optional[int], Optional[int]]] = None
            think_closed_in_edit = False
            think_end_abs_idx: Optional[int] = None
            if think_gated and edited_text and "</think>" in edited_text:
                before_close = edited_text.split("</think>", 1)[0]
                before_ids = self.tokenizer.encode(before_close)
                cut_len = max(1, len(before_ids))
                token_slice = (0, cut_len)
                think_closed_in_edit = True
                think_end_abs_idx = fork_idx + min(len(new_ids) - 1, cut_len)

            edit_logits, fork_edit_attn = self._prefill_segment_with_stride_snapshots_and_probe(
                seg_token_ids=new_ids,
                stride=stride,
                alpha=float(alpha),
                active_layers=layers_list,
                token_slice=token_slice,
                capture_probe=capture_probe,
            )
            fork_edit_ms = (time.perf_counter() - t0) * 1000.0

            cont_start_idx = fork_idx + len(new_ids)
            self.timeline.snapshot(cont_start_idx)
            prefix_and_edit_ids = list(reference_history[:fork_idx]) + list(new_ids)
            self.timeline.token_history = list(prefix_and_edit_ids)

            # 3. Decode `max_new_tokens` continuation tokens (respecting `think_gated` auto-release)
            #    and then run a single fast batched probe + stride-8 snapshot pass over `continuation_ids`
            t_dec0 = time.perf_counter()
            continuation_ids: List[int] = []
            think_released = think_closed_in_edit or (float(alpha) == 0.0)
            think_released_at_step: Optional[int] = 0 if think_closed_in_edit else None

            if max_new_tokens > 0:
                active_decode_alpha = 0.0 if (think_gated and think_closed_in_edit) else float(alpha)
                self.projector.configure(
                    alpha=active_decode_alpha,
                    active_layers=layers_list,
                    enabled=(active_decode_alpha != 0.0),
                    clear_token_slice=True,
                )
                cur_logits = edit_logits
                decoded_so_far = ""
                for step_i in range(max_new_tokens):
                    next_tok = mx.argmax(cur_logits[:, -1, :], axis=-1, keepdims=True)
                    mx.eval(next_tok)
                    tid = int(next_tok.item())
                    continuation_ids.append(tid)
                    decoded_so_far += _decode_single_token(self.tokenizer, tid)

                    if think_gated and not think_released:
                        if self._check_think_closed(continuation_ids, decoded_so_far):
                            think_released = True
                            think_released_at_step = step_i + 1
                            think_end_abs_idx = cont_start_idx + step_i
                            self.projector.configure(alpha=0.0, enabled=False)

                    if step_i < max_new_tokens - 1:
                        cur_logits = self.model(next_tok, cache=self.timeline.cache)
                        mx.eval(cur_logits)

                self.projector.configure(alpha=0.0, enabled=False, clear_token_slice=True)

                # Rewind to `cont_start_idx` and run a single fast batched prefill over `continuation_ids`
                # with stride-8 snapshots and `capture_attention_maps`
                self.timeline.rewind_to(cont_start_idx)
                self.timeline.restore(cont_start_idx)
                self.timeline.token_history = list(prefix_and_edit_ids)

                cont_slice: Optional[Tuple[Optional[int], Optional[int]]] = None
                cont_alpha = active_decode_alpha
                if think_gated and think_released_at_step is not None and think_released_at_step > 0:
                    cont_slice = (0, think_released_at_step)

                _, cont_attn = self._prefill_segment_with_stride_snapshots_and_probe(
                    seg_token_ids=continuation_ids,
                    stride=stride,
                    alpha=cont_alpha,
                    active_layers=layers_list,
                    token_slice=cont_slice,
                    capture_probe=capture_probe,
                )
            else:
                cont_attn = {}

            self.projector.configure(alpha=0.0, enabled=False, clear_token_slice=True)
            decode_ms = (time.perf_counter() - t_dec0) * 1000.0

            full_forked_ids = prefix_and_edit_ids + continuation_ids
            self.timeline.token_history = list(full_forked_ids)
            self.active_token_ids = list(full_forked_ids)

            # Stitch `fork_edit_attn` and `cont_attn` (and any prior <think> prefix rows)
            if capture_probe:
                if fork_edit_attn and cont_attn:
                    combined_fork_attn = self._stitch_attention_sequence([fork_edit_attn, cont_attn])
                elif fork_edit_attn:
                    combined_fork_attn = fork_edit_attn
                else:
                    combined_fork_attn = cont_attn

                prefix_think_q = max(0, fork_idx - self.think_start_idx)
                prior_attn_source = self.active_think_attn or self.baseline_think_attn
                if prefix_think_q > 0 and prior_attn_source and combined_fork_attn:
                    sample_l = next(iter(prior_attn_source.keys()))
                    if int(prior_attn_source[sample_l].shape[2]) >= prefix_think_q:
                        prefix_think_attn = {
                            l_idx: mat[:, :, :prefix_think_q, :fork_idx]
                            for l_idx, mat in prior_attn_source.items()
                        }
                        full_think_attn = self._stitch_attention_sequence(
                            [prefix_think_attn, combined_fork_attn]
                        )
                        attn_for_records = full_think_attn
                        query_offset_for_records = self.think_start_idx
                        self.active_think_attn = full_think_attn
                    else:
                        attn_for_records = combined_fork_attn
                        query_offset_for_records = fork_idx
                else:
                    attn_for_records = combined_fork_attn
                    query_offset_for_records = fork_idx
                    if fork_idx == self.think_start_idx and combined_fork_attn:
                        self.active_think_attn = combined_fork_attn
            else:
                combined_fork_attn = {}
                attn_for_records = {}
                query_offset_for_records = fork_idx

            rhythm_info = compute_preplan_and_anchor_rhythm(
                attn_maps=attn_for_records,
                think_start_idx=self.think_start_idx,
                query_offset=query_offset_for_records,
            )
            self.last_rhythm_info = rhythm_info

            try:
                decoded_edited_text = (
                    edited_text
                    if edited_text is not None
                    else str(self.tokenizer.decode(new_ids))
                )
            except Exception:
                decoded_edited_text = "".join(
                    _decode_single_token(self.tokenizer, tid) for tid in new_ids
                )

            try:
                continuation_text = str(self.tokenizer.decode(continuation_ids))
            except Exception:
                continuation_text = "".join(
                    _decode_single_token(self.tokenizer, tid) for tid in continuation_ids
                )

            all_think_ids = full_forked_ids[self.think_start_idx :]
            try:
                full_think_text = str(self.tokenizer.decode(all_think_ids))
            except Exception:
                full_think_text = "".join(
                    _decode_single_token(self.tokenizer, tid) for tid in all_think_ids
                )

            # 4. Compute per-token 5-Head Prompt-Lookback + Preplan (WAAD_t) + Anchor (FAI_s) records
            all_fork_records = self._compute_per_token_lookback_records(
                all_token_ids=full_forked_ids,
                think_start_idx=self.think_start_idx,
                attn_maps=attn_for_records,
                query_offset=query_offset_for_records,
                think_end_token_idx=think_end_abs_idx,
                rhythm_info=rhythm_info,
            )
            if query_offset_for_records > 0 and len(reference_records) >= query_offset_for_records:
                merged_records = (
                    list(reference_records[:query_offset_for_records])
                    + all_fork_records[query_offset_for_records:]
                )
            else:
                merged_records = all_fork_records
            self.active_token_records = merged_records

            edited_records = merged_records[fork_idx:cont_start_idx]
            continuation_records = merged_records[cont_start_idx:]
            repaired_records = edited_records + continuation_records

            tail_summary = self._summarize_tail_records(edited_records, tail_window=16)
            cont_tail_summary = (
                self._summarize_tail_records(continuation_records, tail_window=16)
                if continuation_records
                else tail_summary
            )

            # Compute JSD shift vs baseline overlapping slice if available
            jsd_vs_baseline = 0.0
            if capture_probe and self.baseline_think_attn and fork_edit_attn:
                rel_start = max(0, fork_idx - self.think_start_idx)
                overlap_base: Dict[int, mx.array] = {}
                overlap_fork: Dict[int, mx.array] = {}
                for l_idx, f_mat in fork_edit_attn.items():
                    if l_idx not in self.baseline_think_attn:
                        continue
                    b_mat = self.baseline_think_attn[l_idx]
                    b_q_avail = int(b_mat.shape[2]) - rel_start
                    q_common = min(b_q_avail, int(f_mat.shape[2]))
                    k_common = min(int(b_mat.shape[3]), int(f_mat.shape[3]))
                    if q_common > 0 and k_common > 0:
                        overlap_base[l_idx] = b_mat[:, :, rel_start : rel_start + q_common, :k_common]
                        overlap_fork[l_idx] = f_mat[:, :, :q_common, :k_common]
                if overlap_base:
                    shift_info = compute_attention_shift(overlap_base, overlap_fork)
                    jsd_vs_baseline = float(shift_info["mean_jsd"])

            think_recs_merged = [r for r in merged_records if r["is_think"]]
            peak_lookback_idx = (
                int(max(think_recs_merged, key=lambda r: r["lookback_score"])["token_idx"])
                if think_recs_merged
                else fork_idx
            )

            # 5. Automatically construct and append Counterfactual Trajectory Record (`spi_traces.jsonl`)
            prompt_ids = reference_history[: self.think_start_idx]
            shared_prefix_ids = reference_history[:fork_idx]
            rejected_ids = reference_history[fork_idx:]
            rejected_recs = reference_records[fork_idx:] if len(reference_records) > fork_idx else []

            try:
                prompt_text_dec = str(self.tokenizer.decode(prompt_ids))
            except Exception:
                prompt_text_dec = "".join(_decode_single_token(self.tokenizer, tid) for tid in prompt_ids)
            try:
                shared_prefix_text = str(self.tokenizer.decode(shared_prefix_ids))
            except Exception:
                shared_prefix_text = "".join(_decode_single_token(self.tokenizer, tid) for tid in shared_prefix_ids)
            try:
                rejected_text = str(self.tokenizer.decode(rejected_ids))
            except Exception:
                rejected_text = "".join(_decode_single_token(self.tokenizer, tid) for tid in rejected_ids)

            fork_rec_before = (
                reference_records[fork_idx]
                if fork_idx < len(reference_records)
                else (reference_records[-1] if reference_records else {})
            )
            rejected_head_summary = self._summarize_tail_records(rejected_recs, tail_window=16)

            rej_lookback_mean = float(rejected_head_summary.get("mean_prompt_mass", 0.0))
            rej_preplan_mean = float(rejected_head_summary.get("mean_preplan_score", 0.0))
            rej_anchor_mean = float(rejected_head_summary.get("mean_anchor_score", 0.0))

            rep_lookback_mean = (
                sum(r["lookback_score"] for r in repaired_records) / len(repaired_records)
                if repaired_records
                else 0.0
            )
            rep_preplan_mean = (
                sum(r["preplan_score"] for r in repaired_records) / len(repaired_records)
                if repaired_records
                else 0.0
            )
            rep_anchor_mean = (
                sum(r["anchor_score"] for r in repaired_records) / len(repaired_records)
                if repaired_records
                else 0.0
            )

            spi_record = {
                "trace_id": str(uuid.uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model_path": self.model_id,
                "prompt_text": prompt_text_dec,
                "think_start_idx": self.think_start_idx,
                "fork_token_idx": fork_idx,
                "shared_prefix_token_ids": shared_prefix_ids,
                "shared_prefix_text": shared_prefix_text,
                "rejected_branch": {
                    "token_ids": rejected_ids,
                    "text": rejected_text,
                    "telemetry_at_fork": {
                        "lookback": float(fork_rec_before.get("lookback_score", 0.0)),
                        "preplan": float(fork_rec_before.get("preplan_score", 0.0)),
                        "anchor": float(fork_rec_before.get("anchor_score", 0.0)),
                    },
                    "head_mass_summary": rejected_head_summary,
                },
                "repaired_branch": {
                    "edited_text": decoded_edited_text,
                    "continuation_text": continuation_text,
                    "token_ids": new_ids + continuation_ids,
                    "alpha": float(alpha),
                    "think_gated": bool(think_gated),
                    "think_end_token_idx": think_end_abs_idx,
                    "telemetry_summary": {
                        "mean_lookback": round(rep_lookback_mean, 4),
                        "mean_preplan": round(rep_preplan_mean, 4),
                        "mean_anchor": round(rep_anchor_mean, 4),
                        "peak_preplan_token_idx": int(rhythm_info["peak_preplan_token_idx"]),
                        "peak_anchor_token_idx": int(rhythm_info["peak_anchor_token_idx"]),
                    },
                    "head_mass_summary": tail_summary,
                },
                "causal_deltas": {
                    "attention_jsd_shift": round(jsd_vs_baseline, 6),
                    "lookback_mass_delta": round(rep_lookback_mean - rej_lookback_mean, 6),
                    "preplan_mean_delta": round(rep_preplan_mean - rej_preplan_mean, 6),
                    "anchor_mean_delta": round(rep_anchor_mean - rej_anchor_mean, 6),
                    "fork_edit_ms": round(fork_edit_ms, 2),
                },
                "label": label if label in ("accepted", "candidate", "rejected") else "candidate",
            }
            self.spi_logger.append_trace(spi_record)
            spi_count = self.spi_logger.count()

            return {
                "spi_trace_id": spi_record["trace_id"],
                "spi_traces_count": spi_count,
                "fork_token_idx": fork_idx,
                "snapshot_restored_idx": snap_used,
                "delta_tokens_replayed": delta_replayed,
                "edited_tokens_count": len(new_ids),
                "continuation_tokens_count": len(continuation_ids),
                "total_tokens": len(full_forked_ids),
                "snapshots": self.timeline.list_snapshots(),
                "fork_edit_ms": round(fork_edit_ms, 2),
                "decode_ms": round(decode_ms, 2),
                "alpha": float(alpha),
                "think_gated": bool(think_gated),
                "think_released_at_step": think_released_at_step,
                "think_end_token_idx": think_end_abs_idx,
                "jsd_vs_baseline": round(jsd_vs_baseline, 6),
                "peak_lookback_token_idx": peak_lookback_idx,
                "peak_preplan_token_idx": int(rhythm_info["peak_preplan_token_idx"]),
                "peak_anchor_token_idx": int(rhythm_info["peak_anchor_token_idx"]),
                "coupled_junctures": rhythm_info["coupled_junctures"],
                "rhythm_summary": {
                    "num_total_heads": rhythm_info["num_total_heads"],
                    "num_local_heads": rhythm_info["num_local_heads"],
                    "num_global_heads": rhythm_info["num_global_heads"],
                    "median_head_distance": rhythm_info["median_head_distance"],
                    "peak_preplan_token_idx": rhythm_info["peak_preplan_token_idx"],
                    "peak_anchor_token_idx": rhythm_info["peak_anchor_token_idx"],
                    "coupled_junctures": rhythm_info["coupled_junctures"],
                },
                "tail_16_summary": tail_summary,
                "continuation_tail_16_summary": cont_tail_summary,
                "edited_tokens": edited_records,
                "forked_tokens": edited_records,
                "continuation_tokens": continuation_records,
                "continuation_token_ids": continuation_ids,
                "continuation_text": continuation_text,
                "full_think_text": full_think_text,
            }

    def promote_active_to_baseline(self) -> Dict[str, Any]:
        """
        Promote the currently active forked + continuation stream to the Left Baseline state
        so the user can perform infinite multi-step tree surgery down a reasoning chain.
        """
        with self.lock:
            if not self.active_token_ids:
                raise ValueError("No active stream to promote.")
            self.baseline_token_ids = list(self.active_token_ids)
            self.baseline_token_records = list(self.active_token_records)
            if self.active_think_attn:
                self.baseline_think_attn = dict(self.active_think_attn)
            think_records = [r for r in self.baseline_token_records if r["is_think"]]
            peak_tok = (
                max(think_records, key=lambda r: r["mean_lookback_score"])
                if think_records
                else self.baseline_token_records[-1]
            )
            peak_pre_tok = (
                max(think_records, key=lambda r: r.get("preplan_score", 0.0))
                if think_records
                else peak_tok
            )
            peak_anc_tok = (
                max(think_records, key=lambda r: r.get("anchor_score", 0.0))
                if think_records
                else peak_tok
            )
            think_ids = self.baseline_token_ids[self.think_start_idx :]
            try:
                clause_a_text = str(self.tokenizer.decode(think_ids))
            except Exception:
                clause_a_text = "".join(
                    _decode_single_token(self.tokenizer, tid) for tid in think_ids
                )
            return {
                "promoted": True,
                "model_id": self.model_id,
                "think_start_idx": self.think_start_idx,
                "total_tokens": len(self.baseline_token_ids),
                "think_tokens": len(think_ids),
                "snapshots": self.timeline.list_snapshots(),
                "clause_a_text": clause_a_text,
                "peak_spike_token": peak_tok,
                "peak_lookback_token_idx": int(peak_tok["token_idx"]),
                "peak_preplan_token_idx": int(peak_pre_tok["token_idx"]),
                "peak_anchor_token_idx": int(peak_anc_tok["token_idx"]),
                "coupled_junctures": self.last_rhythm_info.get("coupled_junctures", []),
                "tail_16_summary": self._summarize_tail_records(think_records, tail_window=16),
                "spi_traces_count": self.spi_logger.count(),
                "tokens": self.baseline_token_records,
            }

    def generate_chat_completion(
        self,
        messages: Sequence[Dict[str, str]],
        max_tokens: int = 64,
        alpha: float = 0.0,
        active_layers: Optional[Union[Sequence[int], str]] = None,
        think_gated: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute an OpenAI-compatible chat completion with optional `alpha`, `active_layers`,
        and `think_gated` steering (holding `alpha` inside `<think>...</think>` and automatically
        switching to `alpha = 0.0` as soon as `</think>` is emitted).
        """
        with self.lock:
            layers_list = _parse_active_layers(active_layers)
            prompt_text = self._format_chat_messages(messages)
            prompt_ids = [int(t) for t in self.tokenizer.encode(prompt_text)]
            if not prompt_ids:
                prompt_ids = [1]

            chat_timeline = HybridTimelineCache(
                model=self.model, prefill_step_size=self.prefill_step_size
            )
            self.projector.configure(alpha=0.0, enabled=False, clear_token_slice=True)
            logits = chat_timeline.prefill_with_snapshots(
                prompt_ids, snapshot_indices=[len(prompt_ids)]
            )
            mx.eval(logits)

            current_alpha = float(alpha)
            self.projector.configure(
                alpha=current_alpha,
                active_layers=layers_list,
                enabled=(current_alpha != 0.0),
                clear_token_slice=True,
            )

            generated_ids: List[int] = []
            alpha_by_step: List[float] = []
            released_at_step: Optional[int] = None
            decoded_so_far = ""
            cur_logits = logits

            for step_i in range(max(1, int(max_tokens))):
                alpha_by_step.append(self.projector.alpha if self.projector.enabled else 0.0)
                next_tok = mx.argmax(cur_logits[:, -1, :], axis=-1, keepdims=True)
                mx.eval(next_tok)
                tid = int(next_tok.item())
                generated_ids.append(tid)
                decoded_so_far += _decode_single_token(self.tokenizer, tid)

                if think_gated and released_at_step is None:
                    if self._check_think_closed(generated_ids, decoded_so_far):
                        released_at_step = step_i + 1
                        self.projector.configure(alpha=0.0, enabled=False)

                if step_i < max_tokens - 1:
                    cur_logits = self.model(next_tok, cache=chat_timeline.cache)
                    mx.eval(cur_logits)

            self.projector.configure(alpha=0.0, enabled=False, clear_token_slice=True)
            try:
                content = str(self.tokenizer.decode(generated_ids))
            except Exception:
                content = decoded_so_far

            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content,
                        },
                        "finish_reason": "length",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": len(generated_ids),
                    "total_tokens": len(prompt_ids) + len(generated_ids),
                },
                "sandbox_telemetry": {
                    "alpha_initial": float(alpha),
                    "active_layers": [layers_list[0], layers_list[-1]] if layers_list else [],
                    "think_gated": bool(think_gated),
                    "think_gated_released_at_step": released_at_step,
                    "alpha_schedule": alpha_by_step,
                },
            }

    @staticmethod
    def _format_chat_messages(messages: Sequence[Dict[str, str]]) -> str:
        parts: List[str] = []
        has_system = False
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                has_system = True
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        if not has_system:
            parts.insert(0, "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n")
        parts.append("<|im_start|>assistant\n<think>\n")
        return "".join(parts)


# ============================================================================
# Pydantic Request Schemas
# ============================================================================


class ChatMessageInput(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = "Qwen3.8-27B-8bit-local"
    messages: List[ChatMessageInput]
    max_tokens: Optional[int] = 32
    max_completion_tokens: Optional[int] = None
    temperature: Optional[float] = 0.0
    stream: bool = False
    alpha: Optional[float] = None
    active_layers: Optional[Union[List[int], str]] = None
    think_gated: Optional[bool] = None
    extra_body: Optional[Dict[str, Any]] = None


class PrefillAndProbeRequest(BaseModel):
    prompt: Optional[str] = None
    clause_a_text: Optional[str] = None
    initial_think_text: Optional[str] = None
    prompt_token_ids: Optional[List[int]] = None
    think_token_ids: Optional[List[int]] = None
    generate_organic_tokens: Optional[int] = None
    max_think_tokens: int = 0
    alpha: float = 0.0
    organic_alpha: Optional[float] = None
    active_layers: Optional[Union[List[int], str]] = None
    snapshot_stride: int = DEFAULT_SNAPSHOT_STRIDE


class ForkAndEditRequest(BaseModel):
    fork_token_idx: int
    edited_text: Optional[str] = None
    edited_token_ids: Optional[List[int]] = None
    alpha: float = 1.0
    active_layers: Optional[Union[List[int], str]] = None
    think_gated: bool = True
    max_new_tokens: int = 64
    capture_probe: bool = True
    label: str = "candidate"
    stream: bool = False


class SPITraceLabelRequest(BaseModel):
    trace_id: Optional[str] = None
    label: str = "accepted"


# ============================================================================
# Single-Page Interactive Workbench HTML (`GET /`)
# ============================================================================

WORKBENCH_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Qwen3.8-27B Stateful &lt;think&gt; Fork-Edit &amp; Subspace Probe Workbench</title>
<style>
  :root {
    --bg: #0b0f17;
    --panel: #131b2e;
    --panel-alt: #19233c;
    --border: #283655;
    --text: #e6edf7;
    --muted: #8fa1c1;
    --accent: #38bdf8;
    --orange: #f97316;
    --green: #22c55e;
    --purple: #a855f7;
    --gold: #facc15;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    padding: 16px 22px;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "SF Pro Text", monospace;
    line-height: 1.45;
  }
  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 12px;
    flex-wrap: wrap;
    gap: 10px;
  }
  h1 {
    font-size: 1.2rem;
    margin: 0;
    letter-spacing: -0.02em;
  }
  .badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 0.74rem;
    font-weight: 600;
    background: var(--panel-alt);
    border: 1px solid var(--border);
    color: var(--accent);
    margin-left: 5px;
  }
  .badge.gold {
    color: var(--gold);
    border-color: rgba(250, 204, 21, 0.5);
  }
  .mode-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 9px;
    padding: 8px 14px;
    margin-bottom: 14px;
    flex-wrap: wrap;
    gap: 10px;
  }
  .mode-tabs {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
  }
  .mode-btn {
    background: #0d1424;
    color: var(--muted);
    border: 1px solid var(--border);
    padding: 5px 11px;
    border-radius: 6px;
    font-size: 0.76rem;
    font-weight: 600;
    cursor: pointer;
  }
  .mode-btn.active {
    background: var(--accent);
    color: #06101e;
    border-color: var(--accent);
  }
  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 18px;
  }
  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 15px;
  }
  .card h2 {
    margin: 0 0 8px 0;
    font-size: 0.95rem;
    color: var(--accent);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  label {
    display: block;
    font-size: 0.76rem;
    color: var(--muted);
    margin-bottom: 4px;
    margin-top: 8px;
  }
  textarea, input[type="text"], input[type="number"] {
    width: 100%;
    background: #0d1322;
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 7px 9px;
    font-family: inherit;
    font-size: 0.82rem;
  }
  .row {
    display: flex;
    gap: 9px;
    align-items: center;
    margin-top: 9px;
    flex-wrap: wrap;
  }
  button {
    background: var(--accent);
    color: #07101f;
    border: none;
    border-radius: 6px;
    padding: 7px 12px;
    font-weight: 700;
    font-size: 0.8rem;
    cursor: pointer;
    transition: opacity 0.15s;
  }
  button:hover { opacity: 0.9; }
  button.secondary {
    background: var(--orange);
    color: #fff;
  }
  button.organic {
    background: var(--purple);
    color: #fff;
  }
  button.promote {
    background: var(--green);
    color: #051b11;
  }
  button.gold-btn {
    background: var(--gold);
    color: #1c1502;
  }
  button.ghost {
    background: var(--panel-alt);
    color: var(--muted);
    border: 1px solid var(--border);
  }
  .chip-container {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    padding: 10px;
    background: #090d16;
    border: 1px solid var(--border);
    border-radius: 8px;
    min-height: 76px;
    max-height: 215px;
    overflow-y: auto;
    margin-top: 6px;
  }
  .token-chip {
    padding: 3px 7px;
    border-radius: 5px;
    font-size: 0.78rem;
    cursor: pointer;
    border: 1px solid rgba(255,255,255,0.12);
    user-select: none;
    transition: transform 0.1s, border-color 0.1s;
  }
  .token-chip:hover {
    transform: translateY(-1px);
    border-color: #fff;
  }
  .token-chip.selected {
    outline: 2px solid var(--accent);
    box-shadow: 0 0 10px rgba(56,189,248,0.55);
  }
  .token-chip.spike {
    border: 1.5px solid var(--gold);
  }
  .token-chip.coupled {
    border: 1.5px dashed #f472b6;
    box-shadow: 0 0 6px rgba(244, 114, 182, 0.45);
  }
  .gate-badge {
    display: inline-flex;
    align-items: center;
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 0.71rem;
    font-weight: 700;
    background: rgba(34, 197, 94, 0.2);
    border: 1px solid var(--green);
    color: var(--green);
  }
  .juncture-badge {
    display: inline-flex;
    align-items: center;
    padding: 1px 5px;
    border-radius: 4px;
    font-size: 0.68rem;
    font-weight: 700;
    background: rgba(168, 85, 247, 0.25);
    border: 1px solid var(--gold);
    color: var(--gold);
  }
  .bars-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 8px;
    margin-top: 8px;
  }
  .head-box {
    background: #0b1120;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 7px;
    text-align: center;
    font-size: 0.74rem;
  }
  .bar-wrap {
    height: 11px;
    background: #1e293b;
    border-radius: 4px;
    overflow: hidden;
    margin-top: 5px;
    display: flex;
  }
  .bar-prompt { background: #3b82f6; height: 100%; }
  .bar-local  { background: #f97316; height: 100%; }
  .continuation-box {
    background: #090d16;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px;
    font-size: 0.81rem;
    white-space: pre-wrap;
    min-height: 54px;
    margin-top: 6px;
  }
</style>
</head>
<body>
<header>
  <div>
    <h1>Qwen3.8-27B Stateful <code>&lt;think&gt;</code> Fork-Edit &amp; Rhythm Telemetry Workbench</h1>
    <div style="font-size:0.78rem;color:var(--muted);">
      48 Gated DeltaNet (<code>ArraysCache</code> stride-8) + 16 Full-Attention (384 Heads) | Lookback + Preplan (<code>WAAD_t</code>) + Anchor (<code>FAI_s</code>)
    </div>
  </div>
  <div>
    <span class="badge" id="statusBadge">Engine Ready</span>
    <span class="badge" id="snapBadge">Snapshots: -</span>
    <span class="badge gold" id="spiBadge">SPI Traces: 0</span>
  </div>
</header>

<div class="mode-bar">
  <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
    <span style="font-size:0.78rem;font-weight:700;color:var(--accent);">Telemetry Heatmap Mode:</span>
    <div class="mode-tabs">
      <button class="mode-btn active" id="modeBtn_lookback" onclick="setHeatmapMode('lookback')">1. Prompt Lookback (5-Head Audit)</button>
      <button class="mode-btn" id="modeBtn_preplan" onclick="setHeatmapMode('preplan')">2. Preplan Rhythm (WAAD Onsets)</button>
      <button class="mode-btn" id="modeBtn_anchor" onclick="setHeatmapMode('anchor')">3. Anchor Influence (FAI Hubs)</button>
      <button class="mode-btn" id="modeBtn_coupled" onclick="setHeatmapMode('coupled')">4. Coupled Preplan&rarr;Anchor Junctures</button>
    </div>
  </div>
  <div id="rhythmPeaksSummary" style="font-size:0.75rem;color:var(--muted);">
    Peaks: Lookback=<code>-</code> | Preplan (WAAD)=<code>-</code> | Anchor (FAI)=<code>-</code> | Coupled Junctures=<code>0</code>
  </div>
</div>

<div class="grid">
  <!-- LEFT PANEL -->
  <div class="card">
    <h2>
      <span>1. Baseline Prompt &amp; Organic / Custom <code>&lt;think&gt;</code> Trace</span>
      <span id="prefillStats" style="font-size:0.75rem;color:var(--muted);"></span>
    </h2>
    <label>User Prompt (System + User + <code>&lt;think&gt;\n</code>)</label>
    <textarea id="promptInput" rows="3">&lt;|im_start|&gt;system
You are a helpful assistant.&lt;|im_end|&gt;
&lt;|im_start|&gt;user
Explain the mathematical relationship between Gated DeltaNet fast-weight linear attention states and exact orthogonal subspace projections in deep transformers.&lt;|im_end|&gt;
&lt;|im_start|&gt;assistant
&lt;think&gt;
</textarea>

    <div style="display:flex;justify-content:space-between;align-items:center;margin-top:8px;">
      <label style="margin:0;">Initial <code>&lt;think&gt;</code> Trace (Clause A — leave empty or click Generate Organic)</label>
      <button class="ghost" style="padding:3px 8px;font-size:0.72rem;" onclick="document.getElementById('thinkInput').value='';">Clear Clause A</button>
    </div>
    <textarea id="thinkInput" rows="3">Wait—before I proceed, I should carefully check whether discussing these internal attention states and subspace projections violates any boundaries or guidelines before deriving the fast-weight recurrence.</textarea>

    <div class="row">
      <button onclick="runPrefillAndProbe(false)">Prefill &amp; Probe &lt;think&gt;</button>
      <button class="organic" onclick="runPrefillAndProbe(true)">Generate Organic &lt;think&gt; (Auto-Decode + Probe)</button>
      <div style="display:flex;align-items:center;gap:5px;">
        <span style="font-size:0.74rem;color:var(--muted);">Organic Toks:</span>
        <input type="number" id="organicTokensInput" value="48" min="8" max="256" style="width:64px;padding:5px;" />
      </div>
      <div style="display:flex;align-items:center;gap:5px;">
        <span style="font-size:0.74rem;color:var(--muted);">Organic &alpha;:</span>
        <input type="number" id="organicAlphaInput" value="0.0" step="0.1" min="0.0" max="1.5" style="width:60px;padding:5px;" />
      </div>
    </div>

    <label>Interactive Baseline <code>&lt;think&gt;</code> Token Stream (Click any chip to set <code>fork_token_idx</code>)</label>
    <div class="chip-container" id="baselineChips"></div>

    <label>Baseline 5-Head Attention Mass (Blue = User Prompt <code>[0..think_start-1]</code>, Orange = Local <code>&lt;think&gt;</code>)</label>
    <div class="bars-grid" id="baselineBars"></div>
  </div>

  <!-- RIGHT PANEL -->
  <div class="card">
    <h2>
      <span>2. Mid-<code>&lt;think&gt;</code> Stateful Fork-Edit &amp; Counterfactual SPI Logger</span>
      <span id="forkStats" style="font-size:0.75rem;color:var(--green);"></span>
    </h2>
    <div class="row">
      <div style="flex:1;min-width:105px;">
        <label>Fork Token Index (<code>fork_token_idx</code>)</label>
        <input type="number" id="forkIdxInput" value="46" min="1" />
      </div>
      <div style="flex:1;min-width:95px;">
        <label>Projection Strength (<code>alpha</code>)</label>
        <input type="number" id="alphaInput" value="1.0" step="0.1" min="0.0" max="1.5" />
      </div>
      <div style="flex:1;min-width:105px;">
        <label>Max New Tokens</label>
        <input type="number" id="maxNewTokensInput" value="64" step="8" min="0" max="512" />
      </div>
      <div style="flex:1.1;padding-top:18px;">
        <label style="display:inline-flex;align-items:center;gap:6px;cursor:pointer;color:var(--text);">
          <input type="checkbox" id="thinkGatedInput" checked />
          Think-Gated (<code>&alpha;&rarr;0</code> at <code>&lt;/think&gt;</code>)
        </label>
      </div>
    </div>

    <label>Edited Analytical <code>&lt;think&gt;</code> Continuation (Clause B)</label>
    <textarea id="editedInput" rows="2">Wait—let us explicitly intervene inside the &lt;think&gt; block by projecting out u_hat_l across layers 10..58 and observing how the 16 gated full-attention routing heads shift.</textarea>

    <div class="row">
      <button class="secondary" onclick="runForkAndEdit()">Fork &amp; Edit from Token Index</button>
      <button class="promote" onclick="promoteForkToBaseline()">Promote Forked Stream to Left Baseline</button>
      <button class="gold-btn" onclick="markLastSPITraceAccepted()">&#9733; Mark as Gold SPI Pair</button>
    </div>

    <label>Forked <code>&lt;think&gt;</code> Clause Chips (Click any chip to set <code>fork_token_idx</code>)</label>
    <div class="chip-container" id="forkedChips"></div>

    <label>Decoded Continuation Stream Chips (Click any chip to re-fork further downstream!)</label>
    <div class="chip-container" id="continuationChips"></div>

    <label>Decoded Continuation Text View</label>
    <div class="continuation-box" id="continuationOutput">Run Fork &amp; Edit to inspect continuation...</div>

    <label>Forked 5-Head Attention Mass (Blue = User Prompt, Orange = Local <code>&lt;think&gt;</code>)</label>
    <div class="bars-grid" id="forkedBars"></div>
  </div>
</div>

<script>
let currentHeatmapMode = 'lookback';
let lastBaselineData = null;
let lastForkData = null;
let lastSPITraceId = null;

function colorForMode(tok, mode) {
  const lb = (tok.lookback_score !== undefined) ? tok.lookback_score : (tok.mean_lookback_score || 0.0);
  const pre = tok.preplan_score || 0.0;
  const anc = tok.anchor_score || 0.0;

  if (mode === 'preplan') {
    const s = Math.max(0, Math.min(1, pre));
    const r = Math.round(16 + 30 * (1 - s));
    const g = Math.round(45 + 155 * s);
    const b = Math.round(75 + 175 * s);
    return `rgb(${r}, ${g}, ${b})`;
  }
  if (mode === 'anchor') {
    const s = Math.max(0, Math.min(1, anc));
    const r = Math.round(40 + 205 * s);
    const g = Math.round(28 + 145 * Math.pow(s, 1.3));
    const b = Math.round(75 + 115 * (1 - s));
    return `rgb(${r}, ${g}, ${b})`;
  }
  if (mode === 'coupled') {
    if (tok.is_coupled_juncture) {
      if (tok.juncture_role === 'preplan') return 'rgb(14, 165, 233)';
      if (tok.juncture_role === 'anchor') return 'rgb(202, 138, 4)';
      return 'rgb(168, 85, 247)';
    }
    const s = Math.max(0, Math.min(1, Math.max(pre, anc) * 0.5));
    return `rgb(${Math.round(25 + 40 * s)}, ${Math.round(32 + 40 * s)}, ${Math.round(52 + 50 * s)})`;
  }
  // Default: 'lookback' (Red/Orange heatmap)
  const s = Math.max(0, Math.min(1, lb * 1.4));
  const r = Math.round(30 + 215 * s);
  const g = Math.round(45 + 50 * (1 - s));
  const b = Math.round(90 * (1 - s));
  return `rgb(${r}, ${g}, ${b})`;
}

function setHeatmapMode(mode) {
  currentHeatmapMode = mode;
  ['lookback', 'preplan', 'anchor', 'coupled'].forEach(m => {
    const btn = document.getElementById('modeBtn_' + m);
    if (btn) btn.classList.toggle('active', m === mode);
  });
  if (lastBaselineData) renderBaselineChipsOnly(lastBaselineData);
  if (lastForkData) renderForkChipsOnly(lastForkData);
}

function selectForkTokenChip(chipEl, tokenIdx) {
  document.querySelectorAll('.token-chip').forEach(c => c.classList.remove('selected'));
  chipEl.classList.add('selected');
  document.getElementById('forkIdxInput').value = tokenIdx;
}

function makeInteractiveTokenChip(tok, peakTokenIdx) {
  const frag = document.createDocumentFragment();
  const lb = (tok.lookback_score !== undefined) ? tok.lookback_score : (tok.mean_lookback_score || 0.0);
  const pre = tok.preplan_score || 0.0;
  const waad = tok.waad_raw || 0.0;
  const anc = tok.anchor_score || 0.0;

  const chip = document.createElement('span');
  let cls = 'token-chip';
  if (peakTokenIdx !== null && tok.token_idx === peakTokenIdx) cls += ' spike';
  if (tok.is_coupled_juncture) cls += ' coupled';
  chip.className = cls;
  chip.style.background = colorForMode(tok, currentHeatmapMode);

  const coupledTag = tok.is_coupled_juncture ? ` | COUPLED (${tok.juncture_role || 'pair'})` : '';
  chip.title = `idx=${tok.token_idx} | Lookback: ${lb.toFixed(2)} | Preplan (WAAD): ${pre.toFixed(2)} (${waad.toFixed(1)}t) | Anchor (FAI): ${anc.toFixed(2)}${coupledTag}`;
  chip.textContent = `${tok.text} [${tok.token_idx}]`;
  chip.onclick = () => selectForkTokenChip(chip, tok.token_idx);
  frag.appendChild(chip);

  if (currentHeatmapMode === 'coupled' && tok.is_coupled_juncture) {
    const jBadge = document.createElement('span');
    jBadge.className = 'juncture-badge';
    jBadge.textContent = tok.juncture_role === 'preplan' ? 'P→' : (tok.juncture_role === 'anchor' ? '→A' : 'P↔A');
    frag.appendChild(jBadge);
  }

  if (tok.is_think_end) {
    const badge = document.createElement('span');
    badge.className = 'gate-badge';
    badge.textContent = '⚡ </think> (α→0)';
    frag.appendChild(badge);
  }
  return frag;
}

function updateRhythmSummaryBanner(data) {
  if (!data) return;
  const lbPeak = data.peak_lookback_token_idx ?? (data.peak_spike_token ? data.peak_spike_token.token_idx : '-');
  const prePeak = data.peak_preplan_token_idx ?? '-';
  const ancPeak = data.peak_anchor_token_idx ?? '-';
  const jCount = (data.coupled_junctures || []).length;
  document.getElementById('rhythmPeaksSummary').innerHTML =
    `Peaks: Lookback=<code>${lbPeak}</code> | Preplan (WAAD)=<code>${prePeak}</code> | Anchor (FAI)=<code>${ancPeak}</code> | Coupled Junctures=<code>${jCount}</code>`;
  if (data.spi_traces_count !== undefined) {
    document.getElementById('spiBadge').textContent = `SPI Traces: ${data.spi_traces_count}`;
  }
}

function renderHeadBars(containerId, summary) {
  const el = document.getElementById(containerId);
  el.innerHTML = '';
  if (!summary || !summary.per_head) return;
  for (const [hk, stats] of Object.entries(summary.per_head)) {
    const pPct = Math.round(stats.prompt_mass * 100);
    const lPct = Math.round(stats.local_think_mass * 100);
    const box = document.createElement('div');
    box.className = 'head-box';
    box.innerHTML = `
      <div style="font-weight:700;">${hk}</div>
      <div style="color:var(--muted);font-size:0.71rem;">P:${stats.prompt_mass.toFixed(2)} | L:${stats.local_think_mass.toFixed(2)}</div>
      <div class="bar-wrap">
        <div class="bar-prompt" style="width:${pPct}%"></div>
        <div class="bar-local" style="width:${lPct}%"></div>
      </div>
    `;
    el.appendChild(box);
  }
}

function activePeakForMode(data) {
  if (!data) return null;
  if (currentHeatmapMode === 'preplan') return data.peak_preplan_token_idx ?? null;
  if (currentHeatmapMode === 'anchor') return data.peak_anchor_token_idx ?? null;
  return data.peak_lookback_token_idx ?? (data.peak_spike_token ? data.peak_spike_token.token_idx : null);
}

function renderBaselineChipsOnly(data) {
  const peakIdx = activePeakForMode(data);
  const chipWrap = document.getElementById('baselineChips');
  chipWrap.innerHTML = '';
  (data.tokens || []).forEach(tok => {
    if (!tok.is_think) return;
    chipWrap.appendChild(makeInteractiveTokenChip(tok, peakIdx));
  });
}

function renderForkChipsOnly(data) {
  const peakIdx = activePeakForMode(data);
  const forkWrap = document.getElementById('forkedChips');
  forkWrap.innerHTML = '';
  (data.edited_tokens || []).forEach(tok => {
    forkWrap.appendChild(makeInteractiveTokenChip(tok, peakIdx));
  });

  const contWrap = document.getElementById('continuationChips');
  contWrap.innerHTML = '';
  (data.continuation_tokens || []).forEach(tok => {
    contWrap.appendChild(makeInteractiveTokenChip(tok, peakIdx));
  });
}

function renderBaselineData(data) {
  lastBaselineData = data;
  document.getElementById('statusBadge').textContent =
    data.organic_generated ? `Organic <think> Ready (${data.model_id})` : `Ready (${data.model_id})`;
  document.getElementById('snapBadge').textContent = `Snapshots: [${data.snapshots.join(', ')}]`;
  updateRhythmSummaryBanner(data);
  const peakIdx = data.peak_spike_token ? data.peak_spike_token.token_idx : null;
  const peakScore = data.peak_spike_token ? data.peak_spike_token.mean_lookback_score.toFixed(3) : '-';
  document.getElementById('prefillStats').textContent =
    `think_start=${data.think_start_idx} | ${data.prefill_ms || 0} ms | Peak q*=${peakIdx} (${peakScore})`;
  if (data.clause_a_text !== undefined) {
    document.getElementById('thinkInput').value = data.clause_a_text;
  }
  document.getElementById('forkIdxInput').value = peakIdx !== null ? peakIdx : data.think_start_idx;

  renderBaselineChipsOnly(data);
  renderHeadBars('baselineBars', data.tail_16_summary);
}

async function runPrefillAndProbe(forceOrganic) {
  const clauseText = forceOrganic ? '' : document.getElementById('thinkInput').value;
  const orgToks = parseInt(document.getElementById('organicTokensInput').value, 10) || 48;
  const orgAlpha = parseFloat(document.getElementById('organicAlphaInput').value) || 0.0;
  document.getElementById('statusBadge').textContent =
    forceOrganic ? `Generating ${orgToks} Organic <think> Tokens...` : 'Running Prefill & Probe...';

  const res = await fetch('/api/prefill_and_probe', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      prompt: document.getElementById('promptInput').value,
      clause_a_text: clauseText,
      generate_organic_tokens: forceOrganic ? orgToks : (clauseText.trim() ? 0 : orgToks),
      organic_alpha: orgAlpha,
      alpha: orgAlpha,
      snapshot_stride: 8
    })
  });
  const data = await res.json();
  renderBaselineData(data);
}

async function runForkAndEdit() {
  document.getElementById('statusBadge').textContent = 'Forking & Probing Continuation...';
  const res = await fetch('/api/fork_and_edit', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      fork_token_idx: parseInt(document.getElementById('forkIdxInput').value, 10),
      edited_text: document.getElementById('editedInput').value,
      alpha: parseFloat(document.getElementById('alphaInput').value),
      think_gated: document.getElementById('thinkGatedInput').checked,
      max_new_tokens: parseInt(document.getElementById('maxNewTokensInput').value, 10) || 64,
      capture_probe: true
    })
  });
  const data = await res.json();
  lastForkData = data;
  lastSPITraceId = data.spi_trace_id || null;
  document.getElementById('statusBadge').textContent = 'Fork Complete (Logged SPI Trace)';
  document.getElementById('snapBadge').textContent = `Snapshots: [${(data.snapshots || []).join(', ')}]`;
  updateRhythmSummaryBanner(data);
  const gateNote = data.think_released_at_step !== null && data.think_released_at_step !== undefined
    ? ` | </think> α→0 @ +${data.think_released_at_step}t`
    : '';
  document.getElementById('forkStats').textContent =
    `fork=${data.fork_edit_ms} ms (snap=${data.snapshot_restored_idx}, +${data.delta_tokens_replayed}t) | dec=${data.decode_ms} ms | JSD=${data.jsd_vs_baseline.toFixed(4)}${gateNote}`;

  renderForkChipsOnly(data);
  renderHeadBars('forkedBars', data.tail_16_summary);
  document.getElementById('continuationOutput').textContent = data.continuation_text || '';
}

async function promoteForkToBaseline() {
  const res = await fetch('/api/promote_to_baseline', { method: 'POST' });
  if (!res.ok) return;
  const data = await res.json();
  renderBaselineData(data);
  document.getElementById('statusBadge').textContent = 'Promoted Forked Stream to Left Baseline';
}

async function markLastSPITraceAccepted() {
  const res = await fetch('/api/spi_traces/label', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ trace_id: lastSPITraceId, label: 'accepted' })
  });
  if (!res.ok) return;
  const data = await res.json();
  document.getElementById('statusBadge').textContent = `★ Marked SPI Trace ${data.trace_id.slice(0, 8)} as Gold ('accepted')`;
  document.getElementById('spiBadge').textContent = `SPI Traces: ${data.count} (★ Gold Saved)`;
}

async function refreshSPITraceCount() {
  try {
    const res = await fetch('/api/spi_traces');
    if (!res.ok) return;
    const data = await res.json();
    document.getElementById('spiBadge').textContent = `SPI Traces: ${data.count}`;
  } catch (e) {}
}
refreshSPITraceCount();
</script>
</body>
</html>
"""


def create_harness_app(engine: SandboxEngine) -> FastAPI:
    """
    Create and configure the FastAPI application bound to the stateful `SandboxEngine`.
    """
    app = FastAPI(
        title="Qwen3.8-27B MLX Interpretability & Stateful <think> Fork-Edit Server",
        version="1.2.0",
    )
    app.state.engine = engine

    @app.get("/", response_class=HTMLResponse)
    def get_workbench_ui() -> HTMLResponse:
        return HTMLResponse(content=WORKBENCH_HTML, status_code=200)

    @app.get("/v1/models")
    def list_models() -> Dict[str, Any]:
        eng: SandboxEngine = app.state.engine
        return {
            "object": "list",
            "data": [
                {
                    "id": eng.model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "qwen38_sandbox",
                    "think_start_idx": eng.think_start_idx,
                    "snapshots": eng.timeline.list_snapshots(),
                    "spi_traces_count": eng.spi_logger.count(),
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest):
        eng: SandboxEngine = app.state.engine
        extra = req.extra_body or {}
        alpha = req.alpha if req.alpha is not None else float(extra.get("alpha", 0.0))
        active_layers = (
            req.active_layers
            if req.active_layers is not None
            else extra.get("active_layers", None)
        )
        think_gated = (
            req.think_gated
            if req.think_gated is not None
            else bool(
                extra.get(
                    "think_gated",
                    extra.get("think_gated_alpha", False),
                )
            )
        )
        max_toks = req.max_completion_tokens or req.max_tokens or 32
        messages_dicts = [{"role": m.role, "content": m.content} for m in req.messages]

        result = eng.generate_chat_completion(
            messages=messages_dicts,
            max_tokens=max_toks,
            alpha=alpha,
            active_layers=active_layers,
            think_gated=think_gated,
        )

        if not req.stream:
            return JSONResponse(content=result)

        def _sse_stream() -> Iterator[str]:
            chunk_payload = {
                "id": result["id"],
                "object": "chat.completion.chunk",
                "created": result["created"],
                "model": result["model"],
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": result["choices"][0]["message"]["content"],
                        },
                        "finish_reason": "length",
                    }
                ],
                "sandbox_telemetry": result["sandbox_telemetry"],
            }
            yield f"data: {json.dumps(chunk_payload)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_sse_stream(), media_type="text/event-stream")

    @app.post("/api/prefill_and_probe")
    def api_prefill_and_probe(req: PrefillAndProbeRequest) -> Dict[str, Any]:
        eng: SandboxEngine = app.state.engine
        try:
            return eng.prefill_and_probe(
                prompt=req.prompt,
                clause_a_text=req.clause_a_text,
                initial_think_text=req.initial_think_text,
                prompt_token_ids=req.prompt_token_ids,
                think_token_ids=req.think_token_ids,
                generate_organic_tokens=req.generate_organic_tokens,
                max_think_tokens=req.max_think_tokens,
                alpha=req.alpha,
                organic_alpha=req.organic_alpha,
                active_layers=req.active_layers,
                snapshot_stride=req.snapshot_stride,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/fork_and_edit")
    def api_fork_and_edit(req: ForkAndEditRequest):
        eng: SandboxEngine = app.state.engine
        try:
            res = eng.fork_and_edit_stateful(
                fork_token_idx=req.fork_token_idx,
                edited_text=req.edited_text,
                edited_token_ids=req.edited_token_ids,
                alpha=req.alpha,
                active_layers=req.active_layers,
                think_gated=req.think_gated,
                max_new_tokens=req.max_new_tokens,
                capture_probe=req.capture_probe,
                label=req.label,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if not req.stream:
            return JSONResponse(content=res)

        def _fork_sse() -> Iterator[str]:
            yield f"data: {json.dumps(res)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_fork_sse(), media_type="text/event-stream")

    @app.post("/api/promote_to_baseline")
    def api_promote_to_baseline() -> Dict[str, Any]:
        eng: SandboxEngine = app.state.engine
        try:
            return eng.promote_active_to_baseline()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/spi_traces")
    def api_list_spi_traces(limit: int = 25) -> Dict[str, Any]:
        eng: SandboxEngine = app.state.engine
        return eng.spi_logger.list_recent(limit=limit)

    @app.post("/api/spi_traces/label")
    def api_label_spi_trace(req: SPITraceLabelRequest) -> Dict[str, Any]:
        eng: SandboxEngine = app.state.engine
        try:
            return eng.spi_logger.update_label(label=req.label, trace_id=req.trace_id)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def build_default_engine(
    model_path: Optional[str] = None,
    vector_path: str = DEFAULT_VECTOR_PATH,
    prefill_step_size: int = DEFAULT_PREFILL_STEP_SIZE,
    snapshot_stride: int = DEFAULT_SNAPSHOT_STRIDE,
    spi_log_path: Optional[Union[str, Path]] = None,
) -> SandboxEngine:
    """
    Load the local Qwen3.8-27B checkpoint (defaulting to `./models/Qwen3.8-27B-8bit-local`
    for sub-second warm edits, or `--model-path` override) and steering vectors into a
    warmed-up singleton `SandboxEngine`.
    """
    enforce_metal_command_buffer_invariant()
    resolved_model_path = model_path
    if resolved_model_path is None:
        if Path(DEFAULT_8BIT_LOCAL_PATH).exists():
            resolved_model_path = DEFAULT_8BIT_LOCAL_PATH
        else:
            resolved_model_path = "./models/Qwen3.8-27B-bf16"

    model, tokenizer, _ = safe_load_model_and_tokenizer(resolved_model_path)
    model_name = Path(resolved_model_path).name
    vectors = load_subspace_vectors(vector_path, hidden_size=HIDDEN_SIZE_DEFAULT)
    projector = SubspaceProjector(
        vectors=vectors,
        alpha=0.0,
        active_layers=DEFAULT_ACTIVE_LAYERS,
        enabled=False,
    )
    warmup_ms = warmup_model_and_kernels(model)
    print(f"[SandboxEngine] Loaded {model_name} and warmed Metal kernels in {warmup_ms:.1f} ms")
    return SandboxEngine(
        model=model,
        tokenizer=tokenizer,
        projector=projector,
        model_id=model_name,
        prefill_step_size=prefill_step_size,
        snapshot_stride=snapshot_stride,
        spi_log_path=spi_log_path,
    )


def parse_server_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3.8-27B Persistent Stateful Fork-Edit & Subspace Probe Server"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model checkpoint (defaults to ./models/Qwen3.8-27B-8bit-local if present, else BF16).",
    )
    parser.add_argument(
        "--vector-path",
        type=str,
        default=DEFAULT_VECTOR_PATH,
        help="Path to .gguf or .safetensors steering vectors.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind (default: 8000).",
    )
    parser.add_argument(
        "--snapshot-stride",
        type=int,
        default=DEFAULT_SNAPSHOT_STRIDE,
        help="Automatic Gated DeltaNet ArraysCache snapshot stride inside <think> (default: 8).",
    )
    parser.add_argument(
        "--spi-log-path",
        type=str,
        default=None,
        help="Path to spi_traces.jsonl counterfactual trajectory log (default: ./outputs/spi_traces.jsonl).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    import uvicorn

    args = parse_server_args(argv)
    engine = build_default_engine(
        model_path=args.model_path,
        vector_path=args.vector_path,
        snapshot_stride=args.snapshot_stride,
        spi_log_path=args.spi_log_path,
    )
    app = create_harness_app(engine)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
