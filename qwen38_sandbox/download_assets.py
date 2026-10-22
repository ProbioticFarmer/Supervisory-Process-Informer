"""
download_assets.py — Safe Vector & Pinned Checkpoint Loader for Qwen3.8-27B (MLX)

Security & Supply-Chain Constraints (STRICT):
  - NEVER passes `trust_remote_code=True` to any Hugging Face call.
  - Pulls `./vectors` from a user-passed `--vector-repo` (defaulting to local `./vectors`
    inspection if already present) and immediately runs a smoke check with
    `load_subspace_vectors()` printing active layer indices, shapes, and L2 norms.
  - Pulls `Qwen/Qwen3.8-27B` pinned to revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
    into `./models/Qwen3.8-27B-bf16` (allowing `"*.json"`, `"*.safetensors"`, `"tokenizer*"`,
    `"merges.txt"`, `"vocab.json"`), or verifies the directory if it already exists.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import mlx.core as mx
from huggingface_hub import hf_hub_download, snapshot_download

from subspace_projector import HIDDEN_SIZE_DEFAULT, load_subspace_vectors


PINNED_QWEN38_REPO_ID = "Qwen/Qwen3.8-27B"
PINNED_QWEN38_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
DEFAULT_MODEL_LOCAL_DIR = "./models/Qwen3.8-27B-bf16"
DEFAULT_VECTOR_LOCAL_DIR = "./vectors"

MODEL_ALLOW_PATTERNS: List[str] = [
    "*.json",
    "*.safetensors",
    "tokenizer*",
    "merges.txt",
    "vocab.json",
]

VECTOR_ALLOW_PATTERNS: List[str] = [
    "*.safetensors",
    "*.gguf",
    "*.json",
]


def smoke_check_vectors(
    vector_path_or_dir: Union[str, Path],
    hidden_size: int = HIDDEN_SIZE_DEFAULT,
) -> Dict[str, Any]:
    """
    Locate `.safetensors` or `.gguf` vector files in `vector_path_or_dir`, load them via
    `load_subspace_vectors()`, and print/return a smoke-check summary of active layer
    indices, shapes, dtypes, and L2 norms.
    """
    target = Path(vector_path_or_dir)
    if not target.exists():
        raise FileNotFoundError(f"Vector path or directory does not exist: {target}")

    if target.is_file():
        candidates = [target]
    else:
        candidates = sorted(
            list(target.glob("*.safetensors")) + list(target.glob("*.gguf"))
        )

    if not candidates:
        raise FileNotFoundError(
            f"No .safetensors or .gguf vector files found in {target}"
        )

    vec_file = candidates[0]
    vectors = load_subspace_vectors(vec_file, hidden_size=hidden_size)
    sorted_layers = sorted(vectors.keys())
    active_layers = [l for l in sorted_layers if l != 0]

    norms: Dict[int, float] = {}
    shapes: Dict[int, tuple] = {}
    for l_idx in sorted_layers:
        v = vectors[l_idx]
        shapes[l_idx] = tuple(v.shape)
        norm_val = float(mx.sqrt(mx.sum(v.astype(mx.float32) ** 2)).item())
        norms[l_idx] = norm_val

    min_norm = min(norms.values())
    max_norm = max(norms.values())
    sample_shape = shapes[sorted_layers[0]]
    sample_dtype = str(vectors[sorted_layers[0]].dtype)

    print(f"[Vector Smoke Check] File           : {vec_file}")
    print(f"[Vector Smoke Check] Total Layers   : {len(sorted_layers)} ({sorted_layers[0]}..{sorted_layers[-1]})")
    print(f"[Vector Smoke Check] Active Layers  : {len(active_layers)} (Layer 0 excluded: {0 not in active_layers})")
    print(f"[Vector Smoke Check] Tensor Shape   : {sample_shape} (dtype={sample_dtype})")
    print(f"[Vector Smoke Check] L2 Norm Range  : [{min_norm:.6f}, {max_norm:.6f}]")

    return {
        "file": str(vec_file),
        "loaded_layers": sorted_layers,
        "active_layers": active_layers,
        "shape": sample_shape,
        "dtype": sample_dtype,
        "norms": norms,
        "min_norm": min_norm,
        "max_norm": max_norm,
    }


def pull_or_verify_vectors(
    vector_dir: Union[str, Path] = DEFAULT_VECTOR_LOCAL_DIR,
    vector_repo: Optional[str] = None,
    vector_filename: Optional[str] = None,
    revision: Optional[str] = None,
    hidden_size: int = HIDDEN_SIZE_DEFAULT,
) -> Dict[str, Any]:
    """
    Pull `./vectors` from `--vector-repo` (or inspect local `./vectors` if already present)
    using `snapshot_download` or `hf_hub_download` with strict allow_patterns, then run
    `smoke_check_vectors()`.
    """
    local_dir = Path(vector_dir)
    existing_files = (
        sorted(list(local_dir.glob("*.safetensors")) + list(local_dir.glob("*.gguf")))
        if local_dir.exists()
        else []
    )

    if vector_repo is not None and not existing_files:
        local_dir.mkdir(parents=True, exist_ok=True)
        if vector_filename is not None:
            hf_hub_download(
                repo_id=vector_repo,
                filename=vector_filename,
                revision=revision,
                local_dir=str(local_dir),
            )
        else:
            snapshot_download(
                repo_id=vector_repo,
                revision=revision,
                local_dir=str(local_dir),
                allow_patterns=list(VECTOR_ALLOW_PATTERNS),
            )

    return smoke_check_vectors(local_dir, hidden_size=hidden_size)


def pull_or_verify_model(
    model_dir: Union[str, Path] = DEFAULT_MODEL_LOCAL_DIR,
    model_repo: str = PINNED_QWEN38_REPO_ID,
    revision: str = PINNED_QWEN38_REVISION,
    verify_only_if_exists: bool = True,
) -> Dict[str, Any]:
    """
    Pull `Qwen/Qwen3.8-27B` pinned to revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
    into `./models/Qwen3.8-27B-bf16` with `allow_patterns=["*.json", "*.safetensors",
    "tokenizer*", "merges.txt", "vocab.json"]`, or verify the directory if it already exists.
    """
    target_dir = Path(model_dir)
    if verify_only_if_exists and target_dir.exists():
        safetensor_shards = sorted(target_dir.glob("*.safetensors"))
        json_files = sorted(target_dir.glob("*.json"))
        if safetensor_shards and (target_dir / "config.json").exists():
            print(f"[Model Check] Verified existing local checkpoint: {target_dir}")
            print(f"[Model Check] Safetensors shards: {len(safetensor_shards)}, JSON files: {len(json_files)}")
            return {
                "status": "verified_existing",
                "model_dir": str(target_dir.resolve()),
                "repo_id": model_repo,
                "revision": revision,
                "num_safetensors": len(safetensor_shards),
                "num_json": len(json_files),
            }

    target_dir.mkdir(parents=True, exist_ok=True)
    downloaded_dir = snapshot_download(
        repo_id=model_repo,
        revision=revision,
        local_dir=str(target_dir),
        allow_patterns=list(MODEL_ALLOW_PATTERNS),
    )
    resolved_dir = Path(downloaded_dir).resolve()
    safetensor_shards = sorted(resolved_dir.glob("*.safetensors"))
    json_files = sorted(resolved_dir.glob("*.json"))

    print(f"[Model Check] Downloaded {model_repo}@{revision[:12]} -> {resolved_dir}")
    print(f"[Model Check] Safetensors shards: {len(safetensor_shards)}, JSON files: {len(json_files)}")
    return {
        "status": "downloaded",
        "model_dir": str(resolved_dir),
        "repo_id": model_repo,
        "revision": revision,
        "num_safetensors": len(safetensor_shards),
        "num_json": len(json_files),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safe Vector & Pinned Qwen3.8-27B Checkpoint Loader (No Remote Code)"
    )
    parser.add_argument(
        "--vector-repo",
        type=str,
        default=None,
        help="Optional Hugging Face repo ID to pull steering/projection vectors from into --vector-dir.",
    )
    parser.add_argument(
        "--vector-filename",
        type=str,
        default=None,
        help="Optional specific .safetensors or .gguf filename inside --vector-repo.",
    )
    parser.add_argument(
        "--vector-dir",
        type=str,
        default=DEFAULT_VECTOR_LOCAL_DIR,
        help="Local directory for steering/projection vectors (default: ./vectors).",
    )
    parser.add_argument(
        "--model-repo",
        type=str,
        default=PINNED_QWEN38_REPO_ID,
        help=f"Hugging Face model repository ID (default: {PINNED_QWEN38_REPO_ID}).",
    )
    parser.add_argument(
        "--model-revision",
        type=str,
        default=PINNED_QWEN38_REVISION,
        help=f"Pinned git commit hash for model download (default: {PINNED_QWEN38_REVISION}).",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=DEFAULT_MODEL_LOCAL_DIR,
        help=f"Local destination directory for BF16 checkpoint (default: {DEFAULT_MODEL_LOCAL_DIR}).",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Only pull/verify `--vector-dir` and skip model download/verification.",
    )
    parser.add_argument(
        "--skip-vectors",
        action="store_true",
        help="Only pull/verify `--model-dir` and skip vector smoke check.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    if not args.skip_vectors:
        pull_or_verify_vectors(
            vector_dir=args.vector_dir,
            vector_repo=args.vector_repo,
            vector_filename=args.vector_filename,
        )

    if not args.skip_model:
        pull_or_verify_model(
            model_dir=args.model_dir,
            model_repo=args.model_repo,
            revision=args.model_revision,
        )


if __name__ == "__main__":
    main()
