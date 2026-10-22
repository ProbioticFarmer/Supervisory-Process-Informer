---
name: qwen38-sandbox-shell-execution
description: Critical shell execution and Python virtual environment rules for the Pi_SynDE / qwen38_sandbox workspace on macOS Apple Silicon.
---

# Critical Shell Execution & Virtual Environment Rules (`Pi_SynDE` / `qwen38_sandbox`)

Every `run_command` tool call executes in a fresh, stateless macOS `zsh` subshell where `.venv` is NOT active and bare `python` / `pip` are not on `PATH` (causing `Exit code 127` if invoked without explicit paths or without access to the underlying `uv` Python toolchain).

## Mandatory Execution Rules
1. **Direct `.venv` Binary Invocation Only:**
   - All shell commands MUST invoke `./.venv/bin/python` or `./.venv/bin/pip` (or `./qwen38_sandbox/.venv/bin/python` / `./qwen38_sandbox/.venv/bin/pip` from the repo root) directly.
   - **NEVER** invoke bare `python`, `python3`, or `pip`.
2. **Zero `Exit Code 127` Retries:**
   - Because `.venv/bin/python` symlinks to `/Users/joshua/.local/share/uv/python/cpython-3.12.9-macos-aarch64-none/bin/python3.12` outside the workspace directory, standard sandbox mode returns `zsh:1: operation not permitted` (`Exit code 127`). Always execute `.venv/bin/python` and `.venv/bin/pip` commands with `BypassSandbox: true` on the first call so there are zero `Exit code 127` retries.
3. **Supply-Chain & Hardware Invariants:**
   - NEVER pass `trust_remote_code=True` to any Hugging Face / `transformers` / `mlx_lm` call.
   - Do NOT set or increase `MLX_METAL_COMMAND_BUFFER_MB`.
   - Never project Layer 0 (`assert 0 not in active_layers`).
