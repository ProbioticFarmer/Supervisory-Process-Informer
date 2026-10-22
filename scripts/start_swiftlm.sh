#!/bin/bash
set -e

PORT=${SWIFTLM_PORT:-9432}
MODEL=${SWIFTLM_MODEL:-"Qwen/Qwen3.6-27B"}

echo "=================================================="
echo "    Launching SwiftLM Local Inference Server      "
echo "=================================================="
echo "Port:  $PORT"
echo "Model: $MODEL"
echo "URL:   http://localhost:$PORT/v1"
echo "=================================================="

SWIFTLM_BIN="./SwiftLM/.build/arm64-apple-macosx/release/SwiftLM"
if [ ! -f "$SWIFTLM_BIN" ]; then
    SWIFTLM_BIN="./SwiftLM/.build/release/SwiftLM"
fi

if [ -f "$SWIFTLM_BIN" ]; then
    exec "$SWIFTLM_BIN" --model "$MODEL" --host "127.0.0.1" --port "$PORT"
else
    echo "SwiftLM binary not found. Run SwiftLM build first."
    exit 1
fi
