# 🤖 AGENT OPERATIONAL GUIDE: Pi Agent & SwiftLM Setup

**Target Reader**: Future AI coding agents (Antigravity, Gemini, Pi Agent, or CLI subagents) operating in this repository.

**Repository Location**: `/Users/joshua/projects/Pi_SynDE`

---

## 📐 1. System Architecture & Component Mapping

```
                                +-----------------------------------+
                                |          USER INTERFACE           |
                                |  Interactive CLI / Script Runner  |
                                +-----------------+-----------------+
                                                  |
                                                  v
                                +-----------------------------------+
                                |    Pi Coding Agent (Local Build)  |
                                |  ./pi/packages/coding-agent       |
                                +--------+-----------------+--------+
                                         |                 |
                   +---------------------+                 +---------------------+
                   |                                                             |
                   v                                                             v
    +------------------------------+                              +------------------------------+
    |   DeepSeek API (Cloud)       |                              |   SwiftLM Server (Local)     |
    |   https://api.deepseek.com   |                              |   http://localhost:9432      |
    |   Models:                    |                              |   Models:                    |
    |   - deepseek-chat (V3)       |                              |   - Qwen/Qwen3.6-27B         |
    |   - deepseek-reasoner (R1)   |                              |     (Native Apple Silicon)   |
    +------------------------------+                              +------------------------------+
```

---

## 📁 2. File & Directory Layout

| File / Path | Purpose & Instructions |
| :--- | :--- |
| [`package.json`](file:///Users/joshua/projects/Pi_SynDE/package.json) | Root project manifest. Defines npm scripts `npm run pi` and `npm run start:swiftlm`. |
| [`.pi/models.json`](file:///Users/joshua/projects/Pi_SynDE/.pi/models.json) | Local Pi Agent provider definitions for `deepseek` and `swiftlm`. |
| [`~/.pi/agent/models.json`](file:///Users/joshua/.pi/agent/models.json) | Global Pi Agent provider definitions (mirrors local configuration). |
| [`scripts/start_swiftlm.sh`](file:///Users/joshua/projects/Pi_SynDE/scripts/start_swiftlm.sh) | Executable bash script to launch SwiftLM server serving `Qwen/Qwen3.6-27B` on port `9432`. |
| [`.env.example`](file:///Users/joshua/projects/Pi_SynDE/.env.example) | Environment template containing `DEEPSEEK_API_KEY`, `SWIFTLM_PORT=9432`, `SWIFTLM_MODEL`. |
| [`SwiftLM/`](file:///Users/joshua/projects/Pi_SynDE/SwiftLM) | Native Apple Silicon inference engine (`.build/release/SwiftLM`). |
| [`pi/`](file:///Users/joshua/projects/Pi_SynDE/pi) | Core Pi Agent monorepo repository. |

---

## ⚡ 3. Operational Protocols for Future Agents

### A. Environment Verification
Before invoking models or running agent sessions, verify that the environment variables are active:
```bash
# Check if DeepSeek API Key is present
echo $DEEPSEEK_API_KEY
```
If missing, read or load from `.env` in project root.

### B. Launching & Managing SwiftLM Local Server
To start serving `Qwen/Qwen3.6-27B` on port `9432`:
```bash
# From workspace root (/Users/joshua/projects/Pi_SynDE)
npm run start:swiftlm
```
Or directly execute:
```bash
./scripts/start_swiftlm.sh
```
*   **Health Check**: Test server status via `curl http://localhost:9432/v1/models` or `curl http://localhost:9432/health`.
*   **Binary Location**: `SwiftLM/.build/release/SwiftLM`.

### C. Launching & Operating Pi Coding Agent
To launch Pi Agent in interactive session:
```bash
npm run pi
```
Or directly run the compiled CLI:
```bash
node ./pi/packages/coding-agent/dist/cli.js
```

#### Switching Models in Pi Agent
- **DeepSeek V3**:
  ```bash
  node ./pi/packages/coding-agent/dist/cli.js --model deepseek/deepseek-chat
  ```
- **DeepSeek R1 Reasoning**:
  ```bash
  node ./pi/packages/coding-agent/dist/cli.js --model deepseek/deepseek-reasoner
  ```
- **SwiftLM Local Qwen 2.5 27B**:
  ```bash
  node ./pi/packages/coding-agent/dist/cli.js --model swiftlm/qwen-27b
  # or
  node ./pi/packages/coding-agent/dist/cli.js --model swiftlm/Qwen/Qwen2.5-27B-Instruct
  ```

---

## ⚙️ 4. Model Provider Schema Reference

If you need to add or update model providers, edit [`.pi/models.json`](file:///Users/joshua/projects/Pi_SynDE/.pi/models.json) using this format:

```json
{
  "providers": {
    "deepseek": {
      "baseUrl": "https://api.deepseek.com/v1",
      "api": "openai-completions",
      "apiKey": "${DEEPSEEK_API_KEY}",
      "models": [
        { "id": "deepseek-chat", "name": "DeepSeek V3", "contextWindow": 64000 },
        { "id": "deepseek-reasoner", "name": "DeepSeek R1", "contextWindow": 64000 }
      ]
    },
    "swiftlm": {
      "baseUrl": "http://localhost:9432/v1",
      "api": "openai-completions",
      "apiKey": "swiftlm-local",
      "models": [
        { "id": "Qwen/Qwen3.6-27B", "name": "Qwen 3.6 27B (SwiftLM Local)", "contextWindow": 32768 }
      ]
    }
  }
}
```

---

## 🛠️ 5. Troubleshooting & Edge Cases

1. **`EPERM: operation not permitted, mkdir '~/.pi/agent'`**:
   - Ensure the execution environment has read/write permissions for `/Users/joshua/.pi`.
2. **SwiftLM Rebuild Needed**:
   - Run `swift build -c release` inside `./SwiftLM`.
   - If compiling custom Metal kernels via `build.sh` fails with missing Metal toolchain, run `xcodebuild -downloadComponent MetalToolchain`.
3. **Pi Monorepo Rebuild**:
   - If TypeScript changes are made in `./pi`, run `npm run build` inside `./pi` (note: script uses `npx tsx` for generator scripts).

---
*Created on 2026-08-02 for AI Agent system continuity.*
