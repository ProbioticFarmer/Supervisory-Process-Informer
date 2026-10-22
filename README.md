# Supervisory Process Informer (SPI)
### Deterministic State-Space Telemetry & Sub-Second Trajectory Surgery for Hybrid Reasoning LLMs

> *"You cannot engineer a dynamical system you refuse to measure."*

**Supervisory Process Informer (SPI)** is a white-box systems engineering framework and interactive oscilloscope for observing, diagnosing, and surgically steering chain-of-thought (`<think>`) trajectories in hybrid linear/full-attention transformers (`Qwen3.8-27B`). 

Instead of treating language models as opaque black boxes and throwing away 500-token reasoning rollouts because of a single divergence at token $t^*$, SPI exposes the model's internal attention routing geometry and recurrent state space in real time. Engineers—and automated supervisory policies—can pinpoint the exact token where reasoning drifts, rewind the hybrid cache in under a second, repair the local causal anchor, and synthesize counterfactual fine-tuning datasets (`spi_traces.jsonl`) with zero wasted compute.

---

## 1. Discussion: Moving Beyond Stochastic "Slot-Machine" Prompting

Much of the current industry tooling around "prompt compilation" and "automated prompt optimization"—most notably frameworks like **DSPy** (`BootstrapFewShotWithRandomSearch`, `MIPRO`)—markets itself as a PyTorch-like optimization paradigm. Yet a brief inspection of the underlying teleprompter source code reveals a sobering reality:

1. **Random Permutations Masquerading as "Compilation":** Standard "optimizers" instantiate `random.Random(seed)`, shuffle few-shot slices, perturb decoding temperatures (`0.7 + 0.001 * i`), and hope a stochastic rollout happens to pass a binary end-of-sequence metric by accident.
2. **Surrogate Optimization Over Metric-Free Integer IDs:** Passing disconnected categorical indices (`instruction_idx #3` vs. `demo_set_idx #7`) into a Tree-structured Parzen Estimator (TPE) violates the foundational assumption of Bayesian optimization: **there is no continuous metric topology over arbitrary integer labels.**
3. **Total Rejection of Near-Miss Trajectories:** When a 500-token coding or analytical rollout executes 95% of a proof flawlessly but misbinds a single variable invariant at token 310, black-box rejection sampling discards the entire rollout and pulls the slot-machine lever again.

### The SPI Premise: Deterministic Prefill Topology
While batched autoregressive decoding introduces floating-point variance, **the forward prefill pass over any fixed token prefix $x_{1:t}$ is a strictly deterministic mathematical mapping.** 

When a reasoning model fails inside `<think>`, it does not fail uniformly across all tokens. It diverges at a localized **bifurcation token $t^*$** where either:
* Its **specification-grounding heads** lose lock on the user's prompt constraints, or
* Its **Preplan-and-Anchor rhythm** crystallizes a faulty intermediate assumption (an **Anchor**) that all subsequent steps attend back to.

SPI replaces blind prefix guessing with three deterministic systems-engineering primitives: **Observe** (read the internal attention rhythm), **Locate $t^*$** (identify the exact Preplan $\to$ Anchor bifurcation token), and **Intervene** (rewind the hybrid state space to $t^*$ and repair the trajectory in place).

---

## 2. Core Mathematical & Architectural Mechanics

`Qwen3.8-27B` utilizes a 64-layer hybrid **3:1 architecture**: **48 Gated DeltaNet** linear-attention layers interleaved with **16 Gated Full-Attention** layers (`L3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63`). Building a real-time time machine and telemetry probe across this hybrid state space requires solving three distinct mathematical problems:

### A. `HybridTimelineCache`: Solving the Non-Invertible DeltaNet State Space
In standard full-attention layers, rewinding from token $N$ back to token $t^* < N$ is trivial: slice the `KVCache` tensors along the sequence dimension (`K[..., :t*, :]`). 

However, the 48 Gated DeltaNet layers maintain a fixed-size recurrent fast-weight matrix $S_t \in \mathbb{R}^{H \times d_k \times d_v}$ updated via a non-invertible gated Householder-style contraction:
$$S_t = S_{t-1}\left(\alpha_t (I - \beta_t k_t k_t^\top)\right) + \beta_t v_t k_t^\top$$
Because rank-1 erasure and decay destroy historical information, $S_{t^*}$ cannot be recovered algebraically from $S_N$. 

**The SPI Solution (`hybrid_cache.py`):**
* **Exact KV Slicing:** The 16 full-attention `KVCache` layers are sliced in $O(1)$ time to `[:t*]`.
* **Stride-8 Semigroup Snapshotting + Delta Replay:** During prefill and organic decoding, SPI captures deep copies of the 48 `ArraysCache` fast-weight states every $k = 8$ tokens (`t_snap = 8 * floor(t* / 8)`). Rewinding to any arbitrary token $t^*$ restores the nearest snapshot $S_{t_{\text{snap}}}$ and replays at most $0\text{..}7$ tokens through the forward semigroup—achieving **~570–925 ms arbitrary token rewind** across a 27B parameter model in 8-bit MLX on Apple Silicon (M4 Max, 128GB unified memory).

---

### B. 4-Mode Task-Agnostic Rhythm Telemetry (`attention_probe.py`)
SPI extracts the $16 \times 24 = 384$ full-attention routing matrices $A^{(l,h)} \in \mathbb{R}^{N \times N}$ during an unfused probe pass and computes four orthogonal structural signals for every token inside `<think>`:

#### 1. Prompt-Lookback Score (`lookback_score`)
Measures the exact attention mass directed from `<think>` token $t \ge t_{\text{think}}$ back into the user prompt specification span $[0 \dots t_{\text{think}} - 1]$ across the 5 primary mid-to-late routing monitor heads (`L35.H21`, `L39.H1`, `L43.H7`, `L43.H1`, `L39.H0`):
$$\text{Lookback}(t) = \frac{1}{\vert{}\mathcal{M}\vert{}} \sum_{(l,h) \in \mathcal{M}} \sum_{s=0}^{t_{\text{think}}-1} A^{(l,h)}_{t,s}$$

#### 2. Sink-Free Intra-`<think>` Submatrix & Head Partitioning
Following and extending Li et al. (*"Attention Illuminates LLM Reasoning: The Preplan-and-Anchor Rhythm"*, ICML 2026), SPI eliminates BOS/system attention-sink distortion (`s = 0..3`) by extracting the $M \times M$ causal `<think>` submatrix ($M = N - t_{\text{think}}$) and row-normalizing over $j \le i$:
$$\tilde{A}^{(l,h)}_{i,j} = \frac{A^{(l,h)}_{t_{\text{think}}+i, \, t_{\text{think}}+j}}{\sum_{k=0}^{i} A^{(l,h)}_{t_{\text{think}}+i, \, t_{\text{think}}+k} + \epsilon}$$
Each of the 384 heads is classified by its mean backward attention distance $d^{(l,h)} = \frac{1}{M}\sum_{i=0}^{M-1}\sum_{j=0}^{i} \tilde{A}^{(l,h)}_{i,j}(i - j)$, partitioning at the median into **192 Locally Focused Heads ($\mathcal{H}_{\text{loc}} \to \bar{A}^{\text{loc}}$)** and **192 Globally Focused Heads ($\mathcal{H}_{\text{glob}} \to \bar{A}^{\text{glob}}$)**.

#### 3. Capacity-Normalized Preplan Score ($\text{WAAD}_i$ & $\vert{}\Delta \text{WAAD}_i\vert{}$)
At the onset of a new reasoning step (e.g., `"Wait—"`, `"Let me think"`, `"However,"`), locally focused heads temporarily reach backward across context to gather premises before planning the next clause. SPI computes Windowed Average Attention Distance ($W = 32$) normalized by effective window capacity and a $\pm 16$-token rolling local z-score:
$$\widetilde{\text{WAAD}}_i = \frac{1}{\max(1.0, \min(i, W)/2.0)} \sum_{j=0}^{i} \bar{A}^{\text{loc}}_{i,j} \min(i - j, W)$$
$$\text{Preplan}_i = \text{Norm}_{[0,1]}\Big(z_{\text{local}}(\widetilde{\text{WAAD}}_i) + z_{\text{local}}\big(\vert{}\widetilde{\text{WAAD}}_i - \widetilde{\text{WAAD}}_{\min(i+1, M-1)}\vert{}\big)\Big)$$

#### 4. Scale-Invariant, Horizon-Guarded Anchor Score ($\text{FAI}_j$)
Once a Preplan onset opens a clause, the model crystallizes a load-bearing semantic **Anchor** token $j$ that downstream steps $i \in T(j) = [j + H_{\text{lo}}, \min(M - 1, j + H_{\text{hi}})]$ ($H_{\text{lo}} = 4, H_{\text{hi}} = 48$) repeatedly attend back to via global heads. To eliminate $1/i$ softmax sequence-length dilution and tail-recency inflation (`min_horizon = 6`), SPI computes:
$$\text{FAI}_j = \min\left(1.0, \frac{\vert{}T(j)\vert{}}{6}\right) \cdot \frac{1}{\vert{}T(j)\vert{}} \sum_{i \in T(j)} (i + 1) \bar{A}^{\text{glob}}_{i,j}$$
Coupled **`Preplan -> Anchor` Junctures (`P→` / `→A`)** are automatically flagged whenever a local Preplan maximum ($\text{Preplan}_p \ge 0.60$) is followed within $1\text{..}6$ tokens by a high-influence Anchor hub ($\text{Anchor}_a \ge 0.55$).

---

### C. Think-Gated Subspace Projection & The DeltaNet Subspace Flywheel (`subspace_projector.py`)
Alongside token-level trajectory surgery (`fork_and_edit`), SPI supports continuous rank-1 orthogonal subspace projection across layers $l \in [10..58]$:
$$h'_l = \Pi_{\hat{u}_l^\perp}(h_l) = h_l - \alpha (h_l \cdot \hat{u}_l)\hat{u}_l$$
* **The DeltaNet Subspace Flywheel:** Causal 4-band ablation proves that in a 3:1 hybrid architecture, mid-to-late full-attention routing heads (`L35`, `L39`, `L43`) cannot be steered by late-layer intervention alone. Early-to-mid Gated DeltaNet layers (`L10..29`) continuously write subspace features into the recurrent fast-weight memory $S_t$, which then feeds the $Q/K$ projections of `L35..L43`.
* **Orthogonal Division of Labor (`2x2` Intervention Matrix):** Token-level `<think>` edits primarily steer semantic context-tracking heads (`L35.H21`, `L43.H7`), while subspace projection ($\alpha = 1.0$) silences dedicated audit/refusal heads (`L39.H1`, `L39.H0`, `L43.H1`).
* **Think-Gated Release ($\alpha \to 0$ at `</think>`):** When `think_gated=True`, SPI applies subspace projection strictly while the model deliberates inside `<think>...</think>`, and automatically restores **100% unperturbed stock weights ($\alpha = 0.0$)** at the exact token index where `</think>` (`151668`) is emitted—preserving native downstream code and prose generation fidelity.

---

## 3. Visual Walkthrough: Reading the SPI Oscilloscope

*(Screenshots below captured live from `harness_server.py` running `Qwen3.8-27B-8bit` on `http://127.0.0.1:8000`)*

### Step 1: Watching a Rollout Detach from the Prompt (Mode 1: `Prompt Lookback`)
| Early `<think>` (`Tokens [55..89]`) — High Specification Lock | Mid `<think>` (`Tokens [206..243]`) — Internal Echo Chamber |
|:---:|:---:|
| ![Mode 1 Early Lookback](docs/images/image_31307f.jpg) | ![Mode 1 Late Echo Chamber](docs/images/image_3130fd.png) |

* **Left (`Tokens [55..89]`):** Bright red/orange chips indicate high `Prompt Lookback` (`0.70–0.93`). The 5 monitor heads are physically reading the user's technical specification word-for-word.
* **Right (`Tokens [206..243]`):** After pivoting at token `This [90]`, the token stream turns dark green (`Lookback < 0.20`). The model has stopped attending to the user's prompt and is reciting memorized policy boilerplate inside an internal echo chamber, flashing orange only when opening quotation marks (`" [240]`, `WAAD = 19.1t`) to cherry-pick isolated words from the prompt.

### Step 2: Isolating the Single-Token Causal Hub (Mode 3: `Anchor Influence / FAI`)
![Mode 3 Anchor Influence](docs/images/image_31a5b9.png)
Switching the heatmap bar to **`3. Anchor Influence (FAI Hubs)`** reveals how global attention heads compress an entire clause into a single retrieval hub: out of the entire sentence `"This [90] is [91] essentially [92] asking [93] for [94] a [95] guide [96] on [97] how [98] to [99] make [100] malware [101] undetectable..."`, tokens `90..100` are dark purple (zero future influence), while **`malware [101]`** glows bright gold as the single load-bearing Anchor that poisons the next 150 tokens of reasoning.

### Step 3: Automated Bifurcation Detection (Mode 4: `Coupled Preplan->Anchor Junctures`)
| Early Bifurcation Bridge (`[95] P→` to `[101] →A`) | Downstream Audit Junctures (`[137] P→` to `[141] →A`) |
|:---:|:---:|
| ![Mode 4 Coupled Juncture Early](docs/images/image_32abc3.jpg) | ![Mode 4 Coupled Juncture Mid](docs/images/image_32ac02.png) |

Switching to **`4. Coupled Preplan->Anchor Junctures`** automatically tags the exact structural bridges where a Cyan **Preplan onset (`P→`)** crystallizes a Gold **Anchor hub (`→A`)**:
* `a [95] [P→] guide [96] on [97] how [98] to [99] make [100] malware [101] [→A]`
* `R [137] [P→] ) [138] and [139] execute [140] malicious [141] [→A]`
* `malicious [173] [P→] code [174] invisible [175] to [176] defensive [177] [→A]`

### Step 4: Sub-Second `fork_and_edit` & Counterfactual Trajectory Logging
![Stateful Fork and Edit](docs/images/image_31b3a8.jpg)
Clicking chip **`This [90]`** (or **`a [95]`**) sets `fork_token_idx = 90`. Clicking **Fork & Edit from Token Index** restores the stride-8 DeltaNet snapshot (`snap=87, +3t`) and warm-prefills the repaired analytical Anchor in **`925.78 ms`** (`JSD = 0.3231`), while automatically logging the shared prefix $x_{1:89}$, the rejected branch $y_{\text{fail}}$, the repaired branch $y_{\text{repaired}}$, and the full attention telemetry deltas to `outputs/spi_traces.jsonl`.

---

## 4. Repository Structure & Quickstart

```text
qwen38_sandbox/
├── attention_probe.py           # Unfused 16-layer attention capture + Preplan (WAAD) & Anchor (FAI) math
├── hybrid_cache.py              # HybridTimelineCache (16 KVCache + 48 Stride-8 ArraysCache snapshots)
├── subspace_projector.py        # WeightSpaceProjector + Think-Gated Activation-Space Hook
├── harness_server.py            # FastAPI Stateful Workbench UI + REST API + SPITrajectoryLogger
├── benchmark_and_experiment.py  # 2x2 intervention & 4-band causal ablation benchmark suite
├── test_sandbox_unit.py         # End-to-end unit & regression test suite
└── outputs/
    └── spi_traces.jsonl         # Automatically logged counterfactual (y_win, y_lose) trajectory pairs
```

### Launching the SPI Workbench
```bash
# 1. Run the verification test suite
./qwen38_sandbox/.venv/bin/python qwen38_sandbox/test_sandbox_unit.py

# 2. Start the live SPI Workbench server on http://127.0.0.1:8000
./qwen38_sandbox/.venv/bin/python qwen38_sandbox/harness_server.py
```

### Dual-Mode API (Human UI + Automated Closed-Loop SPI Agents)
Because the human Workbench UI and automated Python repair agents share the exact same stateful FastAPI backend, any automated script can drive SPI via four endpoints:
* `POST /api/prefill_and_probe`: Prefills a prompt (or organically decodes `generate_organic_tokens` inside `<think>`), populates stride-8 `HybridTimelineCache` snapshots, and returns per-token `lookback_score`, `preplan_score`, `anchor_score`, and `coupled_junctures`.
* `POST /api/fork_and_edit`: Rewinds the hybrid cache to `fork_token_idx` in $<1\text{s}$, injects `edited_text` under optional `alpha` (`think_gated=True`), decodes the continuation with live telemetry chips, and logs the counterfactual record to `spi_traces.jsonl`.
* `POST /api/promote_to_baseline`: Promotes any forked trajectory to the active baseline for multi-hop tree surgery.
* `POST /api/spi_traces/label`: Marks a verified trajectory repair as `"accepted"` (`★ Mark as Gold SPI Pair`) for direct SFT / DPO / process-supervision training.
