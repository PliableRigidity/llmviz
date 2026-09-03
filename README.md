# LLMVis

**Live visualizer and educational debugger for local LLM inference.**

LLMVis sits beside your `ollama run` session in a second terminal and shows you what is happening while a model generates a response — system resources, model metadata, and an honest view of what can and cannot be observed from outside the inference engine.

It is inspired by `btop`, `htop`, and oscilloscopes: a tool that lets you *watch* a system work.

---

## Why

Running `ollama run qwen2.5:3b` gives you a chat interface but tells you nothing about the inference process. What is the model's quantization? How much VRAM is it using? Is it generating right now? How fast?

LLMVis answers these questions without modifying Ollama or replacing your workflow.

---

## Installation

```bash
git clone https://github.com/PliableRigidity/llmviz
cd llmviz
pip install -e .
```

With NVIDIA GPU telemetry (recommended if you have an NVIDIA GPU):

```bash
pip install -e ".[nvidia]"
```

On Apple Silicon (M1/M2/M3/M4), for MLX-based deep instrumentation:

```bash
pip install -e ".[mlx]"
```

For development:

```bash
pip install -e ".[dev,nvidia]"
```

**Requirements:** Python 3.10+, Ollama running locally.

---

## Platform support

| Platform | Observer | Hardware telemetry | Deep instrumentation |
|---|---|---|---|
| Linux + NVIDIA | Yes | CPU / RAM / GPU / VRAM | PyTorch / CUDA |
| Apple Silicon (M1–M4) | Yes | CPU / Unified Memory | MLX / Metal |
| CPU-only Linux | Yes | CPU / RAM | PyTorch / CPU |
| Windows + NVIDIA | Yes | CPU / RAM / GPU / VRAM | PyTorch / CUDA |

**Apple Silicon** uses unified memory — LLMVis shows a single memory pool
rather than separate RAM/VRAM figures. There is no VRAM panel on Apple Silicon
because there is no physically separate GPU memory pool.

### Inference backends

```
LLMVis
   │
   ├── OllamaObserverAdapter    (all platforms, stock Ollama)
   │
   ├── TransformersAdapter      (Linux / Windows / macOS)
   │      ├── CUDA              (NVIDIA)
   │      ├── MPS               (Apple Silicon, via PyTorch)
   │      └── CPU               (fallback)
   │
   └── MLXAdapter               (Apple Silicon only)
          └── Metal             (mlx-lm required)
```

Each backend exposes normalized `LLMVisEvent` objects — the TUI does not
need to know which runtime is in use. Backend capabilities (layer stats,
VRAM, unified memory, etc.) are declared via `BackendCapabilities` and
the TUI renders panels accordingly.

---

## Quick Start

```
Terminal 1:
  ollama run qwen2.5:3b

Terminal 2:
  llmvis run
```

LLMVis will automatically detect Ollama, find the loaded model, and begin monitoring.

### Other commands

```bash
llmvis status          # print current Ollama status and loaded models
llmvis doctor          # run diagnostic checks
llmvis run --host http://192.168.1.20:11434   # connect to a remote Ollama instance
llmvis --help
```

---

## TUI Screenshot (ASCII mockup)

```
┌─ LLMVIS  Ollama ● CONNECTED v0.33.2 │ qwen2.5:3b | Q4_K_M | ctx 4,096 ──────┐
│   http://localhost:11434                                                       │
└────────────────────────────────────────────────────────────────────────────────┘

┌─ MODEL ─────────────────────────────┐ ┌─ SYSTEM ─────────────────────────────┐
│                                     │ │                                      │
│  Model          qwen2.5:3b          │ │  CPU    ██████░░░░░░░░  42.0%  system│
│  Family         qwen2               │ │  RAM    ████████░░░░░░  55.3%        │
│  Parameters     3.1B                │ │         14.5 GB / 31.9 GB  measured  │
│  Quantization   Q4_K_M              │ │  Ollama CPU   1.2%  RAM  2.1 GB      │
│  Format         GGUF                │ │                                      │
│  Context        32,768 tokens       │ │  GPU   NVIDIA GeForce RTX 4090       │
│  Embedding      2,048 dims          │ │  GPU    ████████████░░  80.0% measured│
│  Disk size      1.8 GB              │ │  VRAM   ████░░░░░░░░░░  25.0% measured│
│  VRAM usage     1.7 GB  measured    │ │         4.3 GB / 16.0 GB             │
│  Capabilities   completion, tools   │ │  Temp   72°C                         │
└─────────────────────────────────────┘ └──────────────────────────────────────┘

┌─ INFERENCE ────────────────────────────────────────────────────────────────────┐
│                                                                                │
│  Status:  ACTIVITY DETECTED (estimated)                                        │
│  GPU utilization elevated — inference possibly active in another terminal.     │
│                                                                                │
│  GPU util  avg  62.4%  peak  88.1%  (estimated)                               │
│                                                                                │
│  Note: Stock Ollama does not expose another client's token stream.             │
│  Activity is inferred from GPU/CPU resource usage only.                        │
│                                                                                │
│  DEEP MODEL TELEMETRY — Unavailable with stock Ollama                          │
│  Future adapters can expose: layers · activations · logits · KV cache          │
└────────────────────────────────────────────────────────────────────────────────┘

┌─ MODEL PIPELINE ───────────────────────────────────────────────────────────────┐
│                                                                                │
│  Prompt → Tokenize → Prefill → Decode → Sample → Output                       │
│    │           │         │        │         │                                  │
│    text    token ids  KV cache  next tok  detokenize                           │
└────────────────────────────────────────────────────────────────────────────────┘

 q quit   c concepts   ? help   1 overview   2 system   esc close overlay
```

---

## What V1 Can Observe (Real Data)

All of these are read from Ollama's public HTTP API or system telemetry:

| Data | Source | Label |
|------|--------|-------|
| Ollama connection status | `/api/version` | measured |
| Ollama version | `/api/version` | measured |
| Loaded model name | `/api/ps` | measured |
| Model family | `/api/ps` → details | measured |
| Parameter count | `/api/ps` → details | measured |
| Quantization level | `/api/ps` → details | measured |
| Format (GGUF etc.) | `/api/ps` → details | measured |
| Context length (active) | `/api/ps` | measured |
| Embedding dimensions | `/api/tags` → details | measured |
| VRAM usage (model) | `/api/ps` → `size_vram` | measured |
| Disk size | `/api/tags` → `size` | measured |
| Model capabilities | `/api/tags` | measured |
| CPU usage (system-wide) | psutil | measured |
| RAM usage | psutil | measured |
| Ollama process CPU/RAM | psutil | measured |
| GPU utilization | pynvml (NVIDIA) | measured |
| VRAM utilization | pynvml (NVIDIA) | measured |
| GPU temperature | pynvml (NVIDIA) | measured |
| Inference activity | GPU util spike detection | **estimated** |
| Number of installed models | `/api/tags` | measured |

---

## What Stock Ollama Does NOT Expose

These are impossible to observe from a separate process using stock Ollama's public API:

- **Another client's token stream** — if you run `ollama run` in Terminal 1, the tokens being generated are not visible to an observer in Terminal 2 via any public endpoint.
- **Prefill vs. decode phase distinction** — not externally observable.
- **Per-token timing or speed** — only available in the response after generation completes if you make your own API call.
- **Token probabilities / logits** — not exposed.
- **Transformer layer activations** — not exposed.
- **Attention head statistics** — not exposed.
- **KV-cache contents or size** — not exposed (VRAM usage includes KV cache but is not broken down).
- **MoE expert routing** — not exposed.
- **Prompt token count or context usage** — for another client's session, not exposed.

LLMVis will never display fabricated versions of these. When data is unavailable, it says so.

---

## Architecture

```
┌─────────────────────────────────────────┐
│              LLMVis TUI                 │
│          (Textual framework)            │
└───────────────┬─────────────────────────┘
                │  AppState (read)
                │  refresh_state() calls
┌───────────────▼─────────────────────────┐
│              LLMVisApp                  │
│    (Event bus wiring, background tasks) │
└────────┬──────────────┬─────────────────┘
         │              │
┌────────▼──────┐  ┌────▼────────────┐
│  EventBus     │  │  AppState       │
│  (asyncio Q)  │  │  (single shared │
└────────┬──────┘  │   mutable obj)  │
         │         └─────────────────┘
    ┌────▼──────────────────────┐
    │   Normalized LLMVisEvents │
    │   (typed dataclasses)     │
    └────────────┬──────────────┘
         ┌───────┴───────┐
         │               │
┌────────▼──────┐  ┌─────▼──────────┐
│ OllamaAdapter │  │ SystemTelemetry│
│ (HTTP polling)│  │ + GPU telemetry│
│ /api/ps       │  │ (psutil/pynvml)│
│ /api/tags     │  └────────────────┘
│ /api/version  │
└───────────────┘
```

### Key design principles

1. **Adapter abstraction** — `BaseAdapter` defines the interface; `OllamaAdapter` implements it. Future adapters (llama.cpp hooks, PyTorch forward hooks) implement the same interface.
2. **Normalized events** — all backends emit the same typed event types. The TUI never needs to know which backend is in use.
3. **Future event types defined now** — `TOKEN_GENERATED`, `LAYER_ENTER`, `ACTIVATION_STATS`, etc. are defined as typed dataclasses even though V1 does not emit them. This ensures the frontend needs no rewrite when deeper backends arrive.
4. **Honest labeling** — every displayed value is labeled as "measured" or "estimated". Nothing is fabricated.

---

## Future Roadmap

### V2: Deeper Inference Instrumentation

The event system is already designed to support:

- **llama.cpp instrumented backend** — a custom runner that exposes per-token events, prefill/decode phases, KV-cache size, token probabilities.
- **PyTorch / Hugging Face forward hooks** — per-layer activations, attention head statistics, MLP neuron statistics, residual stream statistics.
- **Custom Ollama runner** — a wrapper that proxies Ollama while emitting detailed events.

When these backends are available, the TUI can display:
- Real-time token generation with timing
- Prefill progress bar (actual, not estimated)
- Per-layer execution timeline
- Attention head heatmaps
- Activation magnitude histograms
- KV-cache size over time
- Top candidate tokens and sampling decisions

### V3: Agent Visualization

The event model includes `AGENT_STEP`, `TOOL_CALL`, `TOOL_RESULT`, `CONTEXT_UPDATE` for future agent trace visualization.

### Platform Support

- AMD GPU telemetry (ROCm)
- Apple Silicon (Metal / MPS)
- Multiple GPU support

---

## Contributing

LLMVis is MIT licensed and open for contributions.

### Setup

```bash
git clone https://github.com/PliableRigidity/llmviz
cd llmviz
pip install -e ".[dev,nvidia]"
pytest tests/ -v
ruff check src/
```

### Adding a new backend adapter

1. Subclass `BaseAdapter` in `src/llmvis/adapters/`.
2. Implement `run()` as an async generator yielding `LLMVisEvent` instances.
3. Use only event types already defined in `core/events.py`. Add new event types there if needed.
4. Label all emitted data honestly — if a value is estimated, use an event type that makes this clear.

### Adding educational concepts

Add entries to `CONCEPTS` in `src/llmvis/educational/concepts.py`. Each entry needs a `name`, `short` (1-2 line label), and `full` (paragraph for the overlay).

### Philosophy

- Never display fabricated internal model data.
- Every displayed value must be either directly measured or clearly labeled as estimated.
- Keep the observer low-overhead — do not noticeably slow down inference.
- The architecture should support deeper instrumentation without rewriting the TUI.

---

## Note on Attention/Activation Visualization

Attention maps, layer activations, token probabilities, and KV-cache contents are **not available with stock Ollama**. These require instrumentation inside the inference engine itself — either via llama.cpp callback hooks, PyTorch forward hooks, or a custom Ollama runner.

LLMVis is designed so these backends can be added later. When they are, the TUI will display real data. Until then, these panels display an honest "unavailable" message rather than fabricated visuals.
