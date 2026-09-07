# LLMVis

**Real-time inference visualizer and educational debugger for local LLMs.**

Two modes. One tool.

| Mode | Terminal 1 | Terminal 2 | What you see |
|---|---|---|---|
| **Observer** (V1) | `ollama run <model>` | `llmvis run` | Ollama system stats, model metadata, resource usage |
| **Deep instrumentation** (V2) | `llmvis instrument <model>` | `llmvis run` | Per-token timing, layer activations, logits, KV cache — from actual model execution |

Everything displayed is either directly measured from the hardware/model or explicitly labelled as estimated. Nothing is fabricated.

---

## Installation

```bash
git clone https://github.com/PliableRigidity/llmviz
cd llmviz
pip install -e .
```

With NVIDIA GPU telemetry:

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

**Requirements:** Python 3.10+. Observer mode requires Ollama. Deep instrumentation requires PyTorch 2.0+ and a HuggingFace-compatible model.

---

## Quick Start

### Observer mode (stock Ollama)

```
Terminal 1:  ollama run qwen2.5:3b
Terminal 2:  llmvis run
```

LLMVis connects to Ollama, reads the loaded model's metadata, and shows live system telemetry.

### Deep instrumentation mode

```
Terminal 1:  llmvis instrument Qwen/Qwen2.5-1.5B-Instruct
Terminal 2:  llmvis run
```

Terminal 1 loads the model, starts a headless telemetry server on port 7654, and opens an interactive REPL for submitting prompts. Terminal 2 auto-discovers the server and opens the 6-tab TUI. Both terminals can be started in either order.

On Apple Silicon with MLX installed:

```bash
llmvis instrument meta-llama/Llama-3.2-1B --backend mlx
```

### Other commands

```bash
llmvis doctor                         # environment diagnostics
llmvis status                         # print Ollama status
llmvis replay recording.jsonl         # replay a saved telemetry session
llmvis replay recording.jsonl --speed 2.0
llmvis demo                           # play the built-in demo session
llmvis instrument <model> --record session.jsonl  # record while running
llmvis --help
```

---

## Deep Instrumentation — What You See

The `llmvis instrument` TUI has 6 tabs. Every value comes from actual tensor operations or hardware queries.

### 1  Overview
- Model: architecture info (layers, heads, hidden size, vocab size, dtype, device)
- System: CPU %, RAM usage, GPU utilization, VRAM, temperature

### 2  Transformer
Live view of all decoder layers as they execute, for each token:
- **Hidden-state RMS** — root mean square of the layer's output activation tensor
- **Layer execution time** (ms) — wall-clock time per forward pass
- **Delta from previous layer** — RMS difference from the preceding layer's output

These are numerical properties of the activations, not interpretations of "importance" or "attention to meaning." The bars show magnitude only.

### 3  Tokens
Scrollable history of all generated tokens with:
- Token text and ID
- Per-token latency (ms)
- KV cache sequence length at that step
- Sampler parameters used (temperature, top-k, top-p)

Use `←`/`→` to browse history, `L` to return to live.

### 4  Logits
Top vocabulary candidates before sampling, for the current/selected token:
- **Post-filter probability** — after temperature scaling, top-k, and top-p filtering
- **Pre-filter probability** — raw softmax before any filtering
- **Actual logit** — the raw LM-head output value

The pre-filter vs. post-filter distinction shows how much nucleus sampling narrows the distribution.

### 5  KV Cache
- Sequence length (tokens in cache)
- Actual tensor shapes (K and V tensors, per layer)
- Measured memory usage (computed from tensor byte sizes, not estimated)
- KV cache dtype

### 6  Performance
- Prefill time (ms) for the prompt
- Decode speed (tokens/sec) rolling average
- Per-token latency histogram

---

## Platform Support

| Platform | Observer | Deep instrumentation | Hardware telemetry |
|---|---|---|---|
| Linux + NVIDIA | Yes | PyTorch / CUDA | CPU / RAM / GPU / VRAM |
| Windows + NVIDIA | Yes | PyTorch / CUDA | CPU / RAM / GPU / VRAM |
| Apple Silicon (M1–M4) | Yes | MLX / Metal | CPU / Unified Memory |
| CPU-only Linux / Windows | Yes | PyTorch / CPU | CPU / RAM |

Apple Silicon uses a unified memory model — LLMVis shows a single memory pool rather than separate RAM/VRAM. The GPU VRAM panel does not appear on Apple Silicon because there is no separate GPU memory.

### Backend architecture

```
Terminal 1 (llmvis instrument)            Terminal 2 (llmvis run)
──────────────────────────────────        ──────────────────────────────
InstrumentedTransformersAdapter           ClientAdapter
  PyTorch forward hooks                     TCP JSONL connection
  CUDA / MPS / CPU                          auto-reconnects
        │                                        │
        ▼                                        ▼
  TelemetryServer ──── TCP port 7654 ──→ LLMVisDeepApp (6-tab TUI)
  localhost:7654
  broadcasts events,
  accepts prompt commands

── OR (Ollama observer mode) ──────────────────────────────────────────
Terminal 1: ollama run <model>            Terminal 2: llmvis run
                                            OllamaObserverAdapter
                                              HTTP polling
                                            LLMVisApp (2-tab TUI)
```

All adapters emit the same `LLMVisEvent` dataclasses over the EventBus. `llmvis run` auto-discovers the best available source: deep session first, then Ollama, then shows a waiting screen.

---

## Observer Mode — What Is Measured

All values come from Ollama's public HTTP API or system telemetry:

| Data | Source | Label |
|---|---|---|
| Ollama version / status | `/api/version` | measured |
| Loaded model name, family, quantization | `/api/ps` | measured |
| Context length (active), embedding dims | `/api/ps`, `/api/tags` | measured |
| VRAM usage (model), disk size | `/api/ps`, `/api/tags` | measured |
| CPU usage (system + Ollama process) | psutil | measured |
| RAM usage | psutil | measured |
| GPU utilization, VRAM, temperature | pynvml (NVIDIA) | measured |
| Inference activity | GPU util spike detection | **estimated** |

### What stock Ollama does NOT expose

These are structurally unavailable from a separate observer process:

- Another client's token stream (generation in `ollama run` Terminal 1 is not visible in Terminal 2 via any public API)
- Prefill vs. decode phase
- Per-token speed or timing
- Token probabilities / logits
- Transformer layer activations
- KV-cache contents or size breakdown

LLMVis does not display fabricated versions of these in observer mode. When data is unavailable, it says so.

---

## Record and Replay

Record a deep instrumentation session for later replay or sharing:

```bash
llmvis instrument meta-llama/Llama-3.2-1B --record session.jsonl
```

Every event is written to `session.jsonl` (JSONL format, one event per line). No raw tensors — only scalar statistics.

Replay at original speed or faster:

```bash
llmvis replay session.jsonl
llmvis replay session.jsonl --speed 4.0
llmvis replay session.jsonl --loop
```

The `llmvis demo` command plays the built-in demo session (`src/llmvis/demo/demo_session.jsonl`) recorded from a small synthetic model.

---

## Deep Mode Keybindings

| Key | Action |
|---|---|
| `1` – `6` | Switch tabs |
| `←` / `→` | Browse token history |
| `L` | Return to live (latest token) |
| `Space` | Pause/resume visualization (inference continues) |
| `Enter` | Submit prompt |
| `?` | Keyboard help + educational notes |
| `Q` | Quit |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   LLMVis TUI                            │
│           (Textual, 6-tab or 2-tab layout)              │
└─────────────────────────┬───────────────────────────────┘
                          │  AppState + DeepState (shared)
                          │
┌─────────────────────────▼───────────────────────────────┐
│                    LLMVisApp / LLMVisDeepApp             │
│          Event bus wiring, async task management         │
└──────────┬─────────────────────────┬────────────────────┘
           │                         │
┌──────────▼──────────┐   ┌──────────▼──────────────────┐
│      EventBus       │   │         AppState             │
│  asyncio.Queue      │   │   (single mutable object,    │
│  maxsize=2000,      │   │    read by all TUI panels)   │
│  drop-on-full       │   └──────────────────────────────┘
└──────────┬──────────┘
           │  typed LLMVisEvent dataclasses
           │
    ┌──────┴──────────────────────────┐
    │                                 │
┌───▼───────────────────┐  ┌──────────▼──────────────┐
│  OllamaObserverAdapter│  │  InstrumentedTransformers│
│  HTTP polling         │  │  Adapter / MLXAdapter    │
│  /api/ps, /api/tags   │  │  PyTorch forward hooks   │
└───────────────────────┘  └─────────────────────────┘
```

### Key design principles

1. **Adapter abstraction** — `BaseAdapter.run()` is an async generator of `LLMVisEvent`. New backends implement this interface; the TUI needs no changes.
2. **No fabricated data** — every value must come from actual model execution, hardware telemetry, or the public Ollama API. Values that are estimated are labelled as estimated.
3. **Bounded event bus** — `asyncio.Queue(maxsize=2000)` with drop-on-full prevents OOM on fast GPU decode.
4. **Honest labeling** — attention bars show numerical RMS magnitude. Token probability bars show post-filter sampling probability. The educational overlay explains what each value actually is.

---

## Testing

```bash
pytest tests/ -v           # 228 tests
pytest tests/ -x -q        # fail-fast
```

Test coverage includes: event serialization round-trips, bounded queue drop behavior, platform detection, KV cache statistics, logit sampling, adapter integration with a synthetic model, MLX top-p, event ordering sequences, demo JSONL validity.

---

## Contributing

LLMVis is MIT licensed.

### Setup

```bash
git clone https://github.com/PliableRigidity/llmviz
cd llmviz
pip install -e ".[dev,nvidia]"
pytest tests/ -v
```

### Adding a backend

1. Subclass `BaseAdapter` in `src/llmvis/adapters/`.
2. Implement `run()` as an async generator yielding `LLMVisEvent` instances.
3. Use event types from `core/events.py`; add new types there if needed.
4. Declare `BackendCapabilities` so the TUI can hide unavailable panels.
5. Label all emitted data honestly.

### Adding educational content

Add entries to `CONCEPTS` in `src/llmvis/educational/concepts.py`. Each needs `name`, `short`, and `full` fields.

### Philosophy

- Never display fabricated internal model data.
- Every displayed value is directly measured or clearly labelled as estimated.
- Keep the observer low-overhead — do not noticeably slow down inference.
- New backends must not require TUI rewrites.
