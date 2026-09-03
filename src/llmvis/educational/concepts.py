"""Educational concept descriptions for LLMVis.

These are shown when the user highlights a concept in the TUI.
Keep descriptions terminal-friendly: short paragraphs, no markdown headers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Concept:
    name: str
    short: str       # 1-2 lines for inline labels
    full: str        # Paragraph for the detail overlay


CONCEPTS: dict[str, Concept] = {
    "context_window": Concept(
        name="Context Window",
        short="Tokens the model can currently attend to.",
        full=(
            "The context window is the total sequence of tokens the model "
            "can see and attend to during a single forward pass. It includes "
            "the system prompt, conversation history, and any pending output. "
            "Tokens beyond the context limit are not visible to the model — "
            "they are simply not there, not summarised. A larger context window "
            "generally requires more memory (especially KV cache) and increases "
            "compute per token."
        ),
    ),
    "prefill": Concept(
        name="Prefill",
        short="Processing the input prompt before generation begins.",
        full=(
            "Before generating any output, the model processes your entire "
            "input prompt in parallel — this is called the prefill phase. "
            "During prefill, the model computes attention over all input tokens "
            "simultaneously, which is fast but memory-intensive. The resulting "
            "key-value (KV) tensors are cached so they don't need recomputation "
            "during decoding. Prefill speed is measured in prompt tokens per second "
            "and is typically much faster than decode speed."
        ),
    ),
    "decode": Concept(
        name="Decode / Generation",
        short="Token-by-token autoregressive output generation.",
        full=(
            "After prefill, the model generates output one token at a time. "
            "Each new token is produced by running a forward pass over the "
            "entire context (using cached KV tensors for prior tokens) and "
            "sampling from the resulting probability distribution. This "
            "autoregressive process is the bottleneck for output speed. "
            "Decode speed (tokens/sec) is what most users experience as "
            "\"how fast the model types\"."
        ),
    ),
    "kv_cache": Concept(
        name="KV Cache",
        short="Cached attention keys & values — avoids recomputing prior tokens.",
        full=(
            "Transformer attention computes keys (K) and values (V) for every "
            "token in the sequence. During decoding, these tensors for already-seen "
            "tokens can be stored and reused, rather than recomputed from scratch "
            "on every new token generation step. This is the KV cache. "
            "Without it, each decode step would be O(n²) in sequence length. "
            "With it, attention over prior tokens is O(1) per step (just a lookup). "
            "The tradeoff is memory: the KV cache grows linearly with context length "
            "and can dominate VRAM usage at long sequences. "
            "Important: the KV cache is NOT long-term memory — it only spans "
            "the current context window and is discarded when the session ends."
        ),
    ),
    "tokens_per_sec": Concept(
        name="Tokens / Second",
        short="Output generation speed (decode phase only).",
        full=(
            "Tokens per second measures how quickly the model produces output "
            "tokens during the decode phase. It is calculated as: "
            "output_tokens / decode_time. This is distinct from prefill speed, "
            "which measures how fast the input prompt is processed. "
            "A typical small model (3-8B) on consumer hardware might achieve "
            "10-80 tok/s, while larger models or CPU-only inference can be "
            "much slower. The number depends on model size, quantization, "
            "hardware, and context length."
        ),
    ),
    "quantization": Concept(
        name="Quantization",
        short="Reduced-precision weights for smaller memory footprint.",
        full=(
            "Quantization reduces model weight precision from the original "
            "training format (typically float32 or bfloat16) to fewer bits. "
            "Common formats: Q4_K_M (4-bit, medium quality), Q8_0 (8-bit, "
            "near-lossless), F16 (16-bit, full precision). "
            "Lower bit-width means smaller model files and less VRAM usage, "
            "at a small cost to output quality. "
            "The K_M suffix in formats like Q4_K_M refers to the specific "
            "quantization scheme (k-quants with mixed precision for some layers)."
        ),
    ),
    "parameters": Concept(
        name="Parameters",
        short="Trainable weights that define the model's learned knowledge.",
        full=(
            "A language model's parameters are the numerical weights learned "
            "during training. Parameter count is often used as a proxy for model "
            "capability. A 3B model has ~3 billion weights; a 70B model has ~70 billion. "
            "More parameters generally mean better performance, but also more memory "
            "and slower inference. At 4-bit quantization, a rough rule of thumb: "
            "1B parameters ≈ 0.5 GB of VRAM."
        ),
    ),
    "vram": Concept(
        name="VRAM",
        short="GPU video memory — where model weights and KV cache live.",
        full=(
            "VRAM (video RAM) is the memory on your GPU. During inference, "
            "VRAM holds the model weights, the KV cache, and intermediate "
            "activations. If a model exceeds available VRAM, Ollama may "
            "offload layers to RAM (slower) or refuse to load. "
            "VRAM usage = model weights + KV cache (grows with context) + "
            "small overhead for activations."
        ),
    ),
    "model_family": Concept(
        name="Model Family",
        short="The architecture and training lineage of a model.",
        full=(
            "Model family refers to the underlying architecture. Examples: "
            "llama (Meta's LLaMA architecture), qwen2 (Alibaba's Qwen), "
            "gemma (Google's Gemma), phi3 (Microsoft's Phi). "
            "Models in the same family share the same transformer architecture "
            "but differ in size, training data, and fine-tuning."
        ),
    ),
    "deep_telemetry": Concept(
        name="Deep Telemetry (Unavailable)",
        short="Per-layer data unavailable with stock Ollama.",
        full=(
            "Deep model telemetry — transformer layers, attention head statistics, "
            "activation magnitudes, logits, token probabilities, KV-cache contents, "
            "MoE expert routing — requires instrumentation inside the inference engine. "
            "Stock Ollama does not expose these. Future LLMVis versions will support "
            "instrumented backends (llama.cpp hooks, PyTorch forward hooks, custom runners). "
            "LLMVis will never display fabricated internal data."
        ),
    ),
}


def get_concept(key: str) -> Concept | None:
    return CONCEPTS.get(key)


def all_concept_keys() -> list[str]:
    return list(CONCEPTS.keys())
