"""InstrumentedTransformersAdapter — V2 core adapter.

Instruments a local HuggingFace Transformers model with PyTorch forward hooks
to emit real runtime events during generation. Uses a manual generation loop
(not model.generate()) for full control over prefill vs decode phases.

Telemetry modes:
  light    — timing, logits, KV cache stats only
  standard — + per-layer hidden-state statistics
  deep     — + MLP sparsity and attention entropy per layer
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from llmvis.adapters.base import BaseAdapter
from llmvis.core.events import (
    AttentionStatsEvent,
    DecodeStartEvent,
    InferenceEndEvent,
    InferenceStartEvent,
    KvCacheUpdateEvent,
    LayerStatsEvent,
    LLMVisEvent,
    LogitsReadyEvent,
    MLPStatsEvent,
    ModelArchitectureEvent,
    PrefillEndEvent,
    PrefillStartEvent,
    PromptReceivedEvent,
    SamplerConfig,
    TokenCandidate,
    TokenEndEvent,
    TokenGeneratedEvent,
    TokenSampledEvent,
    TokenStartEvent,
    TokenizationCompleteEvent,
)

logger = logging.getLogger(__name__)

# Sentinel used to signal the generation thread to stop
_STOP_SENTINEL: str | None = None


@dataclass
class _LayerHookData:
    """Data collected by the pre- and post-forward hooks for one layer during one token."""
    layer_index: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    input_rms: float = 0.0
    input_mean: float = 0.0
    input_std: float = 0.0
    output_rms: float = 0.0
    output_mean: float = 0.0
    output_std: float = 0.0
    # MLP stats (deep mode)
    mlp_output_rms: float = 0.0
    mlp_sparsity: float = 0.0
    # Attention stats (deep mode)
    attn_per_head_entropy: list[float] = field(default_factory=list)
    attn_max_weight: float = 0.0
    attn_mean_weight: float = 0.0


@dataclass
class ModelArchInfo:
    num_layers: int
    num_attention_heads: int
    hidden_size: int
    vocab_size: int
    head_dim: int
    dtype: str
    device: str


def _tensor_rms(t: torch.Tensor) -> float:
    """Root mean square of a tensor, computed safely."""
    return float(t.float().pow(2).mean().sqrt().item())


def _tensor_mean(t: torch.Tensor) -> float:
    return float(t.float().mean().item())


def _tensor_std(t: torch.Tensor) -> float:
    return float(t.float().std().item())


def _sample_logits(
    logits_1d: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    top_n: int,
    tokenizer: Any,
) -> tuple[int, list[TokenCandidate]]:
    """Sample from logits with temperature, top-k, and top-p filtering.

    Args:
        logits_1d: 1-D float tensor of shape [vocab_size]
        temperature: scaling temperature (> 0)
        top_k: keep only top-k logits before nucleus sampling (0 = disabled)
        top_p: nucleus probability threshold (1.0 = disabled)
        top_n: number of top candidates to return in the result list
        tokenizer: HuggingFace tokenizer for decoding candidate token text

    Returns:
        (next_token_id, candidates) where candidates is sorted by filtered
        probability descending and contains at most top_n entries.
    """
    logits = logits_1d.float().clone()

    # ── raw probabilities (before any filtering) ─────────────────────────────
    raw_probs = F.softmax(logits, dim=-1)

    # Collect top_n candidates by raw probability for the event payload
    top_raw_vals, top_raw_ids = torch.topk(raw_probs, min(top_n, raw_probs.size(-1)))
    raw_candidates_by_id: dict[int, float] = {
        int(tid): float(prob)
        for tid, prob in zip(top_raw_ids.tolist(), top_raw_vals.tolist())
    }

    # ── temperature scaling ───────────────────────────────────────────────────
    temperature = max(temperature, 1e-6)  # guard against division by zero
    logits = logits / temperature

    # ── top-k filtering ───────────────────────────────────────────────────────
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        top_k_vals, _ = torch.topk(logits, k)
        min_top_k_val = top_k_vals[-1]
        logits = logits.masked_fill(logits < min_top_k_val, float("-inf"))

    # ── top-p (nucleus) filtering ─────────────────────────────────────────────
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens once cumulative probability exceeds top_p
        # Shift right so that the token that crosses the threshold is kept
        remove_mask = cumulative_probs - sorted_probs > top_p
        sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))

        # Scatter back to original ordering
        logits = torch.zeros_like(logits).scatter_(0, sorted_indices, sorted_logits)

    # ── filtered probabilities and sampling ───────────────────────────────────
    filtered_probs = F.softmax(logits, dim=-1)
    next_token_id = int(torch.multinomial(filtered_probs, num_samples=1).item())

    # ── build candidate list ──────────────────────────────────────────────────
    # Get top_n by filtered probability
    top_filt_vals, top_filt_ids = torch.topk(
        filtered_probs, min(top_n, filtered_probs.size(-1))
    )
    candidates: list[TokenCandidate] = []
    for tid, fprob in zip(top_filt_ids.tolist(), top_filt_vals.tolist()):
        tid_int = int(tid)
        try:
            token_text = tokenizer.decode([tid_int], skip_special_tokens=False)
        except Exception:
            token_text = f"<{tid_int}>"
        candidates.append(
            TokenCandidate(
                token_id=tid_int,
                token_text=token_text,
                logit=float(logits_1d[tid_int].item()),
                probability=float(fprob),
                raw_probability=raw_candidates_by_id.get(tid_int, 0.0),
            )
        )
    candidates.sort(key=lambda c: c.probability, reverse=True)

    return next_token_id, candidates


def _kv_cache_stats(
    past_key_values: Any,
) -> tuple[int, int, list[int], list[int], str]:
    """Extract KV-cache statistics from past_key_values.

    Handles multiple Transformers cache formats:
      - Transformers 5.x  DynamicCache: .layers list of CacheLayer objects,
        each with .keys / .values tensors and a get_seq_length() method.
      - Transformers 4.38–4.45  DynamicCache: .key_cache / .value_cache as
        lists of tensors (one tensor per layer).
      - Classic tuple-of-tuples: ((k0, v0), (k1, v1), …)

    Returns:
        (seq_len, measured_bytes, k_shape, v_shape, dtype_str)
    """
    if past_key_values is None:
        return 0, 0, [], [], "none"

    # ── Transformers 5.x: DynamicCache with .layers list ─────────────────────
    # Each element is a CacheLayer (DynamicLayer, etc.) with .keys / .values.
    layers_attr = getattr(past_key_values, "layers", None)
    if isinstance(layers_attr, list):
        if not layers_attr:
            return 0, 0, [], [], "none"
        layer0 = layers_attr[0]
        k0 = getattr(layer0, "keys", None)
        v0 = getattr(layer0, "values", None)
        if k0 is None or not hasattr(k0, "shape"):
            return 0, 0, [], [], "none"

        k_shape = list(k0.shape)
        v_shape = list(v0.shape) if (v0 is not None and hasattr(v0, "shape")) else k_shape
        dtype_str = str(k0.dtype).replace("torch.", "")

        # get_seq_length() is the most reliable API on the cache object itself.
        seq_len = 0
        if hasattr(past_key_values, "get_seq_length"):
            try:
                seq_len = int(past_key_values.get_seq_length())
            except Exception:
                pass
        if seq_len == 0:
            # Fallback: dim 2 for [batch, heads, seq, head_dim]
            seq_len = k_shape[2] if len(k_shape) >= 3 else 0

        # Byte count: iterate via __iter__ which yields (keys, values, ...)
        total_bytes = 0
        try:
            for item in past_key_values:
                k, v = item[0], item[1]
                if k is not None and hasattr(k, "numel"):
                    total_bytes += k.numel() * k.element_size()
                if v is not None and hasattr(v, "numel"):
                    total_bytes += v.numel() * v.element_size()
        except Exception:
            # Estimation fallback: scale layer-0 by layer count
            try:
                n = len(layers_attr)
                k_bytes = k0.numel() * k0.element_size()
                v_bytes = v0.numel() * v0.element_size() if v0 is not None else k_bytes
                total_bytes = n * (k_bytes + v_bytes)
            except Exception:
                pass

        return seq_len, total_bytes, k_shape, v_shape, dtype_str

    # ── Transformers 4.38–4.45: DynamicCache with .key_cache list ────────────
    key_cache = getattr(past_key_values, "key_cache", None)
    if isinstance(key_cache, list):
        if not key_cache:
            return 0, 0, [], [], "none"
        k0 = key_cache[0]
        val_cache = getattr(past_key_values, "value_cache", [])
        v0 = val_cache[0] if val_cache else k0

        k_shape = list(k0.shape)
        v_shape = list(v0.shape)
        dtype_str = str(k0.dtype).replace("torch.", "")
        seq_len = k_shape[2] if len(k_shape) >= 3 else k_shape[-1]

        total_bytes = 0
        try:
            for k, v in zip(key_cache, val_cache):
                total_bytes += k.numel() * k.element_size()
                total_bytes += v.numel() * v.element_size()
        except Exception:
            pass

        return seq_len, total_bytes, k_shape, v_shape, dtype_str

    # ── Classic tuple-of-tuples: ((k0, v0), (k1, v1), …) ────────────────────
    # Also handles Cache objects whose __iter__ yields (k, v, …) per layer.
    try:
        first_layer = None
        for item in past_key_values:
            first_layer = item
            break
        if first_layer is None:
            return 0, 0, [], [], "none"

        k0 = first_layer[0]
        v0 = first_layer[1]
        if not hasattr(k0, "shape"):
            return 0, 0, [], [], "none"

        k_shape = list(k0.shape)
        v_shape = list(v0.shape)
        dtype_str = str(k0.dtype).replace("torch.", "")
        seq_len = k_shape[2] if len(k_shape) >= 3 else k_shape[-1]

        total_bytes = 0
        for layer in past_key_values:
            k, v = layer[0], layer[1]
            total_bytes += k.numel() * k.element_size()
            total_bytes += v.numel() * v.element_size()

        return seq_len, total_bytes, k_shape, v_shape, dtype_str
    except Exception:
        return 0, 0, [], [], "none"


def _get_model_layers(model: Any) -> list[Any] | None:
    """Discover the list of decoder layers from the model object.

    Tries common HuggingFace decoder architectures in order.
    Returns None if the layers cannot be found.
    """
    # LLaMA / Mistral / Gemma / Phi / Qwen2 style: model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        if hasattr(layers, "__len__") and len(layers) > 0:
            return list(layers)

    # GPT-2 style: model.transformer.h
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        h = model.transformer.h
        if hasattr(h, "__len__") and len(h) > 0:
            return list(h)

    # GPT-NeoX / Pythia style: model.gpt_neox.layers
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        layers = model.gpt_neox.layers
        if hasattr(layers, "__len__") and len(layers) > 0:
            return list(layers)

    # OPT style: model.model.decoder.layers
    if (
        hasattr(model, "model")
        and hasattr(model.model, "decoder")
        and hasattr(model.model.decoder, "layers")
    ):
        layers = model.model.decoder.layers
        if hasattr(layers, "__len__") and len(layers) > 0:
            return list(layers)

    # Falcon style: model.transformer.h (same as GPT-2 — already covered)

    # BLOOM style: model.transformer.h (same path, already covered)

    return None


class InstrumentedTransformersAdapter(BaseAdapter):
    """V2 adapter: instruments a local HuggingFace model with PyTorch forward hooks.

    Parameters
    ----------
    model_id:
        HuggingFace Hub model ID (e.g. ``"meta-llama/Llama-3.2-1B"``) or a
        local filesystem path.
    device:
        ``"cuda"``, ``"cpu"``, or ``"auto"`` (default).  When ``"auto"`` the
        adapter picks CUDA if available, otherwise CPU.
    telemetry_mode:
        * ``"light"``    — timing, top-logits, KV-cache stats.
        * ``"standard"`` — + per-layer hidden-state stats.
        * ``"deep"``     — + MLP sparsity, attention entropy.
    attention_enabled:
        When ``True`` and ``telemetry_mode == "deep"``, request that the model
        returns attention weights (``output_attentions=True``).
    max_new_tokens:
        Maximum tokens to generate per prompt.
    temperature, top_k, top_p:
        Sampling hyper-parameters.
    top_n_logits:
        How many top-candidate tokens to include in :class:`LogitsReadyEvent`.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        telemetry_mode: str = "standard",
        attention_enabled: bool = False,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        top_n_logits: int = 10,
        record_path: str | None = None,
        session_id: str = "",
    ) -> None:
        self._model_id = model_id
        self._device_arg = device
        self._telemetry_mode = telemetry_mode
        self._attention_enabled = attention_enabled and (telemetry_mode == "deep")
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p
        self._top_n_logits = top_n_logits
        self._record_path = record_path
        self._session_id = session_id or ""
        self._sequence_counter: int = 0

        # Async infrastructure
        self._prompt_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._stop_event: asyncio.Event = asyncio.Event()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llmvis_gen")

        # Model state
        self._model: Any = None
        self._tokenizer: Any = None
        self._arch: ModelArchInfo | None = None
        self._model_loaded: bool = False
        self._actual_device: str = "cpu"

        # Hook infrastructure
        self._hooks: list[Any] = []  # list[torch.utils.hooks.RemovableHandle]
        self._hook_data: dict[int, _LayerHookData] = {}  # layer_index -> data
        self._is_prefill: bool = True   # suppress events during prefill

        # Callback set before each generation pass
        self._event_callback: Callable[[LLMVisEvent], None] | None = None

        # Asyncio loop reference (set in run())
        self._loop: asyncio.AbstractEventLoop | None = None

        # Recording file handle (opened lazily)
        self._record_file: Any = None

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    async def load_model(self) -> None:
        """Load the model and tokenizer, register forward hooks, emit ModelArchitectureEvent."""
        if self._model_loaded:
            return

        logger.info("Loading model %s …", self._model_id)
        loop = asyncio.get_running_loop()

        await loop.run_in_executor(self._executor, self._load_model_sync)
        self._model_loaded = True
        logger.info("Model %s loaded on %s", self._model_id, self._actual_device)

    async def submit_prompt(self, text: str) -> None:
        """Queue a prompt for generation."""
        await self._prompt_queue.put(text)

    async def run(self) -> AsyncIterator[LLMVisEvent]:
        """Yield events, driving model generation.

        First loads the model if necessary (emitting :class:`ModelArchitectureEvent`),
        then loops over prompts from :meth:`submit_prompt` until stopped.
        """
        self._loop = asyncio.get_running_loop()

        # Open recording file if requested
        if self._record_path and self._record_file is None:
            try:
                self._record_file = open(self._record_path, "w", encoding="utf-8")
                logger.info("Recording telemetry to %s", self._record_path)
            except OSError as exc:
                logger.error("Cannot open record file %s: %s", self._record_path, exc)

        # Load model and emit architecture event
        if not self._model_loaded:
            try:
                await self.load_model()
            except Exception as exc:
                logger.exception("Failed to load model: %s", exc)
                return

        arch = self._arch
        if arch is not None:
            arch_evt = ModelArchitectureEvent(
                source="instrumented_transformers",
                model_id=self._model_id,
                num_layers=arch.num_layers,
                num_attention_heads=arch.num_attention_heads,
                hidden_size=arch.hidden_size,
                vocab_size=arch.vocab_size,
                head_dim=arch.head_dim,
                dtype=arch.dtype,
                device=arch.device,
                session_id=self._session_id,
                sequence_num=self._next_seq(),
            )
            self._record_event(arch_evt)
            yield arch_evt

        # Main generation loop
        while not self._stop_event.is_set():
            try:
                prompt = await asyncio.wait_for(
                    self._prompt_queue.get(), timeout=0.1
                )
            except asyncio.TimeoutError:
                continue

            if prompt is None:
                break

            async for event in self._run_generation(prompt):
                self._record_event(event)
                yield event

    async def stop(self) -> None:
        """Signal the adapter to stop after the current prompt (if any)."""
        self._stop_event.set()
        await self._prompt_queue.put(None)  # unblock the queue wait

        # Remove all hooks
        self._remove_hooks()

        # Close recording file
        if self._record_file is not None:
            try:
                self._record_file.close()
            except Exception:
                pass
            self._record_file = None

        # Shutdown executor
        self._executor.shutdown(wait=False)

    def _next_seq(self) -> int:
        self._sequence_counter += 1
        return self._sequence_counter

    def _record_event(self, evt: LLMVisEvent) -> None:
        """Write event to JSONL recording file if recording is active."""
        if self._record_file is None:
            return
        try:
            from llmvis.core.events import event_to_jsonl
            self._record_file.write(event_to_jsonl(evt) + "\n")
            self._record_file.flush()
        except Exception as exc:
            logger.debug("Failed to record event: %s", exc)

    # ─────────────────────────────────────────────────────────────────────────
    # Internal: model loading (sync, called from executor)
    # ─────────────────────────────────────────────────────────────────────────

    def _load_model_sync(self) -> None:
        """Synchronous model loading executed in the thread pool."""
        # Import here so the module can be imported without transformers installed
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]

        device_map: str | dict[str, str]
        if self._device_arg == "auto":
            device_map = "auto"
        else:
            device_map = self._device_arg

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_id,
            trust_remote_code=True,
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            torch_dtype="auto",
            device_map=device_map,
            trust_remote_code=True,
        )
        self._model.eval()

        # Detect actual device from first parameter
        try:
            first_param = next(self._model.parameters())
            self._actual_device = str(first_param.device)
        except StopIteration:
            self._actual_device = self._device_arg if self._device_arg != "auto" else "cpu"

        self._arch = self._discover_arch()
        self._register_hooks()

    # ─────────────────────────────────────────────────────────────────────────
    # Internal: architecture discovery
    # ─────────────────────────────────────────────────────────────────────────

    def _discover_arch(self) -> ModelArchInfo:
        config = self._model.config
        num_layers = int(getattr(config, "num_hidden_layers", 0))
        num_heads = int(getattr(config, "num_attention_heads", 0))
        hidden_size = int(getattr(config, "hidden_size", 0))
        vocab_size = int(getattr(config, "vocab_size", 0))
        head_dim_cfg = getattr(config, "head_dim", None)
        head_dim = (
            int(head_dim_cfg)
            if head_dim_cfg is not None
            else (hidden_size // num_heads if num_heads > 0 else 0)
        )

        # Determine dtype string from model parameters
        try:
            first_param = next(self._model.parameters())
            dtype_str = str(first_param.dtype).replace("torch.", "")
        except StopIteration:
            dtype_str = "unknown"

        return ModelArchInfo(
            num_layers=num_layers,
            num_attention_heads=num_heads,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            head_dim=head_dim,
            dtype=dtype_str,
            device=self._actual_device,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Internal: hook registration
    # ─────────────────────────────────────────────────────────────────────────

    def _register_hooks(self) -> None:
        """Register pre- and post-forward hooks on all discovered decoder layers."""
        layers = _get_model_layers(self._model)
        if layers is None:
            logger.warning(
                "Could not discover decoder layers for %s — layer stats will be unavailable.",
                self._model_id,
            )
            return

        for idx, layer in enumerate(layers):
            # Pre-hook: record entry time and input statistics
            pre_handle = layer.register_forward_pre_hook(
                self._make_pre_hook(idx)
            )
            # Post-hook: record exit time and output statistics
            post_handle = layer.register_forward_hook(
                self._make_post_hook(idx)
            )
            self._hooks.extend([pre_handle, post_handle])

        logger.debug(
            "Registered %d hooks across %d layers", len(self._hooks), len(layers)
        )

    def _remove_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for handle in self._hooks:
            try:
                handle.remove()
            except Exception:
                pass
        self._hooks.clear()

    def _make_pre_hook(self, layer_idx: int) -> Callable:
        """Return a pre-forward hook closure for the given layer index."""

        def pre_hook(module: Any, inputs: tuple[Any, ...]) -> None:
            if self._is_prefill and self._telemetry_mode != "deep":
                # During prefill in non-deep mode just record the start time
                data = _LayerHookData(layer_index=layer_idx)
                data.start_time = time.perf_counter()
                self._hook_data[layer_idx] = data
                return

            data = _LayerHookData(layer_index=layer_idx)
            data.start_time = time.perf_counter()

            # Extract the hidden state from inputs (first positional arg)
            hidden = None
            if inputs:
                candidate = inputs[0]
                if isinstance(candidate, torch.Tensor):
                    hidden = candidate
                elif isinstance(candidate, (list, tuple)) and len(candidate) > 0:
                    if isinstance(candidate[0], torch.Tensor):
                        hidden = candidate[0]

            if hidden is not None and hidden.numel() > 0:
                h = hidden.detach().float()
                data.input_rms = float(h.pow(2).mean().sqrt().item())
                data.input_mean = float(h.mean().item())
                data.input_std = float(h.std().item())

            self._hook_data[layer_idx] = data

        return pre_hook

    def _make_post_hook(self, layer_idx: int) -> Callable:
        """Return a post-forward hook closure for the given layer index."""

        def post_hook(module: Any, inputs: tuple[Any, ...], outputs: Any) -> None:
            data = self._hook_data.get(layer_idx)
            if data is None:
                return

            data.end_time = time.perf_counter()

            # Extract hidden state from outputs
            hidden = None
            if isinstance(outputs, torch.Tensor):
                hidden = outputs
            elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
                candidate = outputs[0]
                if isinstance(candidate, torch.Tensor):
                    hidden = candidate

            if hidden is not None and hidden.numel() > 0:
                h = hidden.detach().float()
                data.output_rms = float(h.pow(2).mean().sqrt().item())
                data.output_mean = float(h.mean().item())
                data.output_std = float(h.std().item())

                # deep mode: MLP sparsity (approximate — check output near zero)
                if self._telemetry_mode == "deep" and not self._is_prefill:
                    sparsity = float((h.abs() < 0.01).float().mean().item())
                    data.mlp_sparsity = sparsity
                    data.mlp_output_rms = data.output_rms

            # deep mode: attention weights (only if model exposes them)
            if (
                self._telemetry_mode == "deep"
                and not self._is_prefill
                and isinstance(outputs, (tuple, list))
                and len(outputs) >= 2
            ):
                attn_weights = None
                # Typically outputs = (hidden, present_key_value, attn_weights) or
                # (hidden, attn_weights) depending on model
                for out_item in outputs[1:]:
                    if isinstance(out_item, torch.Tensor) and out_item.dim() == 4:
                        attn_weights = out_item
                        break

                if attn_weights is not None:
                    # shape: [batch, heads, seq, seq]
                    aw = attn_weights.detach().float()
                    # last query position for autoregressive step
                    aw_last = aw[0, :, -1, :]  # [heads, seq]
                    probs = F.softmax(aw_last, dim=-1)
                    # per-head entropy
                    entropy_per_head: list[float] = []
                    for h_idx in range(probs.shape[0]):
                        p = probs[h_idx]
                        ent = float(-((p * (p + 1e-9).log()).sum()).item())
                        entropy_per_head.append(ent)
                    data.attn_per_head_entropy = entropy_per_head
                    data.attn_max_weight = float(probs.max().item())
                    data.attn_mean_weight = float(probs.mean().item())

        return post_hook

    # ─────────────────────────────────────────────────────────────────────────
    # Internal: generation
    # ─────────────────────────────────────────────────────────────────────────

    async def _run_generation(self, prompt: str) -> AsyncIterator[LLMVisEvent]:
        """Drive full generation for a single prompt, yielding events."""
        loop = asyncio.get_running_loop()
        event_queue: asyncio.Queue[LLMVisEvent | None] = asyncio.Queue()

        def callback(evt: LLMVisEvent) -> None:
            """Thread-safe bridge: put event onto asyncio queue."""
            loop.call_soon_threadsafe(event_queue.put_nowait, evt)

        self._event_callback = callback

        # Run generation in thread pool
        future = loop.run_in_executor(
            self._executor,
            self._generate_sync,
            prompt,
        )

        # Yield events as they arrive
        while True:
            try:
                event = await asyncio.wait_for(event_queue.get(), timeout=0.05)
            except asyncio.TimeoutError:
                # Check if generation is done
                if future.done():
                    # Drain any remaining events
                    while not event_queue.empty():
                        try:
                            event = event_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                        if event is None:
                            return
                        yield event
                    break
                continue

            if event is None:
                # Sentinel: generation complete
                break
            yield event

        # Propagate any exception from the generation thread
        try:
            await future
        except Exception as exc:
            logger.exception("Generation failed: %s", exc)

        self._event_callback = None

    def _emit(self, evt: LLMVisEvent) -> None:
        """Emit an event via the registered callback (called from generation thread)."""
        if self._event_callback is not None:
            self._event_callback(evt)

    def _collect_eos_ids(self) -> frozenset[int]:
        """Collect all valid stop token IDs from tokenizer and model generation config.

        Instruct models (e.g. Qwen2.5-Instruct) use both <|endoftext|> and <|im_end|>
        as stop tokens. Reading from generation_config catches model-specific stop IDs
        that tokenizer.eos_token_id alone may miss.
        """
        ids: set[int] = set()
        tok_eos = getattr(self._tokenizer, "eos_token_id", None)
        if isinstance(tok_eos, int):
            ids.add(tok_eos)
        elif isinstance(tok_eos, (list, tuple)):
            ids.update(int(x) for x in tok_eos)

        if hasattr(self._model, "generation_config"):
            gc_eos = getattr(self._model.generation_config, "eos_token_id", None)
            if isinstance(gc_eos, int):
                ids.add(gc_eos)
            elif isinstance(gc_eos, (list, tuple)):
                ids.update(int(x) for x in gc_eos)

        return frozenset(ids)

    def _generate_sync(self, prompt: str) -> None:
        """Synchronous generation loop — runs in executor thread."""
        import sys as _sys
        src = "instrumented_transformers"
        inference_start = time.perf_counter()
        # Define before try so the except block can reference it safely
        generated_ids: list[int] = []

        try:
            # ── InferenceStartEvent ───────────────────────────────────────────
            self._emit(
                InferenceStartEvent(
                    source=src,
                    model_id=self._model_id,
                    prompt_preview=prompt[:100],
                )
            )

            # ── PromptReceivedEvent ──────────────────────────────────────────
            self._emit(PromptReceivedEvent(source=src, text=prompt))

            # ── Tokenization ─────────────────────────────────────────────────
            tok_start = time.perf_counter()
            chat_template = getattr(self._tokenizer, "chat_template", None)
            if chat_template is not None and str(chat_template).strip():
                # Instruct/chat model — use the tokenizer's chat template so the
                # model receives correctly formatted turn markers (e.g. <|im_start|>).
                try:
                    raw = self._tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    )
                    # apply_chat_template may return a bare tensor or a BatchEncoding
                    # depending on the transformers version.
                    if hasattr(raw, "input_ids"):
                        input_ids = raw["input_ids"].to(self._actual_device)
                    else:
                        input_ids = raw.to(self._actual_device)
                except Exception:
                    # Fallback: plain tokenization (e.g. tokenizer API mismatch)
                    encoded = self._tokenizer(prompt, return_tensors="pt")
                    input_ids = encoded["input_ids"].to(self._actual_device)
            else:
                # Base/completion model — plain tokenization
                encoded = self._tokenizer(prompt, return_tensors="pt")
                input_ids = encoded["input_ids"].to(self._actual_device)
            tok_ms = (time.perf_counter() - tok_start) * 1000.0

            token_ids_list: list[int] = input_ids[0].tolist()
            self._emit(
                TokenizationCompleteEvent(
                    source=src,
                    token_count=len(token_ids_list),
                    token_ids=token_ids_list,
                    duration_ms=tok_ms,
                )
            )

            # ── Prefill ───────────────────────────────────────────────────────
            self._emit(PrefillStartEvent(source=src, token_count=len(token_ids_list)))

            self._is_prefill = True
            self._hook_data.clear()

            prefill_start = time.perf_counter()
            with torch.no_grad():
                attention_mask = torch.ones_like(input_ids)
                model_kwargs: dict[str, Any] = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "use_cache": True,
                }
                if self._attention_enabled:
                    model_kwargs["output_attentions"] = True

                outputs = self._model(**model_kwargs)

            prefill_ms = (time.perf_counter() - prefill_start) * 1000.0
            self._emit(
                PrefillEndEvent(
                    source=src,
                    token_count=len(token_ids_list),
                    duration_ms=prefill_ms,
                )
            )

            # past_key_values from prefill
            past_key_values = outputs.past_key_values
            prev_seq_len = len(token_ids_list)

            # ── Decode loop ───────────────────────────────────────────────────
            sampler_cfg = SamplerConfig(
                temperature=self._temperature,
                top_k=self._top_k,
                top_p=self._top_p,
            )
            self._emit(DecodeStartEvent(source=src, sampler=sampler_cfg))

            self._is_prefill = False
            # generated_ids already declared above the try block
            eos_ids = self._collect_eos_ids()

            # Maintain attention mask as we grow the sequence
            current_attention_mask = attention_mask

            for token_index in range(self._max_new_tokens):
                if self._stop_event.is_set():
                    break

                token_step_start = time.perf_counter()

                # ── TokenStartEvent ───────────────────────────────────────────
                self._emit(TokenStartEvent(source=src, token_index=token_index))

                # Prepare next input: the last generated token (or last prompt token for
                # first decode step where past_key_values already covers the prompt)
                if token_index == 0:
                    next_input_ids = input_ids[:, -1:]
                else:
                    next_input_ids = torch.tensor(
                        [[generated_ids[-1]]], device=self._actual_device
                    )

                # Extend attention mask by one
                current_attention_mask = torch.cat(
                    [
                        current_attention_mask,
                        torch.ones(
                            (current_attention_mask.shape[0], 1),
                            dtype=current_attention_mask.dtype,
                            device=self._actual_device,
                        ),
                    ],
                    dim=1,
                )

                # Clear hook data for this step
                self._hook_data.clear()

                step_kwargs: dict[str, Any] = {
                    "input_ids": next_input_ids,
                    "attention_mask": current_attention_mask,
                    "past_key_values": past_key_values,
                    "use_cache": True,
                }
                if self._attention_enabled:
                    step_kwargs["output_attentions"] = True

                with torch.no_grad():
                    step_outputs = self._model(**step_kwargs)

                past_key_values = step_outputs.past_key_values
                # logits shape: [batch=1, seq=1, vocab_size]
                logits_last = step_outputs.logits[0, -1, :]  # [vocab_size]

                # ── Layer stats events ────────────────────────────────────────
                if self._telemetry_mode in ("standard", "deep"):
                    arch = self._arch
                    prev_rms: float | None = None
                    for layer_idx in sorted(self._hook_data.keys()):
                        hd = self._hook_data[layer_idx]
                        exec_ms = (hd.end_time - hd.start_time) * 1000.0
                        delta = 0.0
                        if prev_rms is not None:
                            delta = abs(hd.output_rms - prev_rms)
                        prev_rms = hd.output_rms

                        self._emit(
                            LayerStatsEvent(
                                source=src,
                                token_index=token_index,
                                layer_index=layer_idx,
                                hidden_state_rms=hd.output_rms,
                                hidden_state_mean=hd.output_mean,
                                hidden_state_std=hd.output_std,
                                delta_from_prev=delta,
                                exec_time_ms=exec_ms,
                            )
                        )

                        if self._telemetry_mode == "deep":
                            self._emit(
                                MLPStatsEvent(
                                    source=src,
                                    token_index=token_index,
                                    layer_index=layer_idx,
                                    output_rms=hd.mlp_output_rms,
                                    sparsity=hd.mlp_sparsity,
                                )
                            )

                            if hd.attn_per_head_entropy:
                                self._emit(
                                    AttentionStatsEvent(
                                        source=src,
                                        token_index=token_index,
                                        layer_index=layer_idx,
                                        per_head_entropy=hd.attn_per_head_entropy,
                                        max_weight=hd.attn_max_weight,
                                        mean_weight=hd.attn_mean_weight,
                                    )
                                )

                # ── Sampling ──────────────────────────────────────────────────
                next_token_id, candidates = _sample_logits(
                    logits_last,
                    temperature=self._temperature,
                    top_k=self._top_k,
                    top_p=self._top_p,
                    top_n=self._top_n_logits,
                    tokenizer=self._tokenizer,
                )

                # LogitsReadyEvent
                arch_vocab = self._arch.vocab_size if self._arch else logits_last.size(0)
                self._emit(
                    LogitsReadyEvent(
                        source=src,
                        token_index=token_index,
                        top_candidates=candidates,
                        vocab_size=arch_vocab,
                    )
                )

                # ── EOS check — stop BEFORE emitting token text events ────────
                # This prevents stop tokens (e.g. <|im_end|>, <|endoftext|>) from
                # appearing in token history or Terminal 1 output. LogitsReadyEvent
                # was already emitted above so the TUI sees the final distribution.
                if next_token_id in eos_ids:
                    generated_ids.append(next_token_id)
                    break

                # ── Decode display text (skip special/control tokens) ─────────
                try:
                    sampled_text = self._tokenizer.decode(
                        [next_token_id], skip_special_tokens=True
                    )
                except Exception:
                    sampled_text = f"<{next_token_id}>"

                # logprob of sampled token
                with torch.no_grad():
                    log_probs = F.log_softmax(logits_last.float(), dim=-1)
                    logprob_val = float(log_probs[next_token_id].item())

                self._emit(
                    TokenSampledEvent(
                        source=src,
                        token_index=token_index,
                        token_id=next_token_id,
                        token_text=sampled_text,
                        logprob=logprob_val,
                        sampler=sampler_cfg,
                    )
                )

                # ── KV cache stats ────────────────────────────────────────────
                seq_len_after, measured_bytes, k_shape, v_shape, dtype_str = (
                    _kv_cache_stats(past_key_values)
                )
                num_layers_kv = (
                    self._arch.num_layers
                    if self._arch
                    else len(past_key_values)
                    if past_key_values is not None
                    else 0
                )
                self._emit(
                    KvCacheUpdateEvent(
                        source=src,
                        token_index=token_index,
                        seq_len_before=prev_seq_len,
                        seq_len_after=seq_len_after,
                        num_layers=num_layers_kv,
                        k_shape=k_shape,
                        v_shape=v_shape,
                        dtype=dtype_str,
                        measured_bytes=measured_bytes,
                    )
                )
                prev_seq_len = seq_len_after

                # ── TokenEndEvent ─────────────────────────────────────────────
                token_latency_ms = (time.perf_counter() - token_step_start) * 1000.0
                self._emit(
                    TokenEndEvent(
                        source=src,
                        token_index=token_index,
                        token_text=sampled_text,
                        token_id=next_token_id,
                        latency_ms=token_latency_ms,
                    )
                )

                # ── TokenGeneratedEvent (compatibility) ───────────────────────
                self._emit(
                    TokenGeneratedEvent(
                        source=src,
                        token_id=next_token_id,
                        token_text=sampled_text,
                        logprob=logprob_val,
                        latency_ms=token_latency_ms,
                    )
                )

                generated_ids.append(next_token_id)

            # ── InferenceEndEvent ─────────────────────────────────────────────
            total_ms = (time.perf_counter() - inference_start) * 1000.0
            self._emit(
                InferenceEndEvent(
                    source=src,
                    model_id=self._model_id,
                    total_tokens=len(generated_ids),
                    total_time_ms=total_ms,
                )
            )

        except Exception as exc:
            # Print directly to stderr so the error is always visible,
            # regardless of the configured log level.
            _sys.stderr.write(
                f"\nGeneration error: {type(exc).__name__}: {exc}\n"
            )
            _sys.stderr.flush()
            logger.exception("Error during generation: %s", exc)
            # Emit INFERENCE_END so the event loop knows generation has stopped
            # and the REPL's _generation_done gate gets unblocked.
            total_ms_err = (time.perf_counter() - inference_start) * 1000.0
            self._emit(
                InferenceEndEvent(
                    source=src,
                    model_id=self._model_id,
                    total_tokens=len(generated_ids),
                    total_time_ms=total_ms_err,
                )
            )
        finally:
            # Signal end of events
            if self._event_callback is not None:
                self._event_callback(None)  # type: ignore[arg-type]
