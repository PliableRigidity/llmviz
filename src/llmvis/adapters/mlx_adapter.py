"""MLXAdapter — deep instrumentation for Apple Silicon via MLX.

This adapter provides normalized LLMVis events from models loaded with
mlx-lm, the Apple MLX inference library for Apple Silicon / Metal.

## What IS measurable with MLX (vs PyTorch/CUDA)

| Feature                | PyTorch/CUDA | MLX/Metal |
|------------------------|:---:|:---:|
| Prefill timing         |  ✓  |  ✓ * |
| Decode timing          |  ✓  |  ✓ * |
| Per-token latency      |  ✓  |  ✓ * |
| Top-N logits           |  ✓  |  ✓  |
| Raw/filtered probs     |  ✓  |  ✓  |
| KV cache seq_len       |  ✓  |  ✓  |
| KV cache byte size     |  ✓  |  ✓  |
| Per-layer hidden stats |  ✓  |  ✗  |
| Attention weights      |  ✓  |  ✗  |
| MLP sparsity           |  ✓  |  ✗  |
| Metal allocation bytes |  ✗  |  ✓  |

* Timing on MLX REQUIRES explicit mx.eval() before reading wall-clock.
  MLX uses lazy evaluation — tensors are not computed until evaluated.
  We call mx.eval() at prefill end and after each token decode step.

## Why layer stats are unavailable

MLX does not support PyTorch-style forward hooks (register_forward_hook).
Intercepting hidden states would require monkey-patching every layer class
in mlx-lm, which varies by model architecture and would break on updates.
This is planned for a future version after mlx-lm exposes a stable hook API.

## Layer stats workaround

LAYER_STATS events are NOT emitted by this adapter.
The TUI capability flag `layer_stats=False` prevents the Transformer tab
from rendering when this adapter is in use.

## Telemetry modes

light    — timing, logits, KV stats (default for MLX; all that's available)
standard — same as light (no additional data available)
deep     — same as light

All modes emit the same events because MLX does not expose per-layer data.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from llmvis.adapters.base import BaseAdapter
from llmvis.core.events import (
    DecodeStartEvent,
    InferenceEndEvent,
    InferenceStartEvent,
    KvCacheUpdateEvent,
    LLMVisEvent,
    LogitsReadyEvent,
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


def _check_mlx() -> bool:
    try:
        import mlx.core  # noqa: F401  # type: ignore[import]
        return True
    except (ImportError, ModuleNotFoundError):
        return False


class MLXAdapter(BaseAdapter):
    """Deep-instrumentation adapter for Apple Silicon models via mlx-lm.

    Parameters
    ----------
    model_id:
        HuggingFace Hub model ID or local path. mlx-lm will look for
        converted MLX weights (or convert on the fly if the source is
        a standard HuggingFace model repo).
    telemetry_mode:
        "light" | "standard" | "deep" — all emit the same events on MLX
        because per-layer hook data is not available (see module docstring).
    max_new_tokens:
        Maximum tokens to generate per prompt.
    temperature, top_k, top_p:
        Sampling parameters.
    top_n_logits:
        How many top-candidate tokens to include in LogitsReadyEvent.
    """

    def __init__(
        self,
        model_id: str,
        telemetry_mode: str = "light",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        top_n_logits: int = 10,
    ) -> None:
        if not _check_mlx():
            raise RuntimeError(
                "MLX is not available on this platform. "
                "Install it with: pip install mlx mlx-lm  "
                "(requires Apple Silicon / macOS arm64)"
            )

        self._model_id = model_id
        self._telemetry_mode = telemetry_mode
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p
        self._top_n_logits = top_n_logits

        self._model: Any = None
        self._tokenizer: Any = None
        self._model_loaded = False

        self._prompt_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._stop_event = asyncio.Event()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llmvis_mlx")
        self._event_callback: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None

    # ── Public API ─────────────────────────────────────────────────────────────

    async def load_model(self) -> None:
        if self._model_loaded:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._load_model_sync)
        self._model_loaded = True

    async def submit_prompt(self, text: str) -> None:
        await self._prompt_queue.put(text)

    async def run(self) -> AsyncIterator[LLMVisEvent]:
        self._loop = asyncio.get_running_loop()

        if not self._model_loaded:
            try:
                await self.load_model()
            except Exception as exc:
                logger.exception("Failed to load MLX model: %s", exc)
                return

        # Emit architecture event
        if self._model is not None:
            yield self._make_arch_event()

        while not self._stop_event.is_set():
            try:
                prompt = await asyncio.wait_for(self._prompt_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            if prompt is None:
                break
            async for evt in self._run_generation(prompt):
                yield evt

    async def stop(self) -> None:
        self._stop_event.set()
        await self._prompt_queue.put(None)
        self._executor.shutdown(wait=False)

    # ── Internal: model loading ────────────────────────────────────────────────

    def _load_model_sync(self) -> None:
        try:
            import mlx_lm  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "mlx-lm is not installed. Install with: pip install mlx-lm"
            ) from exc

        self._model, self._tokenizer = mlx_lm.load(self._model_id)
        logger.info("MLX model %s loaded", self._model_id)

    # ── Internal: architecture discovery ──────────────────────────────────────

    def _make_arch_event(self) -> ModelArchitectureEvent:
        config = getattr(self._model, "config", None) or getattr(
            self._model, "model_type", None
        )
        num_layers = 0
        num_heads = 0
        hidden_size = 0
        vocab_size = 0
        head_dim = 0

        if config is not None:
            num_layers = int(getattr(config, "num_hidden_layers", 0))
            num_heads = int(getattr(config, "num_attention_heads", 0))
            hidden_size = int(getattr(config, "hidden_size", 0))
            vocab_size = int(getattr(config, "vocab_size", 0))
            hd = getattr(config, "head_dim", None)
            head_dim = int(hd) if hd else (hidden_size // num_heads if num_heads else 0)

        return ModelArchitectureEvent(
            source="mlx_adapter",
            model_id=self._model_id,
            num_layers=num_layers,
            num_attention_heads=num_heads,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            head_dim=head_dim,
            dtype="float16",  # mlx-lm typically uses float16/bfloat16
            device="metal",
        )

    # ── Internal: generation ───────────────────────────────────────────────────

    async def _run_generation(self, prompt: str) -> AsyncIterator[LLMVisEvent]:
        loop = asyncio.get_running_loop()
        event_queue: asyncio.Queue[LLMVisEvent | None] = asyncio.Queue()

        def callback(evt: LLMVisEvent) -> None:
            loop.call_soon_threadsafe(event_queue.put_nowait, evt)

        self._event_callback = callback
        future = loop.run_in_executor(self._executor, self._generate_sync, prompt)

        while True:
            try:
                evt = await asyncio.wait_for(event_queue.get(), timeout=0.05)
            except asyncio.TimeoutError:
                if future.done():
                    while not event_queue.empty():
                        try:
                            evt = event_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                        if evt is None:
                            return
                        yield evt
                    break
                continue
            if evt is None:
                break
            yield evt

        try:
            await future
        except Exception as exc:
            logger.exception("MLX generation failed: %s", exc)

        self._event_callback = None

    def _emit(self, evt: LLMVisEvent) -> None:
        if self._event_callback is not None:
            self._event_callback(evt)

    def _generate_sync(self, prompt: str) -> None:
        """Synchronous MLX generation. Runs in the thread pool.

        Event ordering matches InstrumentedTransformersAdapter:
          TOKEN_START → LOGITS_READY → TOKEN_SAMPLED → [forward pass]
          → KV_CACHE_UPDATE → TOKEN_GENERATED → TOKEN_END
        """
        import mlx.core as mx  # type: ignore[import]

        src = "mlx_adapter"
        t_inference_start = time.perf_counter()

        try:
            self._emit(InferenceStartEvent(
                source=src,
                model_id=self._model_id,
                prompt_preview=prompt[:100],
            ))
            self._emit(PromptReceivedEvent(source=src, text=prompt))

            # Tokenize
            t_tok = time.perf_counter()
            token_ids: list[int] = self._tokenizer.encode(prompt)
            tok_ms = (time.perf_counter() - t_tok) * 1000.0
            self._emit(TokenizationCompleteEvent(
                source=src,
                token_count=len(token_ids),
                token_ids=token_ids,
                duration_ms=tok_ms,
            ))

            # Prefill
            self._emit(PrefillStartEvent(source=src, token_count=len(token_ids)))
            t_prefill = time.perf_counter()

            input_array = mx.array(token_ids)[None]  # [1, seq]
            cache = None
            if hasattr(self._model, "make_cache"):
                cache = self._model.make_cache()

            logits, cache = self._model(input_array, cache=cache)
            # Force evaluation — MLX is lazy; timing is meaningless without mx.eval()
            mx.eval(logits)

            prefill_ms = (time.perf_counter() - t_prefill) * 1000.0
            self._emit(PrefillEndEvent(
                source=src,
                token_count=len(token_ids),
                duration_ms=prefill_ms,
            ))

            sampler_cfg = SamplerConfig(
                temperature=self._temperature,
                top_k=self._top_k,
                top_p=self._top_p,
            )
            self._emit(DecodeStartEvent(source=src, sampler=sampler_cfg))

            eos_id = self._tokenizer.eos_token_id
            generated_ids: list[int] = []
            prev_seq_len = len(token_ids)

            for token_index in range(self._max_new_tokens):
                if self._stop_event.is_set():
                    break

                t_token = time.perf_counter()
                self._emit(TokenStartEvent(source=src, token_index=token_index))

                # logits[0, -1, :] = prediction for the next token
                # For step 0 this is the prefill output; for subsequent steps it's
                # from the previous decode forward pass.
                logits_last = logits[0, -1, :]  # [vocab]

                # Sample (uses actual logits from model output)
                next_token_id, candidates = self._sample_mlx(logits_last, mx)

                self._emit(LogitsReadyEvent(
                    source=src,
                    token_index=token_index,
                    top_candidates=candidates,
                    vocab_size=int(logits_last.shape[-1]),
                ))

                try:
                    token_text = self._tokenizer.decode([next_token_id])
                except Exception:
                    token_text = f"<{next_token_id}>"

                self._emit(TokenSampledEvent(
                    source=src,
                    token_index=token_index,
                    token_id=next_token_id,
                    token_text=token_text,
                    logprob=0.0,
                    sampler=sampler_cfg,
                ))

                # Run the decode forward pass — this updates the KV cache with
                # the newly sampled token and produces logits for the NEXT step.
                next_array = mx.array([[next_token_id]])
                logits, cache = self._model(next_array, cache=cache)
                mx.eval(logits)  # force Metal execution before timing

                token_ms = (time.perf_counter() - t_token) * 1000.0

                # KV stats measured from the updated cache (post forward pass)
                seq_len_after, kv_bytes, k_shape, v_shape = self._kv_stats(
                    cache, prev_seq_len + 1
                )
                self._emit(KvCacheUpdateEvent(
                    source=src,
                    token_index=token_index,
                    seq_len_before=prev_seq_len,
                    seq_len_after=seq_len_after,
                    num_layers=0,
                    k_shape=k_shape,
                    v_shape=v_shape,
                    dtype="float16",
                    measured_bytes=kv_bytes,
                ))
                prev_seq_len = seq_len_after

                self._emit(TokenGeneratedEvent(
                    source=src,
                    token_id=next_token_id,
                    token_text=token_text,
                    logprob=0.0,
                    latency_ms=token_ms,
                ))
                self._emit(TokenEndEvent(
                    source=src,
                    token_index=token_index,
                    token_text=token_text,
                    token_id=next_token_id,
                    latency_ms=token_ms,
                ))

                generated_ids.append(next_token_id)
                if eos_id is not None and next_token_id == eos_id:
                    break

            total_ms = (time.perf_counter() - t_inference_start) * 1000.0
            self._emit(InferenceEndEvent(
                source=src,
                model_id=self._model_id,
                total_tokens=len(generated_ids),
                total_time_ms=total_ms,
            ))

        except Exception as exc:
            logger.exception("MLX generation error: %s", exc)
        finally:
            if self._event_callback is not None:
                self._event_callback(None)  # type: ignore[arg-type]

    def _sample_mlx(self, logits_1d: Any, mx: Any) -> tuple[int, list[TokenCandidate]]:
        """Sample from MLX logits tensor. Returns (token_id, candidates)."""
        logits = logits_1d.astype(mx.float32)

        # Raw probabilities (before filtering)
        raw_probs = mx.softmax(logits, axis=-1)
        mx.eval(raw_probs)
        raw_np = raw_probs.tolist()  # list of floats

        # Temperature scaling
        temp = max(self._temperature, 1e-6)
        scaled = logits / temp

        # Top-k
        if self._top_k > 0:
            vocab_size = int(logits.shape[-1])
            k = min(self._top_k, vocab_size)
            # Get top-k indices
            top_k_indices = mx.argpartition(-scaled, kth=k - 1)[:k]
            # Build mask
            mask = mx.full(logits.shape, float("-inf"), dtype=mx.float32)
            # Scatter top-k values back
            top_k_vals = scaled[top_k_indices]
            # Rebuild as dense: set top-k positions
            # (MLX does not have scatter_ — use Python loop for tiny k)
            scaled_list = [float("-inf")] * vocab_size
            for idx, val in zip(top_k_indices.tolist(), top_k_vals.tolist()):
                scaled_list[idx] = val
            scaled = mx.array(scaled_list)

        # Top-p (nucleus) — rebuild scaled logits with -inf for excluded tokens
        if self._top_p < 1.0:
            probs_after_topk = mx.softmax(scaled, axis=-1)
            mx.eval(probs_after_topk)
            probs_list = probs_after_topk.tolist()

            # Sort by probability descending to find nucleus threshold
            indexed_probs = sorted(enumerate(probs_list), key=lambda x: x[1], reverse=True)
            cumulative = 0.0
            keep_ids: set[int] = set()
            for tok_id, prob in indexed_probs:
                keep_ids.add(tok_id)
                cumulative += prob
                if cumulative >= self._top_p:
                    break

            # Rebuild scaled with -inf for excluded tokens
            scaled_list_filtered = [
                val if i in keep_ids else float("-inf")
                for i, val in enumerate(probs_list)
            ]
            scaled = mx.array(scaled_list_filtered)

        # Final distribution
        filtered_probs = mx.softmax(scaled, axis=-1)
        mx.eval(filtered_probs)
        filtered_list = filtered_probs.tolist()

        # Sample via numpy/multinomial fallback (MLX has mx.random.categorical)
        try:
            next_token_id = int(mx.random.categorical(mx.log(filtered_probs)).item())
        except Exception:
            # Fallback: greedy
            next_token_id = int(mx.argmax(filtered_probs).item())

        # Build top-N candidate list
        vocab_size = len(filtered_list)
        top_n = min(self._top_n_logits, vocab_size)

        # Sort indices by filtered probability
        indexed = sorted(enumerate(filtered_list), key=lambda x: x[1], reverse=True)[:top_n]
        candidates: list[TokenCandidate] = []
        for tid, fprob in indexed:
            try:
                token_text = self._tokenizer.decode([tid])
            except Exception:
                token_text = f"<{tid}>"
            raw_prob = raw_np[tid] if tid < len(raw_np) else 0.0
            candidates.append(TokenCandidate(
                token_id=tid,
                token_text=token_text,
                logit=float(logits_1d[tid].item()) if hasattr(logits_1d[tid], "item") else float(logits_1d[tid]),
                raw_probability=float(raw_prob),
                probability=float(fprob),
            ))

        return next_token_id, candidates

    def _kv_stats(
        self, cache: Any, expected_seq_len: int
    ) -> tuple[int, int, list[int], list[int]]:
        """Extract KV cache statistics from the mlx-lm cache object."""
        if cache is None:
            return expected_seq_len, 0, [], []

        seq_len = expected_seq_len
        total_bytes = 0
        k_shape: list[int] = []
        v_shape: list[int] = []

        try:
            # mlx-lm KVCache: list of per-layer KVCache objects
            # Each has .keys / .values (MLX arrays)
            if hasattr(cache, "__iter__"):
                for layer_cache in cache:
                    k = getattr(layer_cache, "keys", None) or getattr(layer_cache, "k", None)
                    v = getattr(layer_cache, "values", None) or getattr(layer_cache, "v", None)
                    if k is not None:
                        k_shape = list(k.shape) if hasattr(k, "shape") else []
                        v_shape = list(v.shape) if hasattr(v, "shape") else []
                        k_bytes = k.nbytes if hasattr(k, "nbytes") else 0
                        v_bytes = v.nbytes if hasattr(v, "nbytes") else 0
                        total_bytes += k_bytes + v_bytes
                        # seq dim is usually dim 2 or dim 1
                        if len(k_shape) >= 3:
                            seq_len = k_shape[2]
                        elif len(k_shape) == 2:
                            seq_len = k_shape[1]
        except Exception as exc:
            logger.debug("MLX KV cache stats error: %s", exc)

        return seq_len, total_bytes, k_shape, v_shape
