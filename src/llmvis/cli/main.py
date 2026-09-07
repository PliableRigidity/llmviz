"""LLMVis CLI entry points."""

from __future__ import annotations

import asyncio
import sys

import click
import httpx

# Ensure UTF-8 output on Windows terminals that default to cp1252.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

DEFAULT_HOST = "http://localhost:11434"


@click.group()
@click.version_option(package_name="llmvis")
def cli() -> None:
    """LLMVis -- live visualizer and educational debugger for local LLM inference.

    Run in a second terminal while 'ollama run <model>' runs in the first.
    """


@cli.command()
@click.option(
    "--host",
    default=DEFAULT_HOST,
    show_default=True,
    help="Ollama host URL.",
)
@click.option(
    "--port",
    default=7654,
    type=int,
    show_default=True,
    help="Port to probe for a deep instrumentation session.",
)
@click.option(
    "--log-level",
    default="WARNING",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    show_default=True,
    help="Log level (for diagnostics).",
)
def run(host: str, port: int, log_level: str) -> None:
    """Start the LLMVis visualizer TUI.

    \b
    Auto-discovers the best available telemetry source:
      1. Deep instrumentation session (llmvis instrument running in Terminal 1)
      2. Ollama observer (ollama run <model> in Terminal 1)
      3. Waiting screen (probes every 2 seconds until a source appears)

    \b
    Two-terminal usage:
        Terminal 1:  llmvis instrument Qwen/Qwen2.5-1.5B-Instruct
        Terminal 2:  llmvis run
    """
    import logging
    logging.basicConfig(level=getattr(logging, log_level.upper()))

    from llmvis.tui.waiting_app import probe_source

    while True:
        source = probe_source(ollama_host=host, deep_port=port)

        if source == "deep":
            from llmvis.adapters.client_adapter import ClientAdapter
            from llmvis.tui.app_deep import LLMVisDeepApp
            adapter = ClientAdapter(host="127.0.0.1", port=port)
            app = LLMVisDeepApp(adapter=adapter, model_id="", telemetry_mode="standard")
            app.run()
            break

        elif source == "ollama":
            from llmvis.tui.app import LLMVisApp
            app = LLMVisApp(host=host)
            app.run()
            break

        else:
            # No source available — show waiting screen
            from llmvis.tui.waiting_app import WaitingApp
            waiting = WaitingApp(ollama_host=host, deep_port=port)
            result = waiting.run()
            if result is None:
                # User pressed Q in the waiting screen
                break
            # result is 'deep' or 'ollama' — loop and launch the right TUI


@cli.command()
@click.option("--host", default=DEFAULT_HOST, show_default=True, help="Ollama host URL.")
def status(host: str) -> None:
    """Print current Ollama status and loaded models."""

    async def _check() -> None:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                ver_resp = await client.get(f"{host}/api/version")
                version = ver_resp.json().get("version", "?")
                click.echo(
                    f"Ollama  {click.style('CONNECTED', fg='green')}  v{version}  ({host})"
                )

                ps_resp = await client.get(f"{host}/api/ps")
                models = ps_resp.json().get("models", [])
                if models:
                    click.echo(f"\nLoaded models ({len(models)}):")
                    for m in models:
                        d = m.get("details", {})
                        click.echo(
                            f"  {m['name']}  {d.get('quantization_level', '')}  "
                            f"ctx {m.get('context_length', '?')}"
                        )
                else:
                    click.echo("\nNo models currently loaded.")

                tags_resp = await client.get(f"{host}/api/tags")
                all_models = tags_resp.json().get("models", [])
                click.echo(f"\n{len(all_models)} model(s) installed.")

        except Exception as exc:
            click.echo(
                f"Ollama  {click.style('DISCONNECTED', fg='red')}  ({host})\n"
                f"Error: {exc}"
            )
            sys.exit(1)

    asyncio.run(_check())


@cli.command()
@click.argument("model_id")
@click.option(
    "--telemetry",
    default="standard",
    type=click.Choice(["light", "standard", "deep"], case_sensitive=False),
    show_default=True,
    help="Instrumentation depth: light (timing+logits), standard (+layer stats), deep (+MLP+attention).",
)
@click.option(
    "--attention",
    is_flag=True,
    default=False,
    help="Enable attention weight statistics (deep mode only; may increase VRAM and reduce speed).",
)
@click.option("--device", default="auto", show_default=True, help="Inference device: cuda, cpu, mps, or auto.")
@click.option("--max-new-tokens", default=512, type=int, show_default=True, help="Max tokens per prompt.")
@click.option("--temperature", default=0.7, type=float, show_default=True)
@click.option("--top-k", default=50, type=int, show_default=True)
@click.option("--top-p", default=0.9, type=float, show_default=True)
@click.option(
    "--record",
    default=None,
    type=click.Path(dir_okay=False, writable=True),
    help="Record telemetry to a JSONL file for later replay with 'llmvis replay'.",
)
@click.option(
    "--backend",
    default="auto",
    type=click.Choice(["auto", "transformers", "mlx"], case_sensitive=False),
    show_default=True,
    help=(
        "Inference backend. 'auto' selects the best available backend for the platform. "
        "'transformers' forces PyTorch/HuggingFace. "
        "'mlx' forces Apple MLX (requires Apple Silicon + mlx-lm installed)."
    ),
)
@click.option(
    "--port",
    default=7654,
    type=int,
    show_default=True,
    help="TCP port for the telemetry server (connect with 'llmvis run --port <N>').",
)
@click.option(
    "--log-level",
    default="WARNING",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    show_default=True,
)
def instrument(
    model_id: str,
    telemetry: str,
    attention: bool,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    record: str | None,
    backend: str,
    port: int,
    log_level: str,
) -> None:
    """Load a local model and start a headless telemetry server.

    \b
    MODEL_ID is a HuggingFace Hub model ID or local path.

    \b
    Two-terminal usage:
        Terminal 1:  llmvis instrument Qwen/Qwen2.5-1.5B-Instruct
        Terminal 2:  llmvis run

    \b
    The visualizer (Terminal 2) auto-connects to the telemetry server on port 7654.
    Type prompts at the Terminal 1 REPL: > your prompt here

    \b
    Backend selection:
        auto         — best available for current platform
        transformers — PyTorch/HuggingFace (Linux/Windows/macOS)
        mlx          — Apple MLX/Metal (Apple Silicon only)
    """
    import logging
    import platform as _platform
    logging.basicConfig(level=getattr(logging, log_level.upper()))

    from llmvis.platform.detect import detect_platform

    platform_info = detect_platform()

    # Resolve 'auto' backend
    chosen_backend = backend.lower()
    if chosen_backend == "auto":
        if platform_info.is_apple_silicon and platform_info.mlx_available:
            chosen_backend = "mlx"
        else:
            chosen_backend = "transformers"

    click.echo("LLMVis deep instrumentation — headless telemetry server")
    click.echo(f"  Model:     {model_id}")
    click.echo(f"  Backend:   {chosen_backend}")
    click.echo(f"  Telemetry: {telemetry}")
    click.echo(f"  Port:      {port}")
    if chosen_backend == "transformers":
        click.echo(f"  Device:    {device}")
    if attention and chosen_backend == "transformers":
        click.echo("  Attention: enabled")
    if record:
        click.echo(f"  Record:    {record}")
    click.echo("")
    click.echo("In a second terminal run:  llmvis run")
    click.echo("")

    import uuid
    session_id = str(uuid.uuid4())[:8]

    if chosen_backend == "mlx":
        from llmvis.adapters.mlx_adapter import MLXAdapter
        try:
            adapter = MLXAdapter(
                model_id=model_id,
                telemetry_mode=telemetry,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        except RuntimeError as exc:
            click.echo(f"Error: {exc}", err=True)
            raise SystemExit(1) from exc
    else:
        from llmvis.adapters.instrumented_transformers import InstrumentedTransformersAdapter
        adapter = InstrumentedTransformersAdapter(
            model_id=model_id,
            device=device,
            telemetry_mode=telemetry,
            attention_enabled=attention,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            record_path=record,
            session_id=session_id,
        )

    asyncio.run(
        _serve(
            adapter=adapter,
            model_id=model_id,
            chosen_backend=chosen_backend,
            telemetry_mode=telemetry,
            port=port,
            session_id=session_id,
            platform_name=_platform.system(),
        )
    )


async def _serve(
    adapter,
    model_id: str,
    chosen_backend: str,
    telemetry_mode: str,
    port: int,
    session_id: str,
    platform_name: str,
) -> None:
    """Run the adapter event loop + telemetry server + interactive REPL."""
    from llmvis.ipc.server import TelemetryServer

    # ── Step 1: Load model BEFORE starting server or REPL ─────────────────────
    # This ensures: (a) tqdm output finishes before the > prompt appears;
    # (b) the server is only advertised when inference is actually ready;
    # (c) Terminal 2 can connect the moment it sees "Telemetry server ready".
    click.echo("Loading model (this may take a moment)…")
    try:
        if hasattr(adapter, "load_model"):
            await adapter.load_model()
        device = getattr(adapter, "_actual_device", "unknown")
        click.echo(f"Model ready on {device}.\n")
    except Exception as exc:
        click.echo(
            f"\nFailed to load model: {type(exc).__name__}: {exc}",
            err=True,
        )
        raise SystemExit(1) from exc

    # ── Step 2: Start telemetry server ─────────────────────────────────────────
    server = TelemetryServer(
        port=port,
        session_id=session_id,
        model_id=model_id,
        backend=chosen_backend,
        telemetry_mode=telemetry_mode,
        platform_name=platform_name,
    )

    def _on_prompt(text: str) -> None:
        asyncio.create_task(adapter.submit_prompt(text))

    server.add_prompt_callback(_on_prompt)
    try:
        await server.start()
    except OSError as exc:
        click.echo(
            f"\nCannot start telemetry server on port {port}: {exc}\n"
            "Is another 'llmvis instrument' already running?",
            err=True,
        )
        raise SystemExit(1) from exc

    click.echo(f"Telemetry server ready on 127.0.0.1:{port}")
    click.echo("In a second terminal run:  llmvis run\n")
    click.echo("Type prompts below (Ctrl+C or /quit to exit):\n")

    # ── Step 3: Adapter event loop — broadcasts and prints generated text ───────
    # _generation_done: set = idle, clear = generation in progress.
    # REPL blocks on this so the next > prompt only appears after the full
    # response is printed (same UX as `ollama run`).
    _generation_done = asyncio.Event()
    _generation_done.set()  # start idle

    async def _adapter_loop() -> None:
        from llmvis.core.events import EventType
        try:
            async for evt in adapter.run():
                await server.broadcast(evt)
                t = evt.type
                if t == EventType.INFERENCE_START:
                    _generation_done.clear()
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                elif t == EventType.TOKEN_GENERATED:
                    sys.stdout.write(evt.token_text)
                    sys.stdout.flush()
                elif t == EventType.INFERENCE_END:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    _generation_done.set()
        except Exception as exc:
            click.echo(
                f"\nAdapter error: {type(exc).__name__}: {exc}", err=True
            )
        finally:
            _generation_done.set()  # always unblock REPL on any exit path

    adapter_task = asyncio.create_task(_adapter_loop())

    # ── Step 4: Interactive REPL ───────────────────────────────────────────────
    loop = asyncio.get_event_loop()
    try:
        while True:
            # Wait for any in-progress generation before showing the prompt.
            # 120-second safety timeout guards against INFERENCE_END never arriving.
            try:
                await asyncio.wait_for(_generation_done.wait(), timeout=120.0)
            except asyncio.TimeoutError:
                _generation_done.set()  # unstick

            try:
                prompt_text = await loop.run_in_executor(None, _read_prompt)
            except (EOFError, KeyboardInterrupt):
                break
            if prompt_text is None:
                break  # Ctrl+C or EOF
            if prompt_text.strip().lower() in ("/quit", "/exit"):
                break
            if prompt_text.strip():
                _generation_done.clear()  # expect INFERENCE_END to re-set this
                await adapter.submit_prompt(prompt_text.strip())
            # empty prompt: loop back and show > again
    finally:
        adapter_task.cancel()
        try:
            await adapter_task
        except asyncio.CancelledError:
            pass
        await adapter.stop()
        await server.stop()
        click.echo("\nInstrumentation session ended.")


def _read_prompt() -> str | None:
    """Read one line from stdin. Returns None on Ctrl+C or EOF (to quit REPL)."""
    try:
        return input("> ")
    except (EOFError, KeyboardInterrupt):
        return None


@cli.command()
@click.option("--host", default=DEFAULT_HOST, show_default=True, help="Ollama host URL.")
def doctor(host: str) -> None:
    """Run diagnostic checks and report environment health."""
    import platform as _platform

    OK = click.style("✓", fg="green")
    WARN = click.style("!", fg="yellow")
    FAIL = click.style("✗", fg="red")
    NA = click.style("─", fg="bright_black")

    def row(symbol: str, label: str, detail: str) -> None:
        click.echo(f"  {symbol} {label:<24} {detail}")

    click.echo("LLMVis Doctor\n" + "─" * 50)

    # Python
    py_ver = _platform.python_version()
    row(OK, "Python", py_ver)

    # Platform
    system = _platform.system()
    machine = _platform.machine()
    row(OK, "Platform", f"{system} {machine}")

    # Apple Silicon / platform detection
    try:
        from llmvis.platform.detect import detect_platform
        pinfo = detect_platform()
        if pinfo.is_apple_silicon:
            row(OK, "Apple Silicon", pinfo.chip_model)
            row(OK if pinfo.mlx_available else WARN,
                "MLX",
                "installed" if pinfo.mlx_available else "not installed (pip install llmvis[mlx])")
            row(OK if pinfo.mps_available else NA, "PyTorch MPS",
                "available" if pinfo.mps_available else "not available")
        elif pinfo.cuda_available:
            row(OK, "CUDA GPU", pinfo.cuda_device_name)
            row(NA, "MLX", "not applicable (not Apple Silicon)")
        else:
            row(NA, "GPU acceleration", "none detected (CPU-only mode)")
            row(NA, "MLX", "not applicable (not Apple Silicon)")
    except Exception as exc:
        row(WARN, "Platform detection", f"error: {exc}")

    # PyTorch
    try:
        import torch
        cuda_str = f"CUDA {torch.version.cuda}" if torch.cuda.is_available() else "CPU only"
        row(OK, "PyTorch", f"{torch.__version__} ({cuda_str})")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory
                row(OK, f"  GPU {i}", f"{name} ({mem // 1024**3} GB)")
    except ImportError:
        row(WARN, "PyTorch", "not installed (required for deep instrumentation)")

    # HuggingFace Transformers
    try:
        import transformers
        row(OK, "Transformers", transformers.__version__)
    except ImportError:
        row(WARN, "Transformers", "not installed (required for deep instrumentation)")

    # psutil
    try:
        import psutil
        mem = psutil.virtual_memory()
        row(OK, "psutil", f"{psutil.__version__} ({mem.total // 1024**3} GB RAM)")
    except ImportError:
        row(FAIL, "psutil", "not installed (required)")

    # pynvml
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        names = []
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            n = pynvml.nvmlDeviceGetName(h)
            names.append(n.decode() if isinstance(n, bytes) else n)
        row(OK, "pynvml", f"{count} GPU(s): {', '.join(names)}")
    except ImportError:
        row(WARN, "pynvml", "not installed — no NVIDIA telemetry (pip install llmvis[nvidia])")
    except Exception as exc:
        row(WARN, "pynvml", f"init failed: {exc}")

    # Textual
    try:
        import textual
        row(OK, "Textual TUI", textual.__version__)
    except ImportError:
        row(FAIL, "Textual", "not installed (required)")

    # Deep instrumentation readiness
    click.echo("")
    _check_deep_available = True
    try:
        import torch  # noqa: F811
        import transformers  # noqa: F811
        row(OK, "Deep instrumentation", "available (llmvis instrument <model>)")
    except ImportError:
        row(WARN, "Deep instrumentation",
            "PyTorch and/or Transformers not installed; run: pip install torch transformers")
        _check_deep_available = False

    # Deep IPC endpoint probe
    click.echo("")
    async def _check_deep_ipc() -> None:
        deep_port = 7654
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection("127.0.0.1", deep_port), timeout=0.5
            )
            writer.close()
            await writer.wait_closed()
            row(OK, "Deep IPC endpoint", f"127.0.0.1:{deep_port} — server running")
        except Exception:
            row(NA, "Deep IPC endpoint", f"127.0.0.1:{deep_port} — no server (run llmvis instrument <model>)")

    asyncio.run(_check_deep_ipc())

    async def _check_ollama() -> None:
        click.echo("")
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{host}/api/version")
                ver = resp.json().get("version", "?")
                row(OK, "Ollama", f"v{ver} at {host}")
                ps = await client.get(f"{host}/api/ps")
                models = ps.json().get("models", [])
                if models:
                    for m in models:
                        row(OK, "  Model loaded", m["name"])
                else:
                    row(NA, "Ollama", "no model currently loaded")
        except Exception as exc:
            row(WARN, "Ollama", f"not reachable at {host} ({exc})")
            row(NA, "", "Run 'ollama serve' and 'ollama pull <model>' to enable observer mode")

    asyncio.run(_check_ollama())
    click.echo("\n" + "─" * 50)
    click.echo("Run 'llmvis instrument <model>' to start deep instrumentation.")
    click.echo("Run 'llmvis run' to start the visualizer (auto-detects source).")
    click.echo("Run 'llmvis demo' to see a pre-recorded session without a model.")


@cli.command()
@click.argument("recording", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--speed",
    default=1.0,
    type=float,
    show_default=True,
    help="Playback speed multiplier (1.0 = real time, 2.0 = 2× faster, 0 = instant).",
)
@click.option(
    "--log-level",
    default="WARNING",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    show_default=True,
)
def replay(recording: str, speed: float, log_level: str) -> None:
    """Replay a recorded telemetry session in the TUI.

    \b
    Record a session first with:
        llmvis instrument <model> --record session.jsonl

    \b
    Then replay it:
        llmvis replay session.jsonl
        llmvis replay session.jsonl --speed 2.0
        llmvis replay session.jsonl --speed 0   (instant, no delays)
    """
    import logging
    logging.basicConfig(level=getattr(logging, log_level.upper()))

    from llmvis.adapters.replay_adapter import ReplayAdapter
    from llmvis.tui.app_deep import LLMVisDeepApp

    adapter = ReplayAdapter(path=recording, speed=speed, loop=False)
    app = LLMVisDeepApp(
        adapter=adapter,
        model_id=f"[replay] {recording}",
        telemetry_mode="standard",
    )
    app.run()


@cli.command()
@click.option(
    "--speed",
    default=1.0,
    type=float,
    show_default=True,
    help="Playback speed multiplier.",
)
@click.option("--loop", is_flag=True, default=False, help="Loop the demo continuously.")
def demo(speed: float, loop: bool) -> None:
    """Show a pre-recorded demo session without downloading a model.

    \b
    The demo plays back a real instrumentation session captured from a small
    language model. All telemetry is real — it was measured during actual inference.

    \b
    The TUI is clearly labelled:
        MODE: DEMO / RECORDED TELEMETRY

    \b
    To record your own session:
        llmvis instrument <model> --record my_session.jsonl
    Then replay it:
        llmvis replay my_session.jsonl
    """
    import logging
    from pathlib import Path

    logging.basicConfig(level=logging.WARNING)

    demo_path = Path(__file__).parent.parent / "demo" / "demo_session.jsonl"
    if not demo_path.exists():
        click.echo(
            f"Demo file not found at {demo_path}.\n"
            "Record one with: llmvis instrument <model> --record demo_session.jsonl\n"
            "Then move it to src/llmvis/demo/demo_session.jsonl",
            err=True,
        )
        raise SystemExit(1)

    from llmvis.adapters.replay_adapter import ReplayAdapter
    from llmvis.tui.app_deep import LLMVisDeepApp

    click.echo("LLMVis Demo — playing back recorded session")
    click.echo(
        "  [DEMO / RECORDED TELEMETRY — not live inference]\n"
    )

    adapter = ReplayAdapter(path=demo_path, speed=speed, loop=loop)
    app = LLMVisDeepApp(
        adapter=adapter,
        model_id="[DEMO] Recorded session",
        telemetry_mode="standard",
    )
    app.run()
