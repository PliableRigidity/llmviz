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
    "--log-level",
    default="WARNING",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    show_default=True,
    help="Log level (for diagnostics).",
)
def run(host: str, log_level: str) -> None:
    """Start the LLMVis TUI.

    \b
    Open a second terminal and run:

        Terminal 1:  ollama run qwen2.5:3b

        Terminal 2:  llmvis run
    """
    import logging
    logging.basicConfig(level=getattr(logging, log_level.upper()))

    from llmvis.tui.app import LLMVisApp
    app = LLMVisApp(host=host)
    app.run()


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
    backend: str,
    log_level: str,
) -> None:
    """Start deep instrumentation mode with a local model.

    \b
    MODEL_ID is a HuggingFace Hub model ID or local path.
    Examples:
        llmvis instrument Qwen/Qwen2.5-1.5B-Instruct
        llmvis instrument Qwen/Qwen2.5-1.5B-Instruct --telemetry deep
        llmvis instrument Qwen/Qwen2.5-1.5B-Instruct --attention
        llmvis instrument mlx-community/Qwen2.5-1.5B-Instruct-4bit --backend mlx
        llmvis instrument /path/to/local/model

    \b
    Download a model first with:
        huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct

    \b
    Modes available in a separate terminal window:
        Mode 1 (Ollama observer):   llmvis run
        Mode 2 (deep instrument):   llmvis instrument <model_id>

    \b
    Backend selection:
        auto         — best available for current platform
        transformers — PyTorch/HuggingFace (Linux/Windows/macOS)
        mlx          — Apple MLX/Metal (Apple Silicon only)
    """
    import logging
    logging.basicConfig(level=getattr(logging, log_level.upper()))

    from llmvis.platform.detect import detect_platform
    from llmvis.tui.app_deep import LLMVisDeepApp

    platform_info = detect_platform()

    # Resolve 'auto' backend
    chosen_backend = backend.lower()
    if chosen_backend == "auto":
        if platform_info.is_apple_silicon and platform_info.mlx_available:
            chosen_backend = "mlx"
        else:
            chosen_backend = "transformers"

    click.echo("LLMVis deep instrumentation")
    click.echo(f"  Model:     {model_id}")
    click.echo(f"  Backend:   {chosen_backend}")
    click.echo(f"  Telemetry: {telemetry}")
    if chosen_backend == "transformers":
        click.echo(f"  Device:    {device}")
    if attention and chosen_backend == "transformers":
        click.echo("  Attention: enabled (--attention flag; may reduce throughput)")
    click.echo("")

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
        )

    app = LLMVisDeepApp(adapter=adapter, model_id=model_id, telemetry_mode=telemetry)
    app.run()


@cli.command()
@click.option("--host", default=DEFAULT_HOST, show_default=True, help="Ollama host URL.")
def doctor(host: str) -> None:
    """Run diagnostic checks and report environment health."""

    async def _diagnose() -> None:
        click.echo("LLMVis Doctor\n" + "─" * 40)

        # Check Ollama
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{host}/api/version")
                ver = resp.json().get("version", "?")
                click.echo(
                    f"  {click.style('✓', fg='green')} Ollama reachable at {host} (v{ver})"
                )
        except Exception as exc:
            click.echo(
                f"  {click.style('✗', fg='red')} Ollama not reachable at {host}: {exc}"
            )

        # Check psutil
        try:
            import psutil
            psutil.cpu_percent(interval=0.1)
            click.echo(
                f"  {click.style('✓', fg='green')} psutil available — CPU/RAM telemetry enabled"
            )
        except ImportError:
            click.echo(f"  {click.style('✗', fg='red')} psutil not installed")

        # Check pynvml
        try:
            import pynvml
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            names = []
            for i in range(count):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                n = pynvml.nvmlDeviceGetName(h)
                names.append(n.decode() if isinstance(n, bytes) else n)
            click.echo(
                f"  {click.style('✓', fg='green')} pynvml available — "
                f"{count} GPU(s): {', '.join(names)}"
            )
        except ImportError:
            click.echo(
                f"  {click.style('!', fg='yellow')} pynvml not installed — "
                f"no NVIDIA GPU telemetry (pip install llmvis[nvidia])"
            )
        except Exception as exc:
            click.echo(
                f"  {click.style('!', fg='yellow')} pynvml present but init failed: {exc}"
            )

        # Check Textual
        try:
            import textual
            click.echo(
                f"  {click.style('✓', fg='green')} Textual {textual.__version__} available"
            )
        except ImportError:
            click.echo(f"  {click.style('✗', fg='red')} Textual not installed")

        click.echo("\nDone.")

    asyncio.run(_diagnose())
