"""LLMVis CLI entry points."""

from __future__ import annotations

import asyncio
import sys

import click
import httpx

DEFAULT_HOST = "http://localhost:11434"


@click.group()
@click.version_option(package_name="llmvis")
def cli() -> None:
    """LLMVis — live visualizer and educational debugger for local LLM inference.

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
