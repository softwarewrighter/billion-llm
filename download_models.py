#!/usr/bin/env python3
"""Download all models for billion-llm benchmarks."""

import argparse
import sys

from huggingface_hub import snapshot_download, HfApi
from rich.console import Console
from rich.table import Table

from utils.model_loader import MODELS, MODEL_ALIASES, get_model_info

console = Console()


def check_disk_space(required_gb: float = 20.0) -> bool:
    """Check if there's enough disk space."""
    import shutil

    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024**3)

    if free_gb < required_gb:
        console.print(
            f"[red]Warning: Only {free_gb:.1f} GB free, recommended {required_gb} GB[/red]"
        )
        return False
    return True


def download_model(model_name: str, force: bool = False) -> bool:
    """Download a single model."""
    try:
        console.print(f"  Downloading [cyan]{model_name}[/cyan]...")

        snapshot_download(
            repo_id=model_name,
            local_files_only=not force,
            resume_download=True,
        )
        return True
    except Exception as e:
        if "local_files_only" in str(e):
            # Model not cached, need to download
            try:
                snapshot_download(
                    repo_id=model_name,
                    resume_download=True,
                )
                return True
            except Exception as e2:
                console.print(f"  [red]Error downloading {model_name}: {e2}[/red]")
                return False
        else:
            console.print(f"  [red]Error downloading {model_name}: {e}[/red]")
            return False


def check_model_access(model_name: str) -> bool:
    """Check if we have access to a model (for gated models like Llama)."""
    try:
        api = HfApi()
        api.model_info(model_name)
        return True
    except Exception as e:
        if "401" in str(e) or "403" in str(e) or "gated" in str(e).lower():
            return False
        return True  # Other errors might be network issues


def main():
    parser = argparse.ArgumentParser(description="Download billion-llm models")
    parser.add_argument(
        "--model",
        type=str,
        help="Download specific model (alias or full name)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if cached",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check which models are available",
    )
    args = parser.parse_args()

    console.print("\n[bold]billion-llm Model Downloader[/bold]\n")

    # Determine which models to download
    if args.model:
        model_name = MODEL_ALIASES.get(args.model.lower(), args.model)
        models_to_download = [model_name]
    else:
        models_to_download = MODELS

    # Check disk space
    if not args.check:
        check_disk_space()

    # Display model info
    table = Table(title="Models")
    table.add_column("Model", style="cyan")
    table.add_column("Params", style="green")
    table.add_column("Source", style="yellow")
    table.add_column("Access", style="magenta")

    for model in models_to_download:
        info = get_model_info(model)
        has_access = check_model_access(model)
        access_str = "[green]OK[/green]" if has_access else "[red]Gated[/red]"
        table.add_row(model, info["params"], info["source"], access_str)

    console.print(table)
    console.print()

    if args.check:
        console.print("[yellow]Check mode: not downloading[/yellow]")
        return

    # Check for gated models
    gated_models = [m for m in models_to_download if not check_model_access(m)]
    if gated_models:
        console.print(
            "[yellow]Note: Some models require HuggingFace authentication:[/yellow]"
        )
        for m in gated_models:
            console.print(f"  - {m}")
        console.print("\nTo access gated models:")
        console.print("  1. Create account at https://huggingface.co")
        console.print("  2. Accept model license at model page")
        console.print("  3. Run: huggingface-cli login")
        console.print()

    # Download models
    console.print("[bold]Downloading models...[/bold]\n")

    results = {}
    for model in models_to_download:
        success = download_model(model, force=args.force)
        results[model] = success
        status = "[green]OK[/green]" if success else "[red]FAILED[/red]"
        console.print(f"  {model}: {status}")

    # Summary
    console.print("\n[bold]Summary[/bold]")
    successful = sum(1 for v in results.values() if v)
    console.print(f"  Downloaded: {successful}/{len(results)} models")

    if successful < len(results):
        console.print("\n[yellow]Some models failed to download.[/yellow]")
        console.print("For gated models (like Llama), run: huggingface-cli login")
        sys.exit(1)

    console.print("\n[green]All models ready![/green]")
    console.print("\nNext steps:")
    console.print("  python benchmark.py        # Run benchmarks")
    console.print("  python demo_chat.py        # Interactive chat")


if __name__ == "__main__":
    main()
