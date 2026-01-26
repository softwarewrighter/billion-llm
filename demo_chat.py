#!/usr/bin/env python3
"""Interactive chat demo for billion-llm models."""

import argparse
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live

from utils import load_model, load_tokenizer, get_device, MODEL_ALIASES, get_memory_usage

console = Console()


def generate_response(
    model,
    tokenizer,
    messages: list[dict],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    do_sample: bool = True,
) -> tuple[str, float, int]:
    """
    Generate a response from the model.

    Returns:
        Tuple of (response_text, elapsed_time, num_tokens)
    """
    # Apply chat template
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        # Fallback for models without chat template
        prompt = "\n".join(
            f"{m['role'].title()}: {m['content']}" for m in messages
        )
        prompt += "\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt")

    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    start_time = time.time()

    with console.status("[bold green]Generating..."):
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    elapsed = time.time() - start_time

    # Decode only the new tokens
    input_len = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][input_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return response.strip(), elapsed, len(new_tokens)


def chat_single(model_name: str, quantize: str | None = None):
    """Interactive chat with a single model."""
    console.print(f"\n[bold]Loading {model_name}...[/bold]")

    model = load_model(model_name, quantize=quantize)
    tokenizer = load_tokenizer(model_name)

    device = get_device()
    mem = get_memory_usage()
    console.print(f"[dim]Device: {device}, Memory: {mem.get('gpu_used_gb', mem.get('cpu_used_gb', 0)):.2f} GB[/dim]")

    console.print(f"\n[green]Chat with {model_name}[/green]")
    console.print("[dim]Type 'quit' to exit, 'clear' to reset conversation[/dim]\n")

    messages = []

    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ")
        except (KeyboardInterrupt, EOFError):
            break

        if user_input.lower() == "quit":
            break
        if user_input.lower() == "clear":
            messages = []
            console.print("[dim]Conversation cleared[/dim]\n")
            continue
        if not user_input.strip():
            continue

        messages.append({"role": "user", "content": user_input})

        response, elapsed, num_tokens = generate_response(model, tokenizer, messages)

        messages.append({"role": "assistant", "content": response})

        tokens_per_sec = num_tokens / elapsed if elapsed > 0 else 0

        console.print(f"\n[bold green]Assistant:[/bold green] {response}")
        console.print(
            f"[dim]{num_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)[/dim]\n"
        )

    console.print("\n[dim]Goodbye![/dim]")


def chat_compare(model_names: list[str], quantize: str | None = None):
    """Side-by-side comparison chat with multiple models."""
    console.print(f"\n[bold]Loading {len(model_names)} models for comparison...[/bold]")

    models = {}
    tokenizers = {}

    for name in model_names:
        console.print(f"  Loading {name}...")
        models[name] = load_model(name, quantize=quantize)
        tokenizers[name] = load_tokenizer(name)

    console.print(f"\n[green]Comparing: {', '.join(model_names)}[/green]")
    console.print("[dim]Type 'quit' to exit[/dim]\n")

    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ")
        except (KeyboardInterrupt, EOFError):
            break

        if user_input.lower() == "quit":
            break
        if not user_input.strip():
            continue

        messages = [{"role": "user", "content": user_input}]

        # Create comparison table
        table = Table(title="Model Responses", show_lines=True)
        table.add_column("Model", style="cyan", width=15)
        table.add_column("Response", style="white", width=60)
        table.add_column("Speed", style="green", width=12)

        for name in model_names:
            response, elapsed, num_tokens = generate_response(
                models[name], tokenizers[name], messages
            )
            tokens_per_sec = num_tokens / elapsed if elapsed > 0 else 0

            # Truncate long responses for display
            display_response = response[:500] + "..." if len(response) > 500 else response

            table.add_row(
                name.split("/")[-1][:15],
                display_response,
                f"{tokens_per_sec:.1f} tok/s",
            )

        console.print(table)
        console.print()

    console.print("\n[dim]Goodbye![/dim]")


def main():
    parser = argparse.ArgumentParser(description="Chat with billion-llm models")
    parser.add_argument(
        "--model",
        type=str,
        default="tinyllama",
        help="Model to chat with (default: tinyllama)",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        help="Compare multiple models side-by-side",
    )
    parser.add_argument(
        "--quantize",
        choices=["int4", "int8"],
        help="Quantization mode for lower memory usage",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models",
    )
    args = parser.parse_args()

    if args.list:
        console.print("\n[bold]Available Models[/bold]\n")
        table = Table()
        table.add_column("Alias", style="cyan")
        table.add_column("Full Name", style="white")

        for alias, full_name in MODEL_ALIASES.items():
            table.add_row(alias, full_name)

        console.print(table)
        return

    console.print(Panel.fit(
        "[bold]billion-llm Chat Demo[/bold]\n"
        "The 1B Sweet Spot: Compare billion-parameter models",
        border_style="blue",
    ))

    if args.compare:
        chat_compare(args.compare, quantize=args.quantize)
    else:
        chat_single(args.model, quantize=args.quantize)


if __name__ == "__main__":
    main()
