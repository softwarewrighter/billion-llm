#!/usr/bin/env python3
"""Speculative decoding demo for billion-llm models.

Demonstrates using a fast 1B model as a draft generator to accelerate
inference from a larger target model.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress

from utils import load_model, load_tokenizer, get_device, MODEL_ALIASES
from utils.memory import clear_memory

console = Console()

# Default model configurations
DEFAULT_DRAFT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_TARGET_MODEL = "meta-llama/Llama-3.2-3B-Instruct"


def generate_baseline(
    model, tokenizer, prompt: str, max_new_tokens: int = 100
) -> tuple[str, float, int]:
    """Generate using standard autoregressive decoding."""
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    elapsed = time.time() - start_time

    new_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    return response, elapsed, new_tokens


def speculative_decode(
    draft_model,
    target_model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    k: int = 4,
) -> tuple[str, float, int, dict]:
    """
    Generate using speculative decoding.

    Args:
        draft_model: Fast draft model (1B)
        target_model: Slower target model (3B+)
        tokenizer: Shared tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        k: Number of draft tokens per iteration

    Returns:
        (response, elapsed_time, num_tokens, stats)
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    draft_device = next(draft_model.parameters()).device
    target_device = next(target_model.parameters()).device

    input_ids = inputs["input_ids"].to(draft_device)
    attention_mask = inputs["attention_mask"].to(draft_device)

    generated_tokens = []
    stats = {
        "draft_iterations": 0,
        "tokens_accepted": 0,
        "tokens_rejected": 0,
        "acceptance_rate": 0,
    }

    start_time = time.time()

    with torch.no_grad():
        while len(generated_tokens) < max_new_tokens:
            # Step 1: Generate k draft tokens
            draft_outputs = draft_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=k,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

            draft_tokens = draft_outputs.sequences[0, input_ids.shape[1]:]
            num_draft = len(draft_tokens)

            if num_draft == 0:
                break

            # Step 2: Verify with target model
            # Create sequence with draft tokens for verification
            verify_ids = torch.cat([input_ids, draft_tokens.unsqueeze(0).to(draft_device)], dim=1)
            verify_mask = torch.ones_like(verify_ids)

            # Get target model logits for all positions
            target_inputs = verify_ids.to(target_device)
            target_outputs = target_model(target_inputs, attention_mask=verify_mask.to(target_device))
            target_logits = target_outputs.logits

            # Step 3: Accept/reject tokens
            accepted = 0
            for i in range(num_draft):
                pos = input_ids.shape[1] + i - 1  # Position in target logits
                if pos < 0:
                    pos = 0

                # Get target's prediction for this position
                target_pred = target_logits[0, pos].argmax().item()
                draft_token = draft_tokens[i].item()

                # Simple acceptance: accept if target agrees with draft
                if target_pred == draft_token:
                    accepted += 1
                    generated_tokens.append(draft_token)
                else:
                    # Reject and use target's token instead
                    generated_tokens.append(target_pred)
                    stats["tokens_rejected"] += 1
                    break

            stats["tokens_accepted"] += accepted
            stats["draft_iterations"] += 1

            # Update input for next iteration
            new_tokens = torch.tensor(
                generated_tokens[-accepted-1:] if accepted < num_draft else generated_tokens[-accepted:],
                device=draft_device
            ).unsqueeze(0)

            if len(new_tokens[0]) > 0:
                input_ids = torch.cat([input_ids, new_tokens], dim=1)
                attention_mask = torch.ones_like(input_ids)

            # Check for EOS
            if tokenizer.eos_token_id in generated_tokens[-k:]:
                break

    elapsed = time.time() - start_time

    # Calculate final stats
    total_tokens = stats["tokens_accepted"] + stats["tokens_rejected"]
    if total_tokens > 0:
        stats["acceptance_rate"] = stats["tokens_accepted"] / total_tokens

    # Decode response
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return response, elapsed, len(generated_tokens), stats


def main():
    parser = argparse.ArgumentParser(description="Speculative decoding demo")
    parser.add_argument(
        "--draft-model",
        type=str,
        default="tinyllama",
        help="Draft model (fast, smaller)",
    )
    parser.add_argument(
        "--target-model",
        type=str,
        default=None,
        help="Target model (slower, larger). If not specified, compares draft model baseline only.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of draft tokens per iteration (default: 4)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain the theory of relativity in simple terms.",
        help="Prompt to generate from",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate (default: 100)",
    )
    parser.add_argument(
        "--quantize",
        choices=["int4", "int8"],
        help="Quantization mode for models",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/speculative_results.json",
        help="Output file for results",
    )
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold]billion-llm Speculative Decoding Demo[/bold]\n"
        "Use fast 1B models to accelerate larger model inference",
        border_style="blue",
    ))

    draft_model_name = MODEL_ALIASES.get(args.draft_model.lower(), args.draft_model)

    # Load draft model
    console.print(f"\n[bold]Loading draft model: {draft_model_name}[/bold]")
    draft_model = load_model(draft_model_name, quantize=args.quantize)
    tokenizer = load_tokenizer(draft_model_name)

    results = {
        "timestamp": datetime.now().isoformat(),
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "k": args.k,
        "draft_model": draft_model_name,
        "device": get_device(),
    }

    # Baseline: draft model alone
    console.print("\n[bold]Running draft model baseline...[/bold]")
    baseline_response, baseline_time, baseline_tokens = generate_baseline(
        draft_model, tokenizer, args.prompt, args.max_tokens
    )

    baseline_speed = baseline_tokens / baseline_time if baseline_time > 0 else 0
    console.print(f"  [green]{baseline_speed:.1f} tokens/sec ({baseline_tokens} tokens in {baseline_time:.2f}s)[/green]")

    results["draft_baseline"] = {
        "response": baseline_response,
        "time": baseline_time,
        "tokens": baseline_tokens,
        "tokens_per_second": baseline_speed,
    }

    # If target model specified, run speculative decoding
    if args.target_model:
        target_model_name = MODEL_ALIASES.get(args.target_model.lower(), args.target_model)

        console.print(f"\n[bold]Loading target model: {target_model_name}[/bold]")
        target_model = load_model(target_model_name, quantize=args.quantize)

        # Target model baseline
        console.print("\n[bold]Running target model baseline...[/bold]")
        target_response, target_time, target_tokens = generate_baseline(
            target_model, tokenizer, args.prompt, args.max_tokens
        )

        target_speed = target_tokens / target_time if target_time > 0 else 0
        console.print(f"  [green]{target_speed:.1f} tokens/sec ({target_tokens} tokens in {target_time:.2f}s)[/green]")

        results["target_model"] = target_model_name
        results["target_baseline"] = {
            "response": target_response,
            "time": target_time,
            "tokens": target_tokens,
            "tokens_per_second": target_speed,
        }

        # Speculative decoding
        console.print(f"\n[bold]Running speculative decoding (k={args.k})...[/bold]")
        spec_response, spec_time, spec_tokens, spec_stats = speculative_decode(
            draft_model, target_model, tokenizer,
            args.prompt, args.max_tokens, k=args.k
        )

        spec_speed = spec_tokens / spec_time if spec_time > 0 else 0
        speedup = spec_speed / target_speed if target_speed > 0 else 0

        console.print(f"  [green]{spec_speed:.1f} tokens/sec ({spec_tokens} tokens in {spec_time:.2f}s)[/green]")
        console.print(f"  [cyan]Acceptance rate: {spec_stats['acceptance_rate']*100:.1f}%[/cyan]")
        console.print(f"  [cyan]Speedup vs target: {speedup:.2f}x[/cyan]")

        results["speculative"] = {
            "response": spec_response,
            "time": spec_time,
            "tokens": spec_tokens,
            "tokens_per_second": spec_speed,
            "stats": spec_stats,
            "speedup": speedup,
        }

    # Display summary
    console.print("\n[bold]Summary[/bold]\n")

    table = Table()
    table.add_column("Method", style="cyan")
    table.add_column("Speed (tok/s)", style="green")
    table.add_column("Time (s)", style="yellow")
    table.add_column("Tokens", style="magenta")

    table.add_row(
        f"Draft ({draft_model_name.split('/')[-1]})",
        f"{baseline_speed:.1f}",
        f"{baseline_time:.2f}",
        str(baseline_tokens),
    )

    if args.target_model:
        table.add_row(
            f"Target ({target_model_name.split('/')[-1]})",
            f"{target_speed:.1f}",
            f"{target_time:.2f}",
            str(target_tokens),
        )
        table.add_row(
            f"Speculative (k={args.k})",
            f"{spec_speed:.1f}",
            f"{spec_time:.2f}",
            str(spec_tokens),
        )

    console.print(table)

    # Display responses
    console.print("\n[bold]Generated Responses[/bold]\n")

    console.print(f"[cyan]Draft Model:[/cyan]")
    console.print(f"  {baseline_response[:300]}{'...' if len(baseline_response) > 300 else ''}\n")

    if args.target_model:
        console.print(f"[cyan]Target Model:[/cyan]")
        console.print(f"  {target_response[:300]}{'...' if len(target_response) > 300 else ''}\n")

        console.print(f"[cyan]Speculative:[/cyan]")
        console.print(f"  {spec_response[:300]}{'...' if len(spec_response) > 300 else ''}\n")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    console.print(f"\n[dim]Results saved to {output_path}[/dim]")

    if not args.target_model:
        console.print("\n[yellow]Tip: Add --target-model to compare with speculative decoding[/yellow]")
        console.print("[yellow]Example: python speculative_demo.py --target-model stablelm[/yellow]")


if __name__ == "__main__":
    main()
