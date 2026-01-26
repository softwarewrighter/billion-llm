#!/usr/bin/env python3
"""Benchmark suite for billion-llm models."""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from datasets import load_dataset

from utils import (
    load_model,
    load_tokenizer,
    get_device,
    MODELS,
    MODEL_ALIASES,
    get_model_info,
)
from utils.memory import get_memory_usage, track_memory, clear_memory

console = Console()

# MMLU categories to test (subset for speed)
MMLU_CATEGORIES = [
    "abstract_algebra",
    "computer_security",
    "elementary_mathematics",
    "high_school_physics",
    "high_school_chemistry",
    "logical_fallacies",
    "machine_learning",
    "professional_law",
    "world_religions",
    "us_foreign_policy",
]


def benchmark_speed(
    model, tokenizer, num_runs: int = 5, prompt: str = None
) -> dict:
    """Benchmark inference speed."""
    if prompt is None:
        prompt = "Explain the concept of machine learning in simple terms. Machine learning is"

    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Warmup
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=20, do_sample=False)

    times = []
    tokens_generated = []

    for _ in range(num_runs):
        clear_memory()
        start = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        elapsed = time.time() - start
        num_new_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]

        times.append(elapsed)
        tokens_generated.append(num_new_tokens)

    avg_time = sum(times) / len(times)
    avg_tokens = sum(tokens_generated) / len(tokens_generated)

    return {
        "tokens_per_second": avg_tokens / avg_time,
        "avg_generation_time": avg_time,
        "avg_tokens_generated": avg_tokens,
        "num_runs": num_runs,
    }


def benchmark_mmlu(
    model, tokenizer, num_samples_per_category: int = 10
) -> dict:
    """Benchmark on MMLU subset."""
    console.print("  [dim]Loading MMLU dataset...[/dim]")

    correct = 0
    total = 0
    category_scores = {}

    for category in MMLU_CATEGORIES:
        try:
            dataset = load_dataset("cais/mmlu", category, split="test", trust_remote_code=True)
        except Exception as e:
            console.print(f"  [yellow]Skipping {category}: {e}[/yellow]")
            continue

        # Sample questions
        samples = list(dataset.select(range(min(num_samples_per_category, len(dataset)))))

        cat_correct = 0
        for sample in samples:
            question = sample["question"]
            choices = sample["choices"]
            answer_idx = sample["answer"]

            # Format as multiple choice
            prompt = f"Question: {question}\n"
            for i, choice in enumerate(choices):
                prompt += f"{chr(65+i)}. {choice}\n"
            prompt += "Answer with just the letter (A, B, C, or D):"

            inputs = tokenizer(prompt, return_tensors="pt")
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip().upper()

            # Extract first letter
            predicted = None
            for char in response:
                if char in "ABCD":
                    predicted = ord(char) - ord("A")
                    break

            if predicted == answer_idx:
                cat_correct += 1
                correct += 1
            total += 1

        category_scores[category] = cat_correct / len(samples) if samples else 0

    return {
        "accuracy": correct / total if total > 0 else 0,
        "correct": correct,
        "total": total,
        "category_scores": category_scores,
    }


def benchmark_memory(model, tokenizer) -> dict:
    """Benchmark memory usage."""
    mem = get_memory_usage()

    # Generate some tokens to measure peak usage
    prompt = "Write a short story about a robot learning to paint."
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with track_memory() as tracker:
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

    return {
        "model_memory_gb": mem.get("gpu_used_gb", mem.get("cpu_used_gb", 0)),
        "peak_memory_gb": tracker.get("peak_gpu_gb", mem.get("cpu_used_gb", 0)),
        "device": get_device(),
    }


def benchmark_model(
    model_name: str,
    quantize: str | None = None,
    run_mmlu: bool = True,
    run_speed: bool = True,
    run_memory: bool = True,
) -> dict:
    """Run all benchmarks for a model."""
    console.print(f"\n[bold cyan]Benchmarking: {model_name}[/bold cyan]")

    results = {
        "model": model_name,
        "info": get_model_info(model_name),
        "quantize": quantize,
        "device": get_device(),
    }

    # Load model
    console.print("  Loading model...")
    model = load_model(model_name, quantize=quantize)
    tokenizer = load_tokenizer(model_name)

    # Speed benchmark
    if run_speed:
        console.print("  Running speed benchmark...")
        results["speed"] = benchmark_speed(model, tokenizer)
        console.print(f"    [green]{results['speed']['tokens_per_second']:.1f} tokens/sec[/green]")

    # Memory benchmark
    if run_memory:
        console.print("  Running memory benchmark...")
        results["memory"] = benchmark_memory(model, tokenizer)
        console.print(f"    [green]{results['memory']['peak_memory_gb']:.2f} GB peak[/green]")

    # MMLU benchmark
    if run_mmlu:
        console.print("  Running MMLU benchmark...")
        results["mmlu"] = benchmark_mmlu(model, tokenizer)
        console.print(f"    [green]{results['mmlu']['accuracy']*100:.1f}% accuracy[/green]")

    # Cleanup
    del model
    clear_memory()

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark billion-llm models")
    parser.add_argument(
        "--model",
        type=str,
        help="Benchmark specific model (default: all)",
    )
    parser.add_argument(
        "--quantize",
        choices=["int4", "int8"],
        help="Quantization mode",
    )
    parser.add_argument(
        "--skip-mmlu",
        action="store_true",
        help="Skip MMLU benchmark (faster)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/benchmark_results.json",
        help="Output file for results",
    )
    args = parser.parse_args()

    console.print("\n[bold]billion-llm Benchmark Suite[/bold]\n")

    # Determine models to benchmark
    if args.model:
        model_name = MODEL_ALIASES.get(args.model.lower(), args.model)
        models_to_test = [model_name]
    else:
        models_to_test = MODELS

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "device": get_device(),
        "quantize": args.quantize,
        "models": {},
    }

    # Add hardware info
    if torch.cuda.is_available():
        all_results["gpu_name"] = torch.cuda.get_device_name(0)
        all_results["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    # Run benchmarks
    for model_name in models_to_test:
        try:
            results = benchmark_model(
                model_name,
                quantize=args.quantize,
                run_mmlu=not args.skip_mmlu,
            )
            all_results["models"][model_name] = results
        except Exception as e:
            console.print(f"[red]Error benchmarking {model_name}: {e}[/red]")
            all_results["models"][model_name] = {"error": str(e)}

    # Display summary table
    console.print("\n[bold]Results Summary[/bold]\n")

    table = Table()
    table.add_column("Model", style="cyan")
    table.add_column("Params", style="green")
    table.add_column("Speed (tok/s)", style="yellow")
    table.add_column("Memory (GB)", style="magenta")
    if not args.skip_mmlu:
        table.add_column("MMLU (%)", style="blue")

    for model_name, results in all_results["models"].items():
        if "error" in results:
            table.add_row(model_name, "Error", "-", "-", "-" if not args.skip_mmlu else None)
            continue

        info = results.get("info", {})
        speed = results.get("speed", {}).get("tokens_per_second", 0)
        memory = results.get("memory", {}).get("peak_memory_gb", 0)
        mmlu = results.get("mmlu", {}).get("accuracy", 0) * 100

        row = [
            model_name.split("/")[-1],
            info.get("params", "?"),
            f"{speed:.1f}",
            f"{memory:.2f}",
        ]
        if not args.skip_mmlu:
            row.append(f"{mmlu:.1f}")

        table.add_row(*row)

    console.print(table)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    console.print(f"\n[dim]Results saved to {output_path}[/dim]")


if __name__ == "__main__":
    main()
