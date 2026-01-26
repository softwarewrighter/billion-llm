#!/usr/bin/env python3
"""LoRA fine-tuning demo for billion-llm models."""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from trl import SFTTrainer, SFTConfig

from utils import load_model, load_tokenizer, get_device, MODEL_ALIASES
from utils.memory import track_memory, clear_memory, get_memory_usage

console = Console()

# Sample training data (alpaca-style)
SAMPLE_DATA = [
    {
        "instruction": "Explain what a neural network is.",
        "input": "",
        "output": "A neural network is a computing system inspired by biological neural networks in the brain. It consists of layers of interconnected nodes (neurons) that process information. Each connection has a weight that adjusts during training. Neural networks learn patterns from data and can make predictions or decisions without explicit programming for specific tasks.",
    },
    {
        "instruction": "Write a haiku about programming.",
        "input": "",
        "output": "Code flows like water\nBugs hide in the deepest depths\nDebug, rinse, repeat",
    },
    {
        "instruction": "Translate this sentence to French.",
        "input": "The cat is sleeping on the couch.",
        "output": "Le chat dort sur le canapé.",
    },
    {
        "instruction": "Summarize the benefits of exercise.",
        "input": "",
        "output": "Exercise improves cardiovascular health, boosts mood through endorphin release, helps maintain healthy weight, strengthens muscles and bones, enhances sleep quality, and reduces risk of chronic diseases like diabetes and heart disease.",
    },
    {
        "instruction": "What is the capital of Australia?",
        "input": "",
        "output": "The capital of Australia is Canberra. While Sydney and Melbourne are larger and more well-known cities, Canberra was purpose-built as the capital in the early 20th century as a compromise between the two rival cities.",
    },
    {
        "instruction": "Explain recursion to a beginner.",
        "input": "",
        "output": "Recursion is when a function calls itself to solve a problem. Think of it like Russian nesting dolls - each doll contains a smaller version of itself. In programming, a recursive function breaks a big problem into smaller identical problems until reaching a simple base case it can solve directly.",
    },
    {
        "instruction": "Write a short poem about the moon.",
        "input": "",
        "output": "Silver orb in velvet night,\nGuiding sailors with your light.\nWaxing, waning, ever true,\nAncient friend in midnight blue.",
    },
    {
        "instruction": "What causes rainbows?",
        "input": "",
        "output": "Rainbows form when sunlight enters water droplets in the air. The light bends (refracts) as it enters the droplet, splits into its component colors, reflects off the back of the droplet, and bends again as it exits. This separates white light into the visible spectrum: red, orange, yellow, green, blue, indigo, and violet.",
    },
    {
        "instruction": "Convert this temperature to Fahrenheit.",
        "input": "25 degrees Celsius",
        "output": "25 degrees Celsius equals 77 degrees Fahrenheit. The formula is F = (C × 9/5) + 32, so (25 × 9/5) + 32 = 45 + 32 = 77°F.",
    },
    {
        "instruction": "Explain the difference between a list and a tuple in Python.",
        "input": "",
        "output": "Lists and tuples are both sequences in Python, but lists are mutable (can be changed after creation) while tuples are immutable (cannot be modified). Lists use square brackets [], tuples use parentheses (). Tuples are slightly faster and can be used as dictionary keys, while lists are better when you need to modify the collection.",
    },
]


def format_prompt(example: dict) -> str:
    """Format a training example as a prompt."""
    if example.get("input"):
        return f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
    else:
        return f"""### Instruction:
{example['instruction']}

### Response:
{example['output']}"""


def evaluate_model(model, tokenizer, test_prompts: list[str]) -> list[str]:
    """Generate responses for test prompts."""
    responses = []
    device = next(model.parameters()).device

    for prompt in test_prompts:
        formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"
        inputs = tokenizer(formatted, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        responses.append(response.strip())

    return responses


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning demo")
    parser.add_argument(
        "--model",
        type=str,
        default="tinyllama",
        help="Base model to fine-tune (default: tinyllama)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/finetune",
        help="Output directory for adapter weights",
    )
    parser.add_argument(
        "--quantize",
        choices=["int4", "int8"],
        help="Quantization mode (recommended for <8GB VRAM)",
    )
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold]billion-llm Fine-tuning Demo[/bold]\n"
        "LoRA: Train in minutes on consumer hardware",
        border_style="blue",
    ))

    model_name = MODEL_ALIASES.get(args.model.lower(), args.model)

    # Test prompts for before/after comparison
    test_prompts = [
        "What is machine learning?",
        "Write a short poem about coding.",
        "Explain what an API is.",
    ]

    # Load base model
    console.print(f"\n[bold]Loading base model: {model_name}[/bold]")
    model = load_model(model_name, quantize=args.quantize)
    tokenizer = load_tokenizer(model_name)

    # Ensure padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    # Evaluate before fine-tuning
    console.print("\n[bold]Evaluating before fine-tuning...[/bold]")
    before_responses = evaluate_model(model, tokenizer, test_prompts)

    # Configure LoRA
    console.print("\n[bold]Configuring LoRA adapters...[/bold]")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Get PEFT model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Prepare dataset
    console.print("\n[bold]Preparing training data...[/bold]")

    # Expand sample data for more training
    expanded_data = SAMPLE_DATA * 10  # 100 samples
    train_texts = [format_prompt(ex) for ex in expanded_data]
    dataset = Dataset.from_dict({"text": train_texts})

    console.print(f"  Training samples: {len(dataset)}")

    # Training arguments
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        report_to="none",
        dataset_text_field="text",
        max_length=512,
        packing=False,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Train
    console.print("\n[bold]Starting training...[/bold]")

    with track_memory() as mem_stats:
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time

    console.print(f"\n[green]Training completed in {training_time:.1f}s[/green]")
    if "peak_gpu_gb" in mem_stats:
        console.print(f"[dim]Peak GPU memory: {mem_stats['peak_gpu_gb']:.2f} GB[/dim]")

    # Save adapter
    adapter_path = output_dir / "adapter"
    model.save_pretrained(adapter_path)
    console.print(f"\n[dim]Adapter saved to {adapter_path}[/dim]")

    # Evaluate after fine-tuning
    console.print("\n[bold]Evaluating after fine-tuning...[/bold]")

    # Disable gradient checkpointing for inference
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.eval()

    after_responses = evaluate_model(model, tokenizer, test_prompts)

    # Display comparison
    console.print("\n[bold]Before vs After Comparison[/bold]\n")

    for i, prompt in enumerate(test_prompts):
        table = Table(title=f'Prompt: "{prompt}"', show_lines=True)
        table.add_column("Before", style="yellow", width=40)
        table.add_column("After", style="green", width=40)

        before_text = before_responses[i][:200] + "..." if len(before_responses[i]) > 200 else before_responses[i]
        after_text = after_responses[i][:200] + "..." if len(after_responses[i]) > 200 else after_responses[i]

        table.add_row(before_text, after_text)
        console.print(table)
        console.print()

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "epochs": args.epochs,
        "training_time_seconds": training_time,
        "peak_memory_gb": mem_stats.get("peak_gpu_gb", 0),
        "device": get_device(),
        "lora_config": {
            "r": 8,
            "alpha": 16,
            "target_modules": ["q_proj", "v_proj"],
        },
        "comparisons": [
            {
                "prompt": prompt,
                "before": before_responses[i],
                "after": after_responses[i],
            }
            for i, prompt in enumerate(test_prompts)
        ],
    }

    results_path = output_dir / "finetune_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    console.print(f"[dim]Results saved to {results_path}[/dim]")
    console.print("\n[bold green]Fine-tuning demo complete![/bold green]")


if __name__ == "__main__":
    main()
