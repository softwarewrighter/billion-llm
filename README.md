# billion-llm

**The 1B Sweet Spot**: Compare, fine-tune, and accelerate billion-parameter language models.

Part of the ["Small Models, Big Brains"](https://github.com/softwarewrighter/shorts) YouTube Shorts series.

## Why One Billion?

One billion parameters is the magic number for language models:

- **Below 1B**: Models struggle with complex reasoning
- **Above 1B**: Hardware requirements increase significantly
- **At 1B**: Maximum capability per watt, fine-tunes in minutes, runs anywhere

## Models Compared

| Model | Params | Key Strength | Source |
|-------|--------|--------------|--------|
| [TinyLlama](https://github.com/jzhang38/TinyLlama) | 1.1B | 3T tokens overtraining | Community |
| [Llama-3.2-1B](https://ai.meta.com/llama/) | 1B | Official Meta ecosystem | Meta |
| [StableLM-1.6B](https://huggingface.co/stabilityai/stablelm-2-1_6b-chat) | 1.6B | Multilingual, 2T tokens | Stability AI |
| [Pythia-1B](https://github.com/EleutherAI/pythia) | 1.08B | 154 checkpoints for research | EleutherAI |

## Features

### Benchmarking
- MMLU subset (10 categories)
- HumanEval code generation
- Inference speed (tokens/sec)
- Memory usage profiling

### Fine-tuning Demo
- LoRA adapters (train in minutes)
- Before/after quality comparison
- Memory-efficient training

### Speculative Decoding
- Use 1B as draft model for 7B generation
- Measure speedup vs direct inference
- Visualize acceptance rates

## Quick Start

```bash
# Clone the repo
git clone https://github.com/softwarewrighter/billion-llm
cd billion-llm

# Install dependencies (using uv)
uv pip install -r requirements.txt

# Download models
python download_models.py

# Run benchmarks
python benchmark.py

# Interactive chat
python demo_chat.py --model tinyllama

# Compare models side-by-side
python demo_chat.py --compare tinyllama llama3.2-1b
```

## Hardware Requirements

| Setup | What You Can Run |
|-------|------------------|
| CPU only | All models (slower, quantized) |
| 4GB VRAM | All models (INT4 quantized) |
| 8GB VRAM | All models (FP16) |
| Apple Silicon | All models (MPS acceleration) |

## Project Structure

```
billion-llm/
├── README.md
├── requirements.txt
├── download_models.py      # Download all models
├── benchmark.py            # Run benchmarks
├── demo_chat.py            # Interactive chat
├── finetune_demo.py        # LoRA fine-tuning
├── speculative_demo.py     # Draft model acceleration
├── results/                # Benchmark outputs
└── docs/
    ├── prd.md              # Product requirements
    ├── architecture.md     # System design
    ├── design.md           # Design decisions
    ├── plan.md             # Implementation plan
    ├── status.md           # Current status
    └── COMPARISON.md       # Model comparison
```

## Documentation

- [Product Requirements](docs/prd.md) - What we're building
- [Architecture](docs/architecture.md) - How it's structured
- [Design](docs/design.md) - Why we made these choices
- [Plan](docs/plan.md) - Implementation roadmap
- [Status](docs/status.md) - Current progress

## Key Concepts Demonstrated

### Overtraining
TinyLlama trains on 3T tokens with only 1.1B parameters - 100x more than the Chinchilla-optimal ratio. More data, same model size.

### LoRA Fine-tuning
Low-Rank Adaptation adds trainable parameters without modifying the base model. Fine-tune in minutes on a laptop.

### Speculative Decoding
Use a fast 1B model to draft tokens, verify with a slower 7B model. Get 2-3x speedup on autoregressive generation.

## References

- [TinyLlama Paper](https://arxiv.org/abs/2401.02385)
- [Llama 3.2 Announcement](https://ai.meta.com/llama/)
- [Pythia Paper](https://arxiv.org/abs/2304.01373)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Speculative Decoding Paper](https://arxiv.org/abs/2211.17192)

## License

MIT

## Contributing

Issues and PRs welcome. See [docs/plan.md](docs/plan.md) for implementation roadmap.
