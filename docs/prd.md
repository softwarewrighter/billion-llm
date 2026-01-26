# Product Requirements Document: billion-llm

## Overview

**Project Name**: billion-llm
**Purpose**: Compare 1B-class language models, demonstrate fine-tuning, and showcase speculative decoding
**Target Audience**: Developers, researchers, and AI enthusiasts with consumer-grade hardware

## Problem Statement

One billion parameters represents a critical inflection point in language model capability. Below this threshold, models struggle with complex reasoning. Above it, hardware requirements increase significantly. Developers need:

1. A clear comparison of available 1B-class models
2. Practical demonstrations of fine-tuning workflows
3. Examples of using small models for speculative decoding

## Goals

1. **Compare**: Benchmark TinyLlama, Llama-3.2-1B, StableLM-1.6B, and Pythia-1B on standard tasks
2. **Demonstrate**: Show LoRA fine-tuning on a small dataset with measurable before/after quality
3. **Showcase**: Implement speculative decoding using 1B models as draft generators
4. **Educate**: Explain why 1B is the "sweet spot" for capability vs. efficiency

## Non-Goals

- Training models from scratch
- Supporting models larger than 3B parameters
- Production deployment infrastructure
- GPU cluster optimization

## Models

| Model | Parameters | Source | Key Strength |
|-------|------------|--------|--------------|
| TinyLlama-1.1B-Chat-v1.0 | 1.1B | Community | 3T tokens overtraining |
| Llama-3.2-1B-Instruct | 1B | Meta | Official ecosystem support |
| stablelm-2-1_6b-chat | 1.6B | Stability AI | Multilingual, 2T tokens |
| pythia-1b | 1.08B | EleutherAI | 154 checkpoints for research |

## Features

### Core Features

1. **Model Download Script** (`download_models.py`)
   - Download all 4 models with progress display
   - Verify checksums
   - Support partial downloads/resume

2. **Benchmark Suite** (`benchmark.py`)
   - MMLU subset (10 categories)
   - HumanEval subset (code generation)
   - Inference speed (tokens/sec)
   - Memory usage (peak RAM/VRAM)

3. **Chat Demo** (`demo_chat.py`)
   - Interactive CLI chat with model selection
   - Side-by-side comparison mode
   - Timing and token count display

4. **Fine-tuning Demo** (`finetune_demo.py`)
   - LoRA fine-tuning on 100-example dataset
   - Training time and memory monitoring
   - Before/after quality comparison

5. **Speculative Decoding Demo** (`speculative_demo.py`)
   - Use 1B model as draft for 7B generation
   - Measure speedup vs direct generation
   - Visualize draft acceptance rate

### Output

- `results/` directory with JSON benchmark outputs
- `docs/COMPARISON.md` with model comparison summary
- Terminal visualizations for demos

## Technical Requirements

### Hardware Support

- **Minimum**: CPU-only mode (slower but functional)
- **Recommended**: 4-8GB VRAM GPU
- **Memory**: 8GB RAM minimum, 16GB recommended

### Software Dependencies

- Python 3.10+
- transformers
- peft (for LoRA)
- trl (for training)
- bitsandbytes (for quantization)
- torch

### Package Management

Use `uv` for all package management:
```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

## Success Criteria

1. All 4 models run on 8GB VRAM without OOM
2. Fine-tuning completes in <10 minutes on consumer GPU
3. Speculative decoding shows measurable speedup
4. Clear documentation explaining when to use each model

## Timeline

| Phase | Description | Deliverables |
|-------|-------------|--------------|
| 1 | Setup & Downloads | requirements.txt, download script |
| 2 | Benchmarking | benchmark.py, results/ |
| 3 | Demos | chat, fine-tune, speculative demos |
| 4 | Documentation | COMPARISON.md, README |

## References

- [TinyLlama Paper](https://arxiv.org/abs/2401.02385)
- [Llama 3.2 Announcement](https://ai.meta.com/llama/)
- [Pythia Paper](https://arxiv.org/abs/2304.01373)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
