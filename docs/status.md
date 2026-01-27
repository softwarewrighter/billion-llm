# Project Status: billion-llm

## Current Status: Testing Complete

**Last Updated**: 2026-01-26

## Overview

The billion-llm project compares 1B-class language models, demonstrates fine-tuning workflows, and showcases speculative decoding. This repository accompanies the "Small Models, Big Brains" YouTube Shorts series.

## Completion Status

### Documentation

| Document | Status | Notes |
|----------|--------|-------|
| docs/prd.md | Done | Product requirements defined |
| docs/architecture.md | Done | System architecture documented |
| docs/design.md | Done | Design decisions recorded |
| docs/plan.md | Done | Implementation plan created |
| docs/status.md | Done | This file |
| docs/results.md | Done | Test results documented |
| README.md | Done | Main readme with quick start |
| docs/COMPARISON.md | Pending | Full model comparison |

### Implementation

| Component | Status | Notes |
|-----------|--------|-------|
| requirements.txt | Done | All dependencies listed |
| download_models.py | Done | Downloads all 4 models |
| benchmark.py | Done | Speed, memory benchmarks working |
| demo_chat.py | Done | Single and comparison modes |
| finetune_demo.py | Done | LoRA fine-tuning with before/after |
| speculative_demo.py | Done | Draft model acceleration demo |
| utils/ | Done | Model loader, memory tracking |
| results/ | Done | Output directory (gitignored) |

### Models

| Model | Downloaded | Tested | Benchmarked |
|-------|------------|--------|-------------|
| TinyLlama-1.1B-Chat-v1.0 | Yes | Yes | Yes (42.3 tok/s, 0.50 GB) |
| Llama-3.2-1B-Instruct | No | No | No (requires HF auth) |
| stablelm-2-1_6b-chat | Yes | No | No (config compatibility issue) |
| pythia-1b | Yes | Yes | Yes (46.9 tok/s, 0.69 GB) |

### Test Results Summary

| Test | Status | Key Results |
|------|--------|-------------|
| Chat Demo | Passed | 3.95s response time on MPS |
| Benchmark (TinyLlama) | Passed | 42.4 tok/s, 0.97 GB, 27.0% MMLU |
| Benchmark (Pythia) | Passed | 46.8 tok/s, 0.69 GB, 31.0% MMLU |
| Fine-tuning Demo | Passed | 69s for 3 epochs, loss 1.84→1.60 |
| Speculative Decoding | Passed | Demo works, shows model compatibility importance |

## Quick Start

```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Download models
python download_models.py --model tinyllama
python download_models.py --model pythia

# Run demos
python demo_chat.py --model tinyllama
python benchmark.py --model tinyllama --skip-mmlu
python finetune_demo.py --epochs 3
python speculative_demo.py --draft-model tinyllama
```

## Known Issues

1. **StableLM**: Config compatibility issue with transformers 5.0 (`pad_token_id` missing)
2. **Llama-3.2-1B**: Requires HuggingFace authentication (gated model)
3. **Speculative Decoding**: Shows slowdown with mismatched models (expected, educational)

## Next Steps

1. [ ] Fix StableLM compatibility or document workaround
2. [ ] Test with HF authentication for Llama-3.2-1B
3. [x] Run MMLU benchmarks (TinyLlama: 27.0%, Pythia: 31.0%)
4. [ ] Create docs/COMPARISON.md with full results
5. [ ] Test INT4/INT8 quantization

## Changelog

### 2026-01-26
- Created initial documentation suite
- Implemented all core scripts
- Fixed trl 0.27+ API compatibility in finetune_demo.py
- Fixed chat template in speculative_demo.py
- Added utility modules for model loading and memory tracking
- Downloaded and tested TinyLlama, Pythia models
- Ran all demos successfully
- Documented results in docs/results.md
- Updated .gitignore to exclude generated results

---

## Quick Links

- [PRD](prd.md) - What we're building
- [Architecture](architecture.md) - How it's structured
- [Design](design.md) - Why we made these choices
- [Plan](plan.md) - How we'll build it
- [Results](results.md) - Test results and benchmarks
