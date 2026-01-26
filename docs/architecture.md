# Architecture: billion-llm

## System Overview

```
billion-llm/
├── download_models.py      # Model acquisition
├── benchmark.py            # Performance measurement
├── demo_chat.py            # Interactive inference
├── finetune_demo.py        # LoRA training demo
├── speculative_demo.py     # Draft model acceleration
├── requirements.txt        # Dependencies
├── results/                # Benchmark outputs
│   ├── benchmark_results.json
│   ├── finetune_results.json
│   └── speculative_results.json
└── docs/
    ├── architecture.md     # This file
    ├── prd.md              # Requirements
    ├── design.md           # Design decisions
    ├── plan.md             # Implementation plan
    ├── status.md           # Current status
    └── COMPARISON.md       # Model comparison
```

## Component Architecture

### 1. Model Management Layer

```
┌─────────────────────────────────────────────────────────┐
│                   download_models.py                     │
├─────────────────────────────────────────────────────────┤
│  - HuggingFace Hub integration                          │
│  - Model verification (checksums)                        │
│  - Progress tracking                                     │
│  - Disk space validation                                 │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    ~/.cache/huggingface                  │
│  ├── TinyLlama/TinyLlama-1.1B-Chat-v1.0                 │
│  ├── meta-llama/Llama-3.2-1B-Instruct                   │
│  ├── stabilityai/stablelm-2-1_6b-chat                   │
│  └── EleutherAI/pythia-1b                               │
└─────────────────────────────────────────────────────────┘
```

### 2. Inference Layer

```
┌─────────────────────────────────────────────────────────┐
│                     Model Loader                         │
├─────────────────────────────────────────────────────────┤
│  load_model(name, quantize=False, device="auto")        │
│  - Automatic device selection (CUDA/MPS/CPU)            │
│  - Optional INT4/INT8 quantization                      │
│  - Memory-efficient loading                             │
└─────────────────────────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │  GPU     │    │  Apple   │    │   CPU    │
    │  CUDA    │    │  MPS     │    │  Fallback│
    └──────────┘    └──────────┘    └──────────┘
```

### 3. Benchmark Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    benchmark.py                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   MMLU      │  │  HumanEval  │  │   Speed     │     │
│  │  Evaluator  │  │  Evaluator  │  │  Profiler   │     │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │
│         │                │                │             │
│         └────────────────┼────────────────┘             │
│                          ▼                              │
│              ┌─────────────────────┐                    │
│              │   Results Aggregator │                    │
│              │   - JSON output      │                    │
│              │   - Markdown tables  │                    │
│              └─────────────────────┘                    │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 4. Fine-tuning Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   finetune_demo.py                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────────┐    ┌──────────────────┐          │
│  │   Base Model     │    │   LoRA Config    │          │
│  │   (frozen)       │    │   r=8, alpha=16  │          │
│  └────────┬─────────┘    └────────┬─────────┘          │
│           │                       │                     │
│           └───────────┬───────────┘                     │
│                       ▼                                 │
│           ┌──────────────────────┐                      │
│           │   PEFT Model         │                      │
│           │   (trainable LoRA)   │                      │
│           └──────────┬───────────┘                      │
│                      │                                  │
│                      ▼                                  │
│           ┌──────────────────────┐                      │
│           │   SFTTrainer (trl)   │                      │
│           │   - 100 examples     │                      │
│           │   - Memory tracking  │                      │
│           └──────────┬───────────┘                      │
│                      │                                  │
│                      ▼                                  │
│           ┌──────────────────────┐                      │
│           │   Merged Model       │                      │
│           │   (for comparison)   │                      │
│           └──────────────────────┘                      │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 5. Speculative Decoding Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 speculative_demo.py                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────────┐    ┌──────────────────┐          │
│  │   Draft Model    │    │   Target Model   │          │
│  │   (1B - fast)    │    │   (7B - slow)    │          │
│  └────────┬─────────┘    └────────┬─────────┘          │
│           │                       │                     │
│           ▼                       │                     │
│  ┌──────────────────┐             │                     │
│  │ Generate K tokens│             │                     │
│  │ (draft candidates)│            │                     │
│  └────────┬─────────┘             │                     │
│           │                       │                     │
│           └───────────┬───────────┘                     │
│                       ▼                                 │
│           ┌──────────────────────┐                      │
│           │   Verification       │                      │
│           │   (parallel eval)    │                      │
│           └──────────┬───────────┘                      │
│                      │                                  │
│           ┌──────────┴──────────┐                       │
│           ▼                     ▼                       │
│    ┌────────────┐        ┌────────────┐                │
│    │  Accept    │        │  Reject    │                │
│    │  tokens    │        │  resample  │                │
│    └────────────┘        └────────────┘                │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Data Flow

### Benchmark Flow

```
User runs benchmark.py
         │
         ▼
┌─────────────────┐
│ Load all models │
│ sequentially    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ For each model: │
│ - Run MMLU      │
│ - Run HumanEval │
│ - Measure speed │
│ - Track memory  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Aggregate       │
│ results         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Output:         │
│ - JSON file     │
│ - Console table │
└─────────────────┘
```

### Fine-tuning Flow

```
User runs finetune_demo.py
         │
         ▼
┌─────────────────┐
│ Load base model │
│ (TinyLlama)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Evaluate before │
│ fine-tuning     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Apply LoRA      │
│ adapters        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Train on        │
│ 100 examples    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Evaluate after  │
│ fine-tuning     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Display         │
│ comparison      │
└─────────────────┘
```

## Memory Management

### Quantization Options

| Mode | Memory per 1B params | Speed | Quality |
|------|---------------------|-------|---------|
| FP32 | ~4GB | Baseline | 100% |
| FP16 | ~2GB | ~1.5x | ~99.9% |
| INT8 | ~1GB | ~1.2x | ~99% |
| INT4 | ~0.5GB | ~1.1x | ~97% |

### Device Selection Logic

```python
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
```

## External Dependencies

### Required Libraries

| Library | Purpose | Version |
|---------|---------|---------|
| transformers | Model loading, inference | >=4.36.0 |
| torch | Tensor operations | >=2.0.0 |
| peft | LoRA implementation | >=0.7.0 |
| trl | Training utilities | >=0.7.0 |
| bitsandbytes | Quantization | >=0.41.0 |
| datasets | Data loading | >=2.14.0 |
| accelerate | Multi-device support | >=0.24.0 |

### Optional Libraries

| Library | Purpose |
|---------|---------|
| flash-attn | Faster attention (GPU only) |
| triton | Kernel optimization |
