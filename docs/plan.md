# Implementation Plan: billion-llm

## Phase Overview

| Phase | Focus | Deliverables | Dependencies |
|-------|-------|--------------|--------------|
| 1 | Project Setup | requirements.txt, download script | None |
| 2 | Core Infrastructure | Model loader, utilities | Phase 1 |
| 3 | Benchmarking | benchmark.py, results format | Phase 2 |
| 4 | Demos | chat, finetune, speculative | Phase 2 |
| 5 | Documentation | README, COMPARISON.md | Phase 3-4 |

## Phase 1: Project Setup

### 1.1 Create requirements.txt

```
torch>=2.0.0
transformers>=4.36.0
peft>=0.7.0
trl>=0.7.0
datasets>=2.14.0
accelerate>=0.24.0
bitsandbytes>=0.41.0
sentencepiece
protobuf
```

### 1.2 Create download_models.py

**Functionality**:
- Download all 4 models from HuggingFace Hub
- Show progress bars
- Verify disk space before download
- Handle authentication for gated models (Llama)

**Models to download**:
```python
MODELS = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "meta-llama/Llama-3.2-1B-Instruct",
    "stabilityai/stablelm-2-1_6b-chat",
    "EleutherAI/pythia-1b",
]
```

### 1.3 Setup Results Directory

```
results/
├── .gitkeep
└── README.md  # Explains output format
```

## Phase 2: Core Infrastructure

### 2.1 Create utils/model_loader.py

```python
# Key functions:
def load_model(name: str, quantize: str = None, device: str = "auto"):
    """Load model with automatic device selection and optional quantization."""

def get_device() -> str:
    """Return best available device (cuda > mps > cpu)."""

def get_tokenizer(name: str):
    """Load tokenizer with proper padding configuration."""
```

### 2.2 Create utils/memory.py

```python
# Key functions:
def get_memory_usage() -> dict:
    """Return current memory usage for CPU and GPU."""

def memory_tracker(func):
    """Decorator to track peak memory during function execution."""
```

### 2.3 Create Model Name Mapping

```python
MODEL_ALIASES = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "llama3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "stablelm": "stabilityai/stablelm-2-1_6b-chat",
    "pythia-1b": "EleutherAI/pythia-1b",
}
```

## Phase 3: Benchmarking

### 3.1 Create benchmark.py

**CLI Interface**:
```bash
python benchmark.py [--model MODEL] [--quantize {int4,int8}] [--device {cuda,cpu}]
```

**Benchmark Components**:

1. **MMLU Subset** (10 categories, 5-shot)
   - Load from datasets hub
   - Format as multiple choice
   - Extract answer and compare

2. **HumanEval Subset** (20 problems)
   - Load problems
   - Generate completions
   - Execute and verify

3. **Speed Test**
   - Fixed prompt, measure tokens/second
   - Warm-up run, then 5 timed runs
   - Report mean and std

4. **Memory Test**
   - Track peak memory during inference
   - Report for GPU and CPU

### 3.2 Output Format

```python
results = {
    "timestamp": datetime.now().isoformat(),
    "hardware": {
        "device": device,
        "gpu_name": torch.cuda.get_device_name() if cuda else None,
    },
    "models": {
        model_name: {
            "mmlu_accuracy": float,
            "humaneval_pass_rate": float,
            "tokens_per_second": float,
            "peak_memory_gb": float,
        }
    }
}
```

## Phase 4: Demos

### 4.1 Create demo_chat.py

**Features**:
- Single model chat mode
- Side-by-side comparison mode
- Token count and timing display
- System prompt configuration

**CLI Interface**:
```bash
# Single model
python demo_chat.py --model tinyllama

# Comparison
python demo_chat.py --compare tinyllama llama3.2-1b
```

### 4.2 Create finetune_demo.py

**Features**:
- Load base model
- Apply LoRA configuration
- Train on small dataset (alpaca-style, 100 samples)
- Before/after comparison
- Save adapter weights

**Training Configuration**:
```python
training_args = TrainingArguments(
    output_dir="./results/finetune",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
)
```

### 4.3 Create speculative_demo.py

**Features**:
- Load draft model (1B) and target model (7B)
- Implement speculative decoding loop
- Measure speedup vs baseline
- Show acceptance rate statistics

**Configuration**:
```python
DRAFT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TARGET_MODEL = "meta-llama/Llama-2-7b-chat-hf"  # Or similar
K = 4  # Number of draft tokens
```

## Phase 5: Documentation

### 5.1 Create README.md

**Sections**:
1. Overview (why 1B is special)
2. Quick Start
3. Model Comparison Table
4. Usage Examples
5. Hardware Requirements
6. Contributing

### 5.2 Create docs/COMPARISON.md

**Content**:
- Detailed benchmark results
- When to use each model
- Strengths and weaknesses
- Fine-tuning recommendations

## Task Breakdown

### Week 1

| Day | Tasks |
|-----|-------|
| 1 | Phase 1 complete: requirements, download script |
| 2 | Phase 2 complete: utils, model loader |
| 3 | Phase 3.1: MMLU benchmark |
| 4 | Phase 3.2-3.3: HumanEval, speed tests |
| 5 | Phase 4.1: demo_chat.py |
| 6 | Phase 4.2: finetune_demo.py |
| 7 | Phase 4.3: speculative_demo.py |

### Week 2

| Day | Tasks |
|-----|-------|
| 1 | Run full benchmarks, collect results |
| 2 | Phase 5: Documentation |
| 3 | Testing and bug fixes |
| 4 | Polish and release |

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Llama model gated | Provide instructions for HF token setup |
| OOM on small GPUs | Default to quantization, test on 4GB cards |
| Benchmark variance | Multiple runs, report std dev |
| API changes | Pin dependency versions |

## Definition of Done

- [ ] All 4 models download successfully
- [ ] Benchmarks complete without errors
- [ ] Chat demo works on CPU
- [ ] Fine-tuning completes in <10 min
- [ ] Speculative decoding shows speedup
- [ ] README has quick start that works
- [ ] Results directory populated
