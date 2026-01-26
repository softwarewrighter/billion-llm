# Design Document: billion-llm

## Design Philosophy

### Core Principles

1. **Accessibility First**: Every feature must work on consumer hardware (CPU, 4-8GB VRAM)
2. **Educational Value**: Code should teach, not just demonstrate
3. **Reproducibility**: Same inputs produce same outputs across runs
4. **Minimal Dependencies**: Only include what's necessary

## Key Design Decisions

### 1. Model Selection

**Decision**: Include TinyLlama, Llama-3.2-1B, StableLM-1.6B, and Pythia-1B

**Rationale**:
- **TinyLlama**: Represents the "overtraining" paradigm (3T tokens on 1.1B params)
- **Llama-3.2-1B**: Official Meta support ensures long-term ecosystem compatibility
- **StableLM-1.6B**: Multilingual capability differentiates it
- **Pythia-1B**: Research-focused with 154 checkpoints for interpretability

**Alternatives Considered**:
- OpenELM-1.1B: Apple-focused, less universal
- MobileLLM-1B: Phone-optimized, less general-purpose
- SantaCoder: Code-only, too specialized

### 2. Quantization Strategy

**Decision**: Support INT4 and INT8 via bitsandbytes, FP16 as default

**Rationale**:
- INT4 enables running on 4GB VRAM
- INT8 balances quality and memory
- FP16 is the quality baseline
- bitsandbytes is mature and widely supported

**Trade-offs**:
```
┌─────────────────────────────────────────────────────────┐
│                  Quality vs Memory Trade-off            │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Quality                                                 │
│    ▲                                                     │
│    │     ● FP16 (100% quality, 2GB/B)                   │
│    │                                                     │
│    │        ● INT8 (99% quality, 1GB/B)                 │
│    │                                                     │
│    │            ● INT4 (97% quality, 0.5GB/B)           │
│    │                                                     │
│    └──────────────────────────────────────────▶ Memory  │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 3. Fine-tuning Approach

**Decision**: LoRA with r=8, alpha=16, dropout=0.05

**Rationale**:
- r=8 balances parameter count with expressiveness
- alpha=16 (2x rank) follows best practices
- Dropout prevents overfitting on small datasets
- Target modules: q_proj, v_proj (standard for LLMs)

**Configuration**:
```python
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

### 4. Benchmark Selection

**Decision**: MMLU subset (10 categories) + HumanEval subset + custom speed tests

**Rationale**:
- MMLU tests general knowledge and reasoning
- HumanEval tests practical code generation
- Subset approach keeps benchmarks fast (<30 min total)
- Custom speed tests measure real-world performance

**MMLU Categories Selected**:
1. Abstract Algebra
2. Computer Science
3. Elementary Mathematics
4. High School Physics
5. High School Chemistry
6. Logical Fallacies
7. Machine Learning
8. Professional Law
9. World Religions
10. US History

### 5. Speculative Decoding Implementation

**Decision**: Implement basic speculative decoding with configurable K (draft length)

**Rationale**:
- K=4 is optimal for most 1B→7B combinations
- Simple implementation prioritizes understanding over optimization
- Acceptance rate visualization helps users understand the mechanism

**Algorithm**:
```
1. Draft model generates K candidate tokens
2. Target model evaluates all K+1 positions in one forward pass
3. Compare draft probabilities with target probabilities
4. Accept tokens until first rejection
5. If rejected, sample from adjusted distribution
6. Repeat from accepted position
```

### 6. CLI Design

**Decision**: Argparse-based CLI with sensible defaults

**Rationale**:
- Works in any terminal
- No external dependencies for UI
- Easy to script and automate
- Accessible to beginners

**Interface Pattern**:
```bash
# Benchmark all models
python benchmark.py

# Benchmark specific model with quantization
python benchmark.py --model tinyllama --quantize int4

# Interactive chat
python demo_chat.py --model llama3.2-1b

# Side-by-side comparison
python demo_chat.py --compare tinyllama llama3.2-1b
```

## Error Handling Strategy

### Graceful Degradation

```python
def load_model(name, quantize=None):
    try:
        if quantize == "int4":
            return load_quantized_4bit(name)
    except ImportError:
        print("bitsandbytes not available, falling back to FP16")
        quantize = None

    try:
        return load_fp16(name)
    except torch.cuda.OutOfMemoryError:
        print("GPU OOM, falling back to CPU")
        return load_cpu(name)
```

### User-Friendly Errors

| Error Type | Message | Action |
|------------|---------|--------|
| Model not found | "Model 'X' not downloaded. Run download_models.py first." | Point to download script |
| OOM | "Out of memory. Try --quantize int4 or --device cpu" | Suggest alternatives |
| Missing dependency | "bitsandbytes required for INT4. Install: uv pip install bitsandbytes" | Provide install command |

## Output Formats

### Benchmark Results (JSON)

```json
{
  "timestamp": "2024-01-26T12:00:00Z",
  "models": {
    "tinyllama-1.1b": {
      "mmlu_accuracy": 0.42,
      "humaneval_pass_rate": 0.18,
      "tokens_per_second": 45.2,
      "peak_memory_gb": 2.4
    }
  },
  "hardware": {
    "device": "cuda",
    "gpu_name": "RTX 3080",
    "vram_gb": 10
  }
}
```

### Console Output (Tables)

```
╭──────────────────────────────────────────────────────╮
│              Benchmark Results Summary                │
├─────────────┬──────────┬───────────┬────────┬────────┤
│ Model       │ MMLU (%) │ HumanEval │ tok/s  │ Memory │
├─────────────┼──────────┼───────────┼────────┼────────┤
│ TinyLlama   │   42.1   │   18.2%   │  45.2  │ 2.4 GB │
│ Llama-3.2   │   45.3   │   22.1%   │  52.1  │ 2.1 GB │
│ StableLM    │   44.8   │   19.5%   │  38.7  │ 3.2 GB │
│ Pythia      │   39.2   │   15.3%   │  48.9  │ 2.3 GB │
╰─────────────┴──────────┴───────────┴────────┴────────╯
```

## Testing Strategy

### Unit Tests

- Model loading with various configurations
- Quantization correctness
- Benchmark metric calculations

### Integration Tests

- End-to-end benchmark run (small subset)
- Fine-tuning cycle (1 epoch, 10 samples)
- Speculative decoding verification

### Manual Tests

- Visual inspection of chat outputs
- Memory monitoring during long sessions
- Cross-platform verification (Linux, macOS, Windows)

## Future Considerations

### Potential Extensions

1. **Web UI**: Gradio interface for non-CLI users
2. **Model merging**: Combine fine-tuned LoRAs
3. **Quantization comparison**: GPTQ, AWQ, GGUF formats
4. **Distributed inference**: Multi-GPU support
5. **Continuous benchmarking**: Track model updates

### Intentionally Out of Scope

- Training from scratch (use dedicated repos)
- Models >3B parameters (different hardware requirements)
- Production serving (use vLLM, TGI, etc.)
- Cloud deployment (focus on local)
