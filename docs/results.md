# Test Results: billion-llm

## Test Environment

| Property | Value |
|----------|-------|
| Date | 2026-01-26 |
| Device | Apple Silicon (MPS) |
| Python | 3.10.18 |
| PyTorch | 2.10.0 |
| Transformers | 5.0.0 |
| TRL | 0.27.1 |

---

## 1. Chat Demo Test

**Model**: TinyLlama-1.1B-Chat-v1.0

| Test | Result |
|------|--------|
| Model Loading | Success (MPS device) |
| Prompt | "What is the capital of France? Answer briefly." |
| Response | "The capital of France is Paris." |
| Generation Time | 3.95 seconds |

---

## 2. Benchmark Results

**Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0
**Quantization**: None (FP16)
**Device**: MPS (Apple Silicon)

### Speed Benchmark

| Metric | Value |
|--------|-------|
| Tokens per second | **42.3 tok/s** |
| Avg generation time | 1.63 seconds |
| Avg tokens generated | 69 tokens |
| Number of runs | 5 |

### Memory Benchmark

| Metric | Value |
|--------|-------|
| Model memory | 0.50 GB |
| Peak memory | 0.50 GB |

### Summary Table

| Model | Params | Speed (tok/s) | Memory (GB) | MMLU (%) |
|-------|--------|---------------|-------------|----------|
| TinyLlama-1.1B-Chat-v1.0 | 1.1B | 42.4 | 0.97 | 27.0 |
| Pythia-1b | 1.08B | 46.8 | 0.69 | 31.0 |

**Observations:**
- Pythia scores higher on MMLU (31.0% vs 27.0%) despite being a base model
- Pythia is slightly faster (46.8 vs 42.4 tok/s)
- TinyLlama uses more memory during MMLU evaluation (0.97 GB peak)
- Both run comfortably on consumer hardware
- Note: Random baseline for 4-choice MMLU is 25%

---

## 3. Fine-tuning Demo Results

**Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0
**Device**: MPS (Apple Silicon)

### Configuration

| Parameter | Value |
|-----------|-------|
| LoRA rank (r) | 8 |
| LoRA alpha | 16 |
| Target modules | q_proj, v_proj |
| Trainable params | 1,126,400 (0.1% of total) |
| Training samples | 100 |
| Epochs | 3 |
| Batch size | 4 |
| Gradient accumulation | 4 |
| Learning rate | 2e-4 |

### Training Metrics

| Metric | Value |
|--------|-------|
| Training time | **69.0 seconds** |
| Initial loss | 1.844 |
| Final loss | 1.602 |
| Initial token accuracy | 59.5% |
| Final token accuracy | 63.7% |

### Before/After Comparison

#### Prompt 1: "What is machine learning?"

**Before:**
> Machine learning is a field of computer science that involves the use of algorithms and statistical models to learn from data. It is a powerful tool for analyzing and predicting complex patterns in data, and has many applications in fields such as healthcare, finance, and marketing. In this project, we will use machine learning to predict the likelihood of a customer churning from a subscription-based service.

**After:**
> Machine learning is a field of computer science that involves the use of algorithms and statistical models to learn from data. It is used in a variety of applications, including image recognition, natural language processing, and fraud detection.

**Observation**: More concise, focused response without hallucinating project context.

---

#### Prompt 2: "Write a short poem about coding."

**Before:**
> Coding is a language that has the power to transform the world. It is a language that has the power to create, to innovate, and to inspire. It is a language that has the power to connect people from all over the world, to share ideas, and to build bridges...

**After:**
> Coding is a language that tells the computer what to do.
> It's a way to express ideas and thoughts.
> It's a way to create, to build, to make.
> Coding is a way to express oneself.

**Observation**: Learned poem structure from training data (line breaks, shorter lines).

---

#### Prompt 3: "Explain what an API is."

**Before:**
> An API is a set of rules and guidelines that define how data is exchanged between applications. It is a set of protocols, standards, and conventions that allow developers to communicate with each other using a common language...

**After:**
> I'm not capable of creating an API. However, I can provide you with a brief explanation of what an API is. An API (Application Programming Interface) is a set of rules that allow two or more programs to communicate with each other...

**Observation**: Learned humility pattern from training data, but still provides accurate information.

---

---

## 4. Speculative Decoding Demo Results

**Draft Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0 (instruction-tuned)
**Target Model**: EleutherAI/pythia-1b (base model)
**Device**: MPS (Apple Silicon)
**Draft tokens (k)**: 4

### Configuration

| Parameter | Value |
|-----------|-------|
| Prompt | "Explain the theory of relativity in simple terms." |
| Max tokens | 100 |
| Draft tokens per iteration (k) | 4 |

### Performance Results

| Method | Speed (tok/s) | Time (s) | Tokens | Notes |
|--------|---------------|----------|--------|-------|
| Draft (TinyLlama) | **32.4** | 3.09 | 100 | Instruction-tuned, coherent output |
| Target (Pythia-1b) | **25.2** | 3.97 | 100 | Base model, incoherent with chat format |
| Speculative | **7.4** | 13.91 | 103 | 48.5% acceptance rate |

### Speculative Decoding Stats

| Metric | Value |
|--------|-------|
| Draft iterations | 53 |
| Tokens accepted | 50 |
| Tokens rejected | 53 |
| Acceptance rate | **48.5%** |
| Speedup vs target | **0.29x** (slowdown) |

### Analysis

The speculative decoding demo shows a **slowdown** rather than speedup. This is expected and educational:

**Why no speedup?**
1. **Similar model sizes**: Both models are ~1B parameters, so draft isn't significantly faster
2. **Distribution mismatch**: TinyLlama (instruction-tuned) vs Pythia (base model) have very different output distributions
3. **Low acceptance rate**: 48.5% acceptance means frequent rejections and resampling overhead

**When speculative decoding helps:**
- Draft model is **significantly faster** (e.g., 1B draft for 7B+ target)
- Models from **same family** (e.g., TinyLlama draft for Llama-2-7B target)
- **Similar training** (both instruction-tuned or both base)

### Sample Outputs

**Draft Model (TinyLlama):**
> The theory of relativity is a scientific theory that explains how objects move and interact with each other in a way that is consistent with the principles of physics. It is based on the principle of relativity, which states that the laws of physics are the same for all observers, regardless of their location or velocity.

**Target Model (Pythia-1b):**
> (Incoherent output - base model doesn't understand chat format)

**Conclusion**: This demo successfully demonstrates the speculative decoding mechanism, but highlights that model compatibility is crucial for achieving speedup.

---

## Key Findings

1. **Speed**: TinyLlama achieves 32-42 tokens/second on Apple Silicon MPS
2. **Memory**: Only 0.5 GB required for inference (FP16)
3. **Fine-tuning**: LoRA enables training in ~1 minute with minimal memory overhead
4. **Quality**: Fine-tuning with 100 samples shows measurable improvement in response style
5. **Speculative Decoding**: Requires compatible model pairs (same family, similar training) for speedup

## Next Steps

- [ ] Benchmark remaining models (Llama-3.2-1B, StableLM-1.6B)
- [ ] Run MMLU evaluation
- [ ] Test with compatible model pair for speculative decoding speedup
- [ ] Compare INT4/INT8 quantization performance
