# Model Comparison: billion-llm

A comprehensive comparison of 1B-class language models tested on Apple Silicon (MPS).

## Quick Summary

| Model | Params | Speed | Memory | MMLU | Best For |
|-------|--------|-------|--------|------|----------|
| **TinyLlama-1.1B** | 1.1B | 42.4 tok/s | 0.97 GB | 27.0% | Chat, instruction-following |
| **Pythia-1b** | 1.08B | 46.8 tok/s | 0.69 GB | 31.0% | Research, interpretability |

## Detailed Comparison

### TinyLlama-1.1B-Chat-v1.0

| Metric | Value |
|--------|-------|
| **Parameters** | 1.1 billion |
| **Architecture** | Llama 2 |
| **Training Tokens** | 3 trillion (overtraining) |
| **Context Length** | 2,048 tokens |
| **Source** | Community (jzhang38) |

**Benchmark Results:**
- Speed: 42.4 tokens/second
- Memory: 0.97 GB peak (during MMLU)
- MMLU: 27.0% accuracy

**Strengths:**
- Instruction-tuned (follows prompts well)
- Chat template support
- Large community and ecosystem
- Proven overtraining methodology

**Weaknesses:**
- Lower MMLU score than Pythia
- Older Llama 2 architecture
- Higher memory usage during complex tasks

**Best Use Cases:**
- Chatbots and assistants
- Instruction-following tasks
- Fine-tuning base for custom applications
- LoRA experimentation

---

### Pythia-1b

| Metric | Value |
|--------|-------|
| **Parameters** | 1.08 billion |
| **Architecture** | GPT-NeoX |
| **Training Tokens** | 300 billion |
| **Context Length** | 2,048 tokens |
| **Source** | EleutherAI |

**Benchmark Results:**
- Speed: 46.8 tokens/second
- Memory: 0.69 GB peak
- MMLU: 31.0% accuracy

**Strengths:**
- Higher MMLU accuracy
- Faster inference
- Lower memory footprint
- 154 training checkpoints available
- Excellent for interpretability research

**Weaknesses:**
- Base model (not instruction-tuned)
- No chat template (produces incoherent output with chat prompts)
- Requires careful prompt engineering

**Best Use Cases:**
- Research and interpretability
- Studying training dynamics
- Text completion tasks
- Base for instruction fine-tuning

---

## Head-to-Head Comparison

### Speed Comparison

```
Pythia-1b     ████████████████████████████████████████████████ 46.8 tok/s
TinyLlama     ███████████████████████████████████████████ 42.4 tok/s
              0        10        20        30        40        50
```

**Winner: Pythia-1b** (+10% faster)

### Memory Efficiency

```
Pythia-1b     ███████████████████████████████████ 0.69 GB
TinyLlama     █████████████████████████████████████████████████ 0.97 GB
              0.0      0.2      0.4      0.6      0.8      1.0
```

**Winner: Pythia-1b** (29% less memory)

### MMLU Accuracy

```
Pythia-1b     ███████████████████████████████ 31.0%
TinyLlama     ███████████████████████████ 27.0%
Random        █████████████████████████ 25.0%
              0%       10%       20%       30%       40%
```

**Winner: Pythia-1b** (+4 percentage points)

### Chat Quality

| Prompt | TinyLlama | Pythia |
|--------|-----------|--------|
| "What is the capital of France?" | Clear, correct answer | Incoherent (base model) |
| "Write a poem about coding" | Structured poem | Unstructured text |
| "Explain machine learning" | Concise explanation | Rambling continuation |

**Winner: TinyLlama** (instruction-tuned)

---

## When to Use Each Model

### Choose TinyLlama When:
- ✅ Building a chatbot or assistant
- ✅ Need instruction-following capability
- ✅ Want to fine-tune for specific tasks
- ✅ Require chat template support
- ✅ Community support is important

### Choose Pythia When:
- ✅ Conducting research on model behavior
- ✅ Studying training dynamics (154 checkpoints)
- ✅ Need maximum inference speed
- ✅ Memory is constrained
- ✅ Building your own instruction-tuned model
- ✅ Interpretability is important

---

## Fine-tuning Comparison

Both models support LoRA fine-tuning. Results from our demo:

| Metric | TinyLlama |
|--------|-----------|
| Training Time (3 epochs) | 69 seconds |
| Trainable Params | 1.1M (0.1%) |
| Initial Loss | 1.844 |
| Final Loss | 1.602 |
| Token Accuracy | 59.5% → 63.7% |

TinyLlama is recommended for fine-tuning due to:
- Pre-existing instruction tuning
- Better base performance on chat tasks
- Larger training token count (3T vs 300B)

---

## Speculative Decoding Compatibility

| Draft Model | Target Model | Compatible? | Notes |
|-------------|--------------|-------------|-------|
| TinyLlama | Pythia | ❌ No | Different architectures, low acceptance rate |
| TinyLlama | Llama-2-7B | ✅ Yes | Same architecture family |
| Pythia-70M | Pythia-1b | ✅ Yes | Same architecture family |

**Key Insight:** Speculative decoding requires models from the same family for optimal speedup.

---

## Hardware Requirements

| Model | Minimum | Recommended |
|-------|---------|-------------|
| TinyLlama-1.1B | 2 GB RAM | 4 GB RAM / GPU |
| Pythia-1b | 2 GB RAM | 4 GB RAM / GPU |

Both models run on:
- ✅ CPU (slower)
- ✅ Apple Silicon (MPS)
- ✅ NVIDIA GPU (CUDA)

**Quantization Notes:**
| Platform | Recommendation |
|----------|----------------|
| NVIDIA CUDA | INT8/INT4 recommended (bitsandbytes optimized) |
| Apple MPS | FP16 recommended (quantization 12x slower) |
| CPU | Consider GGUF format (llama.cpp) |

---

## Models Not Yet Tested

| Model | Status | Notes |
|-------|--------|-------|
| Llama-3.2-1B-Instruct | Requires HF auth | Gated model |
| StableLM-1.6B | Config issue | `pad_token_id` compatibility |

---

## Recommendations

### For Beginners
**Start with TinyLlama** - It's instruction-tuned, has great community support, and works out of the box for chat applications.

### For Researchers
**Use Pythia** - The 154 training checkpoints and interpretability focus make it ideal for studying how models learn.

### For Production
**Consider both** - Use TinyLlama for user-facing features, Pythia for backend analysis.

### For Fine-tuning
**TinyLlama is better** - Already instruction-tuned, so fine-tuning builds on a solid foundation.

---

## Test Environment

| Property | Value |
|----------|-------|
| Device | Apple Silicon (MPS) |
| Python | 3.10.18 |
| PyTorch | 2.10.0 |
| Transformers | 5.0.0 |
| Date | 2026-01-26 |

---

## References

- [TinyLlama GitHub](https://github.com/jzhang38/TinyLlama)
- [TinyLlama Paper](https://arxiv.org/abs/2401.02385)
- [Pythia GitHub](https://github.com/EleutherAI/pythia)
- [Pythia Paper](https://arxiv.org/abs/2304.01373)
