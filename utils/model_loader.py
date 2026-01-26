"""Model loading utilities for billion-llm."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Model registry
MODELS = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "meta-llama/Llama-3.2-1B-Instruct",
    "stabilityai/stablelm-2-1_6b-chat",
    "EleutherAI/pythia-1b",
]

MODEL_ALIASES = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "llama3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama": "meta-llama/Llama-3.2-1B-Instruct",
    "stablelm": "stabilityai/stablelm-2-1_6b-chat",
    "pythia": "EleutherAI/pythia-1b",
    "pythia-1b": "EleutherAI/pythia-1b",
}

# Chat templates for models that need them
CHAT_TEMPLATES = {
    "EleutherAI/pythia-1b": "{% for message in messages %}{% if message['role'] == 'user' %}User: {{ message['content'] }}\n{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}\n{% endif %}{% endfor %}Assistant:",
}


def get_device() -> str:
    """Return the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def resolve_model_name(name: str) -> str:
    """Resolve model alias to full HuggingFace model name."""
    return MODEL_ALIASES.get(name.lower(), name)


def get_quantization_config(quantize: str | None) -> BitsAndBytesConfig | None:
    """Get quantization config for the specified mode."""
    if quantize == "int4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif quantize == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def load_tokenizer(name: str) -> AutoTokenizer:
    """Load tokenizer with proper configuration."""
    model_name = resolve_model_name(name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set chat template for models that don't have one
    if model_name in CHAT_TEMPLATES and tokenizer.chat_template is None:
        tokenizer.chat_template = CHAT_TEMPLATES[model_name]

    return tokenizer


def load_model(
    name: str,
    quantize: str | None = None,
    device: str = "auto",
    torch_dtype: torch.dtype | None = None,
):
    """
    Load a model with automatic device selection and optional quantization.

    Args:
        name: Model name or alias (e.g., "tinyllama", "llama3.2-1b")
        quantize: Quantization mode ("int4", "int8", or None for FP16)
        device: Device to load on ("auto", "cuda", "mps", "cpu")
        torch_dtype: Override torch dtype (default: float16 for GPU, float32 for CPU)

    Returns:
        Loaded model
    """
    model_name = resolve_model_name(name)

    # Determine device
    if device == "auto":
        device = get_device()

    # Set dtype based on device if not specified
    if torch_dtype is None:
        if device == "cpu":
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.float16

    # Get quantization config
    quant_config = None
    if quantize and device != "cpu":
        try:
            quant_config = get_quantization_config(quantize)
        except ImportError:
            print(f"Warning: bitsandbytes not available, loading without quantization")
            quantize = None

    # Load model
    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
    }

    if quant_config:
        load_kwargs["quantization_config"] = quant_config
        load_kwargs["device_map"] = "auto"
    elif device != "cpu":
        load_kwargs["device_map"] = device

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    except torch.cuda.OutOfMemoryError:
        print(f"GPU OOM, falling back to CPU")
        load_kwargs.pop("quantization_config", None)
        load_kwargs.pop("device_map", None)
        load_kwargs["torch_dtype"] = torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    # Move to device if not using device_map
    if "device_map" not in load_kwargs and device != "cpu":
        model = model.to(device)

    return model


def get_model_info(name: str) -> dict:
    """Get information about a model."""
    model_name = resolve_model_name(name)

    info = {
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
            "params": "1.1B",
            "source": "Community",
            "key_feature": "3T tokens overtraining",
            "context_length": 2048,
        },
        "meta-llama/Llama-3.2-1B-Instruct": {
            "params": "1B",
            "source": "Meta",
            "key_feature": "Official Llama ecosystem",
            "context_length": 128000,
        },
        "stabilityai/stablelm-2-1_6b-chat": {
            "params": "1.6B",
            "source": "Stability AI",
            "key_feature": "Multilingual, 2T tokens",
            "context_length": 4096,
        },
        "EleutherAI/pythia-1b": {
            "params": "1.08B",
            "source": "EleutherAI",
            "key_feature": "154 checkpoints for research",
            "context_length": 2048,
        },
    }

    return info.get(model_name, {"params": "Unknown", "source": "Unknown"})
