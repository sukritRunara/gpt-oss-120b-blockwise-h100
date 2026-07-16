"""
Model structure discovery utilities for GPTQ.

Provides helpers to find linear layers, identify transformer layer lists,
and locate embedding modules for different model architectures.
"""

import torch.nn as nn


def find_layers(module, layer_types=None, prefix=""):
    """Recursively find all layers of specified types in a module.

    Args:
        module: PyTorch module to search.
        layer_types: List of layer types to find. Defaults to [nn.Linear].
        prefix: Dotted path prefix for nested names.

    Returns:
        Dict mapping dotted path names to module instances.
    """
    if layer_types is None:
        layer_types = [nn.Linear]
    layers = {}
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, tuple(layer_types)):
            layers[full_name] = child
        else:
            layers.update(find_layers(child, layer_types, prefix=full_name))
    return layers


def _get_model_type(model):
    """Return model_type string from config, or None."""
    if hasattr(model, "config") and hasattr(model.config, "model_type"):
        return model.config.model_type
    return None


def get_model_layers(model):
    """Return (layers_list, arch_type) for supported architectures.

    Supported:
        - Llama/Mistral/Qwen (dense): model.model.layers, arch="llama"
        - OPT: model.model.decoder.layers, arch="opt"
        - GPT-OSS (MoE): model.model.layers + mlp.experts check, arch="gpt_oss"
        - Qwen3 MoE: model.model.layers, model_type="qwen3_moe", arch="qwen3_moe"
        - DeepSeek V2/V2-Lite (MLA+MoE): model.model.layers,
              model_type="deepseek_v2", arch="deepseek_v2"

    Returns:
        layers: nn.ModuleList of transformer layers.
        arch_type: String identifier for the architecture.
    """
    model_type = _get_model_type(model)

    # Try Llama-style (also covers Mistral, Qwen, TinyLlama, GPT-OSS, DeepSeek)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers

        # DeepSeek V2 / V2-Lite — must be checked BEFORE the generic MoE check
        # because it also has mlp.experts but has a distinct architecture
        # (MLA attention + mixed dense/MoE layers + shared_experts).
        if model_type == "deepseek_v2":
            return layers, "deepseek_v2"

        # Qwen3 MoE — check config.model_type first (most reliable)
        if model_type == "qwen3_moe":
            return layers, "qwen3_moe"

        # Check for MoE experts (GPT-OSS)
        if len(layers) > 0 and hasattr(layers[0], "mlp") and hasattr(layers[0].mlp, "experts"):
            return layers, "gpt_oss"

        return layers, "llama"

    # Try OPT-style
    if hasattr(model, "model") and hasattr(model.model, "decoder"):
        if hasattr(model.model.decoder, "layers"):
            return model.model.decoder.layers, "opt"

    raise ValueError(
        f"Unsupported model architecture: {type(model).__name__}. "
        "Supported: Llama, OPT, GPT-OSS, Qwen3-MoE, DeepSeek-V2."
    )


def get_embedding_layers(model, arch_type):
    """Return list of embedding/norm modules to move to device during calibration.

    These are the modules that run before the first transformer layer and must
    be on-device to produce inputs for layer-by-layer quantization.

    Args:
        model: The full model.
        arch_type: Architecture string from get_model_layers().

    Returns:
        List of nn.Module instances.
    """
    if arch_type in ("llama", "gpt_oss", "qwen3_moe", "deepseek_v2"):
        modules = [model.model.embed_tokens]
        if hasattr(model.model, "norm"):
            modules.append(model.model.norm)
        return modules

    if arch_type == "opt":
        decoder = model.model.decoder
        modules = [decoder.embed_tokens, decoder.embed_positions]
        if hasattr(decoder, "project_in") and decoder.project_in is not None:
            modules.append(decoder.project_in)
        if hasattr(decoder, "project_out") and decoder.project_out is not None:
            modules.append(decoder.project_out)
        return modules

    raise ValueError(f"Unknown arch_type: {arch_type}")