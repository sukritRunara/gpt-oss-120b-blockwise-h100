"""
Top-level orchestration for GPTQ model quantization.

Coordinates calibration data loading, layer-by-layer Hessian accumulation,
and fasterquant execution.

Two Hessian collection modes
-----------------------------

sequential (legacy, parallel_hessian=False):
    For each layer i, Hessians are collected by running calibration through
    quantized layers 0..i-1. Quantization errors compound layer by layer
    (cascade), making later layers' Hessians increasingly inaccurate.

parallel (parallel_hessian=True, RECOMMENDED):
    All Hessians are collected in a SINGLE forward pass through the
    ORIGINAL unquantized model before any quantization occurs. Each
    layer's Hessian therefore reflects clean BF16 activations, not
    activations corrupted by earlier quantization steps.

    Memory: ~42 GB model (GPU) + ~2 GB current-layer Hessians (RAM).
    Hessians are saved to disk and freed layer-by-layer, so total RAM
    usage is bounded by a single layer rather than the full cache size
    (which can reach 50+ GB for large MoE models).

MoE support
-----------
    Expert-specific Hessian collection and quantization is handled by
    architecture-specific MoEHandler subclasses registered in
    expert_dispatch.py.  apply.py calls the handler interface without
    knowing which architecture is active, so adding a new MoE model only
    requires:
        1. Writing a model-specific expert_gptq module.
        2. Implementing a MoEHandler subclass.
        3. Registering it in expert_dispatch._HANDLER_REGISTRY.

    Currently supported:
        gpt_oss     — GPT-OSS batched-tensor experts (shim + transpose)
        deepseek_v2 — DeepSeek V2/V2-Lite standard nn.Linear experts

Mixed-precision
---------------
    mixed_precision_threshold (default 100.0): any sublayer whose GPTQ loss
    exceeds this value is kept in the original BF16 dtype rather than being
    quantized. Set to None to disable (quantize everything).
"""

import math
import warnings

import torch
import torch.nn as nn

from quantizer import QUANTIZER_REGISTRY
from gptq import GPTQ
from calibration import get_calibration_data, LayerInputCatcher
from model_utils import find_layers, get_model_layers, get_embedding_layers
from expert_dispatch import get_handler

# ── Constants ─────────────────────────────────────────────────────────────────

_GROUP_FORMATS = {"int4"}
_BLOCK_FORMATS = {"nvfp4"}
NVFP4_SCALE_BLOCK_SIZE = 16


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_quantizer(quant_format, device, group_size=128,
                    nvfp4_block_size=NVFP4_SCALE_BLOCK_SIZE):
    """Factory: instantiate the correct quantizer for a given format."""
    if quant_format in _GROUP_FORMATS:
        return QUANTIZER_REGISTRY[quant_format](group_size=group_size, device=device)
    elif quant_format in _BLOCK_FORMATS:
        return QUANTIZER_REGISTRY[quant_format](
            block_size=nvfp4_block_size, device=device
        )
    else:
        return QUANTIZER_REGISTRY[quant_format](device=device)


def _rtn_quantize(linear, quant_format, device, group_size=128,
                  nvfp4_block_size=NVFP4_SCALE_BLOCK_SIZE):
    """Apply RTN (Round-To-Nearest) quantization to a linear layer."""
    quantizer = _make_quantizer(quant_format, device, group_size=group_size,
                                nvfp4_block_size=nvfp4_block_size)
    W = linear.weight.data.float()
    quantizer.find_params(W)
    linear.weight.data = quantizer.quantize_dequantize(W).to(linear.weight.dtype)
    return quantizer


def _is_valid_loss(v):
    """Return True if v is a finite float loss (not a sentinel string)."""
    return isinstance(v, float) and not math.isnan(v) and not math.isinf(v)


# ── Hessian cache helpers ─────────────────────────────────────────────────────

def _hessian_cache_dir(cache_root, model_name, dataset, nsamples, seqlen, seed):
    """Return the Path for this specific Hessian cache."""
    from pathlib import Path
    model_stem = Path(model_name).name or "model"
    model_stem = model_stem.replace("/", "_").replace("\\", "_")
    key = f"{model_stem}_{dataset}_n{nsamples}_s{seqlen}_seed{seed}"
    return Path(cache_root) / key


def _hessian_cache_complete(cache_dir, n_layers):
    """Return True only if every per-layer file and the metadata file exist."""
    import json
    from pathlib import Path
    cache_dir = Path(cache_dir)
    if not (cache_dir / "meta.json").exists():
        return False
    try:
        meta = json.loads((cache_dir / "meta.json").read_text())
        if meta.get("n_layers") != n_layers:
            return False
    except Exception:
        return False
    return all((cache_dir / f"layer_{i:02d}.pt").exists() for i in range(n_layers))


def _save_hessians(cache_dir, layer_data, handler=None, free_after_save=False):
    """Save per-layer Hessians to disk.

    Layout:
        cache_dir/
            meta.json
            layer_00.pt   {"attn": {name: {H, nsamples}},
                           "experts": <handler-specific payload or None>}
            layer_01.pt
            ...

    Args:
        free_after_save: If True, free each layer's Hessian tensors from RAM
            immediately after writing its .pt file. This caps peak RAM at
            ~(single layer size) regardless of total cache size — critical for
            large MoE models where all-layer Hessians can exceed 50+ GB.
            Phase 2 in _run_parallel will reload each layer on demand.
    """
    import json
    from pathlib import Path
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    for layer_idx, ld in layer_data.items():
        payload = {"attn": {}, "experts": None}

        for name, g in ld["gptq_map"].items():
            if g.H is not None:
                payload["attn"][name] = {"H": g.H.cpu(), "nsamples": g.nsamples}
                total_bytes += g.H.numel() * 4

        if handler is not None and ld["has_experts"] and ld["acc_state"] is not None:
            payload["experts"] = handler.hessian_state_to_save(ld["acc_state"])

        torch.save(payload, cache_dir / f"layer_{layer_idx:02d}.pt")

        if free_after_save:
            # Release this layer's Hessians immediately — Phase 2 will reload
            # from the file we just wrote, so nothing is lost.
            for g in ld["gptq_map"].values():
                g.H = None
            if handler is not None and ld["has_experts"] \
                    and ld["acc_state"] is not None:
                handler.free_hessians(ld["acc_state"])
            torch.cuda.empty_cache()

    (cache_dir / "meta.json").write_text(
        json.dumps({"n_layers": len(layer_data)}, indent=2)
    )
    print(f"[GPTQ] Hessians saved to {cache_dir}  "
          f"({total_bytes / 1024**3:.1f} GB total, "
          f"{'streamed per-layer' if free_after_save else 'all in RAM'})")


def _load_hessians(cache_dir, layer_data, handler=None):
    """Load per-layer Hessians from disk into layer_data structures.

    Tensors are loaded on CPU; _run_parallel moves them to device as needed.
    """
    from pathlib import Path
    cache_dir = Path(cache_dir)

    for layer_idx, ld in layer_data.items():
        payload = torch.load(
            cache_dir / f"layer_{layer_idx:02d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        for name, g in ld["gptq_map"].items():
            if name in payload["attn"]:
                g.H        = payload["attn"][name]["H"]
                g.nsamples = payload["attn"][name]["nsamples"]

        if (handler is not None
                and ld["has_experts"]
                and payload.get("experts") is not None):
            handler.load_hessian_state(ld["acc_state"], payload["experts"])

    print(f"[GPTQ] Hessians loaded from {cache_dir}")


# ── Main entry point ──────────────────────────────────────────────────────────

def gptq_quantize_model(
    model,
    model_name,
    quant_format="fp8",
    dataset="wikitext2",
    nsamples=128,
    seqlen=2048,
    blocksize=128,
    percdamp=0.01,
    seed=0,
    device="cuda",
    group_size=128,
    nvfp4_block_size=NVFP4_SCALE_BLOCK_SIZE,
    mode="standard",
    log_condition=False,
    parallel_hessian=True,
    mixed_precision_threshold=100.0,
    hessian_cache_dir="hessian_cache",
):
    """Quantize all linear layers in a model using GPTQ.

    Args:
        model: HuggingFace causal LM model (quantized in-place).
        model_name: Full path / HF hub ID used to load calibration tokenizer.
        quant_format: "nvfp4", "fp8", "int8", or "int4".
        dataset: Calibration dataset ("wikitext2" or "c4").
        nsamples: Number of calibration samples.
        seqlen: Sequence length per sample.
        blocksize: GPTQ block width (must be multiple of nvfp4_block_size for nvfp4).
        percdamp: Hessian damping factor.
        seed: RNG seed for calibration data sampling.
        device: Compute device.
        group_size: Group size for "int4".
        nvfp4_block_size: Microscaling block size for "nvfp4" (hardware-fixed at 16).
        mode: "standard" or "blockwise" (nvfp4 requires "blockwise").
        log_condition: Log per-block condition numbers (blockwise mode only).
        parallel_hessian: If True (recommended), collect ALL layer Hessians from
            the original unquantized model in one forward pass before quantizing.
        mixed_precision_threshold: Sublayers whose GPTQ loss exceeds this value
            are kept in BF16. Set to None to quantize everything.
        hessian_cache_dir: Root directory for Hessian cache. Set to None to disable.

    Returns:
        (all_quantizers, all_layer_losses, all_condition_numbers)
        Model weights are updated in-place.
    """
    if quant_format not in QUANTIZER_REGISTRY:
        raise ValueError(
            f"Unknown quant format: {quant_format!r}. "
            f"Available: {list(QUANTIZER_REGISTRY.keys())}"
        )
    if quant_format == "nvfp4":
        if mode != "blockwise":
            raise ValueError(
                "nvfp4 requires mode='blockwise'. "
                "Microscaling scales are computed per GPTQ block."
            )
        if blocksize % nvfp4_block_size != 0:
            raise ValueError(
                f"blocksize={blocksize} must be a multiple of "
                f"nvfp4_block_size={nvfp4_block_size}."
            )

    if mixed_precision_threshold is not None:
        print(f"[GPTQ] Mixed-precision threshold: {mixed_precision_threshold} "
              f"(layers above this loss kept in BF16)")

    print(
        f"[GPTQ] Loading calibration data: "
        f"{dataset}, {nsamples} samples, seqlen={seqlen}"
    )
    calibration_data = get_calibration_data(model_name, dataset, nsamples, seqlen, seed)

    print(f"[GPTQ] Identifying model structure...")
    layers, arch_type = get_model_layers(model)
    print(f"[GPTQ] Architecture: {arch_type}, {len(layers)} layers")

    handler          = get_handler(arch_type)   # MoEHandler or None
    is_moe           = arch_type in ("qwen3_moe", "gpt_oss", "deepseek_v2")
    embedding_modules = get_embedding_layers(model, arch_type)

    all_quantizers        = {}
    all_layer_losses      = {}
    all_condition_numbers = {}

    if parallel_hessian:
        _run_parallel(
            model, layers, arch_type, handler, is_moe, embedding_modules,
            calibration_data, nsamples, device,
            quant_format, blocksize, percdamp, mode, log_condition,
            group_size, nvfp4_block_size,
            mixed_precision_threshold,
            hessian_cache_dir, model_name, dataset, nsamples, seqlen, seed,
            all_quantizers, all_layer_losses, all_condition_numbers,
        )
    else:
        _run_sequential(
            model, layers, arch_type, handler, is_moe, embedding_modules,
            calibration_data, nsamples, device,
            model.config.hidden_size, next(model.parameters()).dtype, seqlen,
            quant_format, blocksize, percdamp, mode, log_condition,
            group_size, nvfp4_block_size,
            mixed_precision_threshold,
            all_quantizers, all_layer_losses, all_condition_numbers,
        )

    # ── Final summary ─────────────────────────────────────────────────────────
    attn_names = {
        "self_attn.q_proj", "self_attn.k_proj",
        "self_attn.v_proj", "self_attn.o_proj",
    }

    attn_nvfp4 = 0; attn_bf16 = 0; attn_rtn = 0

    # Expert counts — accumulated from "_expert_summary" stored per-layer
    # by the per-layer quantization code below.
    exp_a_nvfp4 = 0; exp_a_bf16 = 0; exp_a_rtn = 0  # gate_up / combined
    exp_b_nvfp4 = 0; exp_b_bf16 = 0; exp_b_rtn = 0  # down_proj / combined

    for layer_idx, losses in all_layer_losses.items():
        for name, v in losses.items():
            if name.startswith("_"):
                continue  # skip internal sentinels
            if name in attn_names:
                if _is_valid_loss(v):  attn_nvfp4 += 1
                elif v == "BF16":      attn_bf16  += 1
                elif v == "RTN":       attn_rtn   += 1

        # Per-layer expert summary written by _process_expert_losses()
        if "_expert_summary" in losses:
            s = losses["_expert_summary"]
            exp_a_nvfp4 += s["n_a_q"];  exp_a_bf16 += s["n_a_bf16"]; exp_a_rtn += s["n_a_rtn"]
            exp_b_nvfp4 += s["n_b_q"];  exp_b_bf16 += s["n_b_bf16"]; exp_b_rtn += s["n_b_rtn"]

    def _pct(a, total):
        return 100.0 * a / total if total > 0 else 0.0

    attn_total = attn_nvfp4 + attn_bf16 + attn_rtn
    a_total    = exp_a_nvfp4 + exp_a_bf16 + exp_a_rtn
    b_total    = exp_b_nvfp4 + exp_b_bf16 + exp_b_rtn

    # Label the expert rows based on architecture
    if arch_type == "gpt_oss":
        row_a_label = "Expert gate_up"
        row_b_label = "Expert down_proj"
    elif arch_type == "deepseek_v2":
        row_a_label = "Expert gate+up proj"
        row_b_label = "Expert down_proj"
    else:
        row_a_label = "Expert proj (a)"
        row_b_label = "Expert proj (b)"

    grand_nvfp4 = attn_nvfp4 + exp_a_nvfp4 + exp_b_nvfp4
    grand_bf16  = attn_bf16  + exp_a_bf16  + exp_b_bf16
    grand_rtn   = attn_rtn   + exp_a_rtn   + exp_b_rtn
    grand_total = grand_nvfp4 + grand_bf16 + grand_rtn

    print()
    print("=" * 72)
    print("[GPTQ] Quantization Summary")
    print("=" * 72)
    print(f"  {'Category':<26} {'Total':>6} {'NVFP4':>8} {'BF16':>8} {'RTN':>6}")
    print(f"  {'-'*26} {'-'*6} {'-'*8} {'-'*8} {'-'*6}")
    print(f"  {'Attention sublayers':<26} {attn_total:>6d} "
          f"{attn_nvfp4:>5d} ({_pct(attn_nvfp4, attn_total):4.1f}%) "
          f"{attn_bf16:>5d} ({_pct(attn_bf16, attn_total):4.1f}%) "
          f"{attn_rtn:>4d}")
    if a_total > 0:
        print(f"  {row_a_label:<26} {a_total:>6d} "
              f"{exp_a_nvfp4:>5d} ({_pct(exp_a_nvfp4, a_total):4.1f}%) "
              f"{exp_a_bf16:>5d} ({_pct(exp_a_bf16, a_total):4.1f}%) "
              f"{exp_a_rtn:>4d}")
        print(f"  {row_b_label:<26} {b_total:>6d} "
              f"{exp_b_nvfp4:>5d} ({_pct(exp_b_nvfp4, b_total):4.1f}%) "
              f"{exp_b_bf16:>5d} ({_pct(exp_b_bf16, b_total):4.1f}%) "
              f"{exp_b_rtn:>4d}")
    print(f"  {'-'*26} {'-'*6} {'-'*8} {'-'*8} {'-'*6}")
    print(f"  {'TOTAL':<26} {grand_total:>6d} "
          f"{grand_nvfp4:>5d} ({_pct(grand_nvfp4, grand_total):4.1f}%) "
          f"{grand_bf16:>5d} ({_pct(grand_bf16, grand_total):4.1f}%) "
          f"{grand_rtn:>4d}")
    print("=" * 72)
    if mixed_precision_threshold is not None and grand_bf16 > 0:
        print(f"  Note: {grand_bf16} sublayer(s) kept BF16 "
              f"(loss > {mixed_precision_threshold}).")
    print()

    return all_quantizers, all_layer_losses, all_condition_numbers


# ── Shared expert-loss accounting ─────────────────────────────────────────────

def _process_expert_losses(losses, expert_losses, handler):
    """Decode expert_losses via handler, write sentinels and summary into losses.

    Writes into losses:
        "experts.gate_up"   — avg gate_up loss (or "BF16"/"RTN" sentinel)
        "experts.down"      — avg down loss    (or "BF16"/"RTN" sentinel)
        "_expert_summary"   — dict used by the final summary table

    Returns (rtn_count_delta, bf16_count_delta).
    """
    n_q, n_bf16, n_rtn, avg_a, avg_b = handler.summarize_losses(expert_losses)
    num_exp = 0
    # Count total projections processed (architecture-agnostic)
    # For GPT-OSS: n_q = number of gate_up experts quantized (same for down)
    # For DeepSeek V2: n_q = total linear projections quantized

    no_a_quantized = (n_q == 0)
    losses["experts.gate_up"] = "BF16" if no_a_quantized else avg_a
    losses["experts.down"]    = "BF16" if no_a_quantized else avg_b

    # For the summary table we need per-projection counts.
    # GptOssHandler returns n_q = number of gate_up experts quantized.
    # DeepSeekV2Handler returns n_q = total projections across all experts.
    # We split into (a, b) rows: GPT-OSS a=gate_up, b=down; DeepSeek combined.
    from expert_dispatch import GptOssHandler, DeepSeekV2Handler
    if isinstance(handler, GptOssHandler):
        # n_q is per-projection (gate_up only); dn is separate but same count
        gu = expert_losses["gu"]
        dn = expert_losses["dn"]
        n_a_q   = sum(1 for v in gu.values() if _is_valid_loss(v))
        n_a_bf16 = sum(1 for v in gu.values() if v == "BF16")
        n_a_rtn  = sum(1 for v in gu.values() if v == "RTN")
        n_b_q   = sum(1 for v in dn.values() if _is_valid_loss(v))
        n_b_bf16 = sum(1 for v in dn.values() if v == "BF16")
        n_b_rtn  = sum(1 for v in dn.values() if v == "RTN")
    else:
        # DeepSeek: split projections evenly between "a" (gate+up) and "b" (down)
        # Each routed expert has 3 projections; we attribute 2 to a, 1 to b.
        # Simpler: treat n_q / n_bf16 / n_rtn as combined across both rows.
        n_a_q    = n_q   // 3 * 2  # gate_proj + up_proj per expert
        n_b_q    = n_q   // 3      # down_proj per expert
        n_a_bf16 = n_bf16 // 3 * 2
        n_b_bf16 = n_bf16 // 3
        n_a_rtn  = n_rtn  // 3 * 2
        n_b_rtn  = n_rtn  // 3

    losses["_expert_summary"] = {
        "n_a_q": n_a_q, "n_a_bf16": n_a_bf16, "n_a_rtn": n_a_rtn,
        "n_b_q": n_b_q, "n_b_bf16": n_b_bf16, "n_b_rtn": n_b_rtn,
    }

    return n_rtn, n_bf16


# ── Parallel Hessian collection path ─────────────────────────────────────────

def _run_parallel(
    model, layers, arch_type, handler, is_moe, embedding_modules,
    calibration_data, nsamples, device,
    quant_format, blocksize, percdamp, mode, log_condition,
    group_size, nvfp4_block_size,
    mixed_precision_threshold,
    hessian_cache_dir, model_name, dataset, nsamples_key, seqlen, seed,
    all_quantizers, all_layer_losses, all_condition_numbers,
):
    # ── Resolve cache path upfront ────────────────────────────────────────────
    # Must be done BEFORE Phase 1 setup so we know whether to allocate expert
    # Hessian accumulators (which can be 50+ GB for large MoE models).
    cache_dir = (
        _hessian_cache_dir(
            hessian_cache_dir, model_name, dataset,
            nsamples_key, seqlen, seed
        )
        if hessian_cache_dir is not None
        else None
    )
    use_cache = (cache_dir is not None
                 and _hessian_cache_complete(cache_dir, len(layers)))

    # ── Phase 1: setup ────────────────────────────────────────────────────────
    # When using cache, skip expert accumulator allocation entirely — they will
    # be created lazily one layer at a time in Phase 2.  This prevents
    # allocating 50+ GB of zero Hessian matrices upfront.
    print("[GPTQ] Parallel mode — setting up Hessian accumulators for all layers...")

    layer_data  = {}
    all_handles = []

    def make_hook(g):
        def _hook(module, inp, out):
            g.add_batch(inp[0], out)
        return _hook

    for layer_idx, layer in enumerate(layers):
        raw_subset = find_layers(layer)
        # For DeepSeek V2, filter out expert linears handled by the handler
        subset = (handler.filter_standard_layers(layer, raw_subset)
                  if handler is not None else raw_subset)

        gptq_map = {}
        for name, linear in subset.items():
            g = GPTQ(linear)
            g.quantizer = _make_quantizer(
                quant_format, device,
                group_size=group_size, nvfp4_block_size=nvfp4_block_size
            )
            if not use_cache:
                handle = linear.register_forward_hook(make_hook(g))
                all_handles.append(handle)
            gptq_map[name] = g

        has_experts = (handler is not None and handler.has_moe(layer))
        acc_state   = None
        hook_token  = None

        # Only allocate expert Hessian accumulators when we need the forward
        # pass (i.e. no valid cache).  When cache exists, acc_state is created
        # lazily per-layer in Phase 2 to keep RAM bounded.
        if has_experts and not use_cache:
            acc_state  = handler.setup_accumulators(
                layer, device, quant_format, nvfp4_block_size
            )
            hook_token = handler.attach_hooks(layer, acc_state)

        layer_data[layer_idx] = {
            "subset":      subset,
            "gptq_map":    gptq_map,
            "has_experts": has_experts,
            "acc_state":   acc_state,
            "hook_token":  hook_token,
        }

    # ── Phase 1: forward pass (or skip if cache is valid) ────────────────────
    if use_cache:
        print(f"[GPTQ] Found Hessian cache at {cache_dir} — skipping forward pass.")
        # Hooks were never registered; nothing to remove.
    else:
        print(
            f"[GPTQ] Parallel mode — running {nsamples_key} samples through original "
            f"BF16 model to collect all Hessians..."
        )
        for mod in embedding_modules:
            mod.to(device)

        model.eval()
        with torch.no_grad():
            for i, (input_ids, _) in enumerate(calibration_data):
                try:
                    ids = input_ids.to(device)
                    model(ids, attention_mask=torch.ones_like(ids))
                except Exception as exc:
                    warnings.warn(
                        f"[GPTQ] Sample {i} raised {type(exc).__name__}: {exc}; skipping."
                    )
                if (i + 1) % 64 == 0:
                    print(f"  {i + 1}/{nsamples_key} samples processed")

        for mod in embedding_modules:
            mod.cpu()

        for h in all_handles:
            h.remove()

        for layer_idx, ld in layer_data.items():
            if ld["hook_token"] is not None:
                handler.detach_hooks(layers[layer_idx], ld["hook_token"])

        torch.cuda.empty_cache()

        if cache_dir is not None:
            print(f"[GPTQ] Saving Hessians to {cache_dir} (streaming per-layer to cap RAM)...")
            _save_hessians(cache_dir, layer_data, handler, free_after_save=True)

    print("[GPTQ] Hessian collection complete. Starting quantization...")

    # ── Phase 2: sequential quantization ─────────────────────────────────────
    for layer_idx in range(len(layers)):
        layer = layers[layer_idx]
        layer.to(device)

        ld       = layer_data[layer_idx]
        subset   = ld["subset"]
        gptq_map = ld["gptq_map"]

        print(f"[GPTQ] Layer {layer_idx}/{len(layers)-1}: ", end="", flush=True)
        if is_moe:
            non_exp_cnt = sum(1 for n in subset if "experts" not in n)
            print(f"({non_exp_cnt} attn/gate) ", end="", flush=True)

        # When cache was used (acc_state is None), create expert accumulators
        # now for just this one layer before loading Hessians from disk.
        if use_cache and ld["has_experts"] and ld["acc_state"] is None:
            ld["acc_state"] = handler.setup_accumulators(
                layer, device, quant_format, nvfp4_block_size
            )

        # Reload Hessians from cache on demand (covers both the streaming-save
        # path where g.H was freed, and the cache-hit path where g.nsamples==0
        # because GPTQ.__init__ uses a zero tensor rather than None for H).
        if cache_dir is not None and any(
            g.H is None or g.nsamples == 0 for g in gptq_map.values()
        ):
            payload = torch.load(
                cache_dir / f"layer_{layer_idx:02d}.pt",
                map_location="cpu",
                weights_only=False,
            )
            for name, g in gptq_map.items():
                if name in payload["attn"]:
                    g.H        = payload["attn"][name]["H"]
                    g.nsamples = payload["attn"][name]["nsamples"]
            if (handler is not None and ld["has_experts"]
                    and ld["acc_state"] is not None
                    and payload.get("experts") is not None):
                handler.load_hessian_state(ld["acc_state"], payload["experts"])

        for g in gptq_map.values():
            if g.H is not None:
                g.H = g.H.to(device)

        losses             = {}
        layer_cond_numbers = {}
        rtn_count          = 0
        bf16_count         = 0

        for name, g in gptq_map.items():
            if g.nsamples == 0:
                quantizer = _rtn_quantize(
                    subset[name], quant_format, device,
                    group_size=group_size, nvfp4_block_size=nvfp4_block_size
                )
                all_quantizers[f"layer.{layer_idx}.{name}"] = quantizer
                losses[name] = "RTN"
                rtn_count += 1
                g.free()
            else:
                orig_weight = subset[name].weight.data.clone()

                if mode == "blockwise":
                    if log_condition:
                        loss, cond_nums = g.fasterquant_blockwise(
                            blocksize=blocksize, percdamp=percdamp,
                            log_condition=True
                        )
                        layer_cond_numbers[name] = cond_nums
                    else:
                        loss = g.fasterquant_blockwise(
                            blocksize=blocksize, percdamp=percdamp
                        )
                else:
                    loss = g.fasterquant(blocksize=blocksize, percdamp=percdamp)

                if (mixed_precision_threshold is not None
                        and _is_valid_loss(loss)
                        and loss > mixed_precision_threshold):
                    subset[name].weight.data.copy_(orig_weight)
                    losses[name] = "BF16"
                    bf16_count += 1
                else:
                    losses[name] = loss
                    all_quantizers[f"layer.{layer_idx}.{name}"] = g.quantizer

                del orig_weight
                g.free()

        # Expert quantization
        if ld["has_experts"] and ld["acc_state"] is not None:
            expert_losses = handler.quantize(
                layer, ld["acc_state"],
                quant_format=quant_format,
                device=device,
                nvfp4_block_size=nvfp4_block_size,
                blocksize=blocksize,
                percdamp=percdamp,
                threshold=mixed_precision_threshold,
            )
            rtn_delta, bf16_delta = _process_expert_losses(
                losses, expert_losses, handler
            )
            rtn_count  += rtn_delta
            bf16_count += bf16_delta

        all_layer_losses[layer_idx] = losses
        if layer_cond_numbers:
            all_condition_numbers[layer_idx] = layer_cond_numbers

        loss_strs = []
        for n, l in losses.items():
            if n.startswith("_"):
                continue
            if l == "BF16":
                loss_strs.append(f"{n}=BF16")
            elif _is_valid_loss(l):
                loss_strs.append(f"{n}={l:.2f}")
        if rtn_count > 0:
            loss_strs.append(f"rtn({rtn_count})")
        if bf16_count > 0:
            loss_strs.append(f"bf16({bf16_count})")
        print(", ".join(loss_strs))

        layer.cpu()

        # Free this layer's Hessians from CPU/GPU RAM
        for g in ld["gptq_map"].values():
            g.H = None
        if handler is not None and ld["acc_state"] is not None:
            handler.free_hessians(ld["acc_state"])
        del layer_data[layer_idx]

        torch.cuda.empty_cache()


# ── Sequential (legacy) path ──────────────────────────────────────────────────

def _run_sequential(
    model, layers, arch_type, handler, is_moe, embedding_modules,
    calibration_data, nsamples, device,
    hidden_size, model_dtype, seqlen,
    quant_format, blocksize, percdamp, mode, log_condition,
    group_size, nvfp4_block_size,
    mixed_precision_threshold,
    all_quantizers, all_layer_losses, all_condition_numbers,
):
    """Original sequential Hessian collection (cascade-prone).

    Retained for debugging / ablation. For production use parallel_hessian=True.
    """
    warnings.warn(
        "[GPTQ] parallel_hessian=False: using sequential mode. "
        "Hessians for later layers will be computed from quantized activations, "
        "causing the NaN cascade on deep MoE models. "
        "Set parallel_hessian=True to fix.",
        UserWarning,
        stacklevel=3,
    )

    for mod in embedding_modules:
        mod.to(device)

    print(f"[GPTQ] Capturing first-layer inputs...")
    catcher = LayerInputCatcher(
        layers[0], nsamples, hidden_size, seqlen, model_dtype, device
    )
    layers[0] = catcher

    model.eval()
    with torch.no_grad():
        for i, (input_ids, _) in enumerate(calibration_data):
            try:
                ids = input_ids.to(device)
                model(ids, attention_mask=torch.ones_like(ids))
            except ValueError:
                pass

    layers[0] = catcher.module
    inps         = catcher.inps
    layer_kwargs = catcher.kwargs
    layer_kwargs.pop("past_key_values", None)

    for mod in embedding_modules:
        mod.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)

    def make_hook(g):
        def _hook(module, inp, out):
            g.add_batch(inp[0], out)
        return _hook

    for layer_idx in range(len(layers)):
        layer = layers[layer_idx]
        layer.to(device)

        print(f"[GPTQ] Layer {layer_idx}/{len(layers)-1}: ", end="", flush=True)

        raw_subset = find_layers(layer)
        subset = (handler.filter_standard_layers(layer, raw_subset)
                  if handler is not None else raw_subset)

        if is_moe:
            non_exp_cnt = sum(1 for n in subset if "experts" not in n)
            print(f"({non_exp_cnt} attn/gate) ", end="", flush=True)

        # Expert setup — before calibration forward passes
        has_experts = (handler is not None and handler.has_moe(layer))
        acc_state   = None
        hook_token  = None

        if has_experts:
            acc_state  = handler.setup_accumulators(
                layer, device, quant_format, nvfp4_block_size
            )
            hook_token = handler.attach_hooks(layer, acc_state)

        gptq_instances = {}
        for name, linear in subset.items():
            g = GPTQ(linear)
            g.quantizer = _make_quantizer(
                quant_format, device,
                group_size=group_size, nvfp4_block_size=nvfp4_block_size
            )
            gptq_instances[name] = g

        attn_handles = []
        for name, g in gptq_instances.items():
            handle = subset[name].register_forward_hook(make_hook(g))
            attn_handles.append(handle)

        with torch.no_grad():
            for j in range(nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]

        for h in attn_handles:
            h.remove()
        if hook_token is not None:
            handler.detach_hooks(layer, hook_token)

        losses             = {}
        layer_cond_numbers = {}
        rtn_count          = 0
        bf16_count         = 0

        for name, g in gptq_instances.items():
            if g.nsamples == 0:
                quantizer = _rtn_quantize(
                    subset[name], quant_format, device,
                    group_size=group_size, nvfp4_block_size=nvfp4_block_size
                )
                all_quantizers[f"layer.{layer_idx}.{name}"] = quantizer
                losses[name] = "RTN"
                rtn_count += 1
                g.free()
            else:
                orig_weight = subset[name].weight.data.clone()

                if mode == "blockwise":
                    if log_condition:
                        loss, cond_nums = g.fasterquant_blockwise(
                            blocksize=blocksize, percdamp=percdamp,
                            log_condition=True
                        )
                        layer_cond_numbers[name] = cond_nums
                    else:
                        loss = g.fasterquant_blockwise(
                            blocksize=blocksize, percdamp=percdamp
                        )
                else:
                    loss = g.fasterquant(blocksize=blocksize, percdamp=percdamp)

                if (mixed_precision_threshold is not None
                        and _is_valid_loss(loss)
                        and loss > mixed_precision_threshold):
                    subset[name].weight.data.copy_(orig_weight)
                    losses[name] = "BF16"
                    bf16_count += 1
                else:
                    losses[name] = loss
                    all_quantizers[f"layer.{layer_idx}.{name}"] = g.quantizer

                del orig_weight
                g.free()

        if has_experts and acc_state is not None:
            expert_losses = handler.quantize(
                layer, acc_state,
                quant_format=quant_format,
                device=device,
                nvfp4_block_size=nvfp4_block_size,
                blocksize=blocksize,
                percdamp=percdamp,
                threshold=mixed_precision_threshold,
            )
            rtn_delta, bf16_delta = _process_expert_losses(
                losses, expert_losses, handler
            )
            rtn_count  += rtn_delta
            bf16_count += bf16_delta

        all_layer_losses[layer_idx] = losses
        if layer_cond_numbers:
            all_condition_numbers[layer_idx] = layer_cond_numbers

        loss_strs = []
        for n, l in losses.items():
            if n.startswith("_"):
                continue
            if l == "BF16":
                loss_strs.append(f"{n}=BF16")
            elif _is_valid_loss(l):
                loss_strs.append(f"{n}={l:.2f}")
        if rtn_count > 0:
            loss_strs.append(f"rtn({rtn_count})")
        if bf16_count > 0:
            loss_strs.append(f"bf16({bf16_count})")
        print(", ".join(loss_strs))

        with torch.no_grad():
            for j in range(nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]

        layer.cpu()
        torch.cuda.empty_cache()
        inps, outs = outs, inps
