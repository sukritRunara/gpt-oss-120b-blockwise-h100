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
    if hasattr(quantizer, "set_global_scale_from"):        # D-010, nvfp4
        quantizer.set_global_scale_from(W)
    quantizer.find_params(W)
    linear.weight.data = quantizer.quantize_dequantize(W).to(linear.weight.dtype)
    return quantizer


def _is_valid_loss(v):
    """Return True if v is a finite float loss (not a sentinel string)."""
    return isinstance(v, float) and not math.isnan(v) and not math.isinf(v)


# ── Hessian cache helpers ─────────────────────────────────────────────────────

def _hessian_cache_dir(cache_root, model_name, dataset, nsamples, seqlen, seed):
    """Return the Path for this specific Hessian cache.

    Relative cache roots are resolved against the repository root (parent of
    opteam-blockwise-gptq/), never the process CWD (P0.1 portability rule).
    """
    from pathlib import Path
    cache_root = Path(cache_root)
    if not cache_root.is_absolute():
        cache_root = Path(__file__).resolve().parents[1] / cache_root
    model_stem = Path(model_name).name or "model"
    model_stem = model_stem.replace("/", "_").replace("\\", "_")
    key = f"{model_stem}_{dataset}_n{nsamples}_s{seqlen}_seed{seed}"
    return cache_root / key


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
    hessian_layer_group_size=1,
    artifact_dir=None,
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
        parallel_hessian: If True (recommended), collect Hessians from the
            original unquantized model (no quantization cascade) in
            memory-bounded layer groups before quantizing anything.
        mixed_precision_threshold: Sublayers whose GPTQ loss exceeds this value
            are kept in BF16. Set to None to quantize everything.
        hessian_cache_dir: Root directory for the Hessian cache. Required in
            parallel mode (the grouped design streams each group to disk to
            bound memory — see P0.4). Relative paths resolve against the repo
            root, not the CWD.
        hessian_layer_group_size: Number of layers whose Hessians are collected
            per full-model calibration pass. Peak accumulator memory scales
            linearly with this; total collection time scales inversely. Start
            at 1 for large MoE models and raise only after measuring headroom.
        artifact_dir: If set (nvfp4 + parallel mode only), exact quantization
            artifacts (E2M1 codes + FP8 scales, P0.6) are streamed per layer
            into this directory as safetensors shards, each tensor verified
            bit-exact against its QDQ weight before the run continues.

    Returns:
        (all_quantizers, all_layer_losses, all_condition_numbers, quant_records)
        quant_records is a list of per-tensor disposition records (P0.5) ready
        for the Stage 5 manifest; empty when artifact_dir is None.
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

    if artifact_dir is not None:
        if quant_format != "nvfp4":
            raise ValueError(
                "artifact_dir (exact code/scale capture, P0.6) is only "
                f"supported for quant_format='nvfp4', got {quant_format!r}."
            )
        if not parallel_hessian:
            raise ValueError(
                "artifact_dir requires parallel_hessian=True — the sequential "
                "debug path does not emit a tensor manifest."
            )

    # Full module paths for manifest records (e.g. "model.layers.7"), never
    # positional guesses — resolved by module identity.
    id_to_name    = {id(m): n for n, m in model.named_modules()}
    layer_prefixes = {i: id_to_name.get(id(l), f"layers.{i}")
                      for i, l in enumerate(layers)}

    all_quantizers        = {}
    all_layer_losses      = {}
    all_condition_numbers = {}
    quant_records         = []

    if parallel_hessian:
        _run_parallel(
            model, layers, arch_type, handler, is_moe, embedding_modules,
            calibration_data, nsamples, device,
            quant_format, blocksize, percdamp, mode, log_condition,
            group_size, nvfp4_block_size,
            mixed_precision_threshold,
            hessian_cache_dir, model_name, dataset, nsamples, seqlen, seed,
            all_quantizers, all_layer_losses, all_condition_numbers,
            hessian_layer_group_size,
            artifact_dir, layer_prefixes, quant_records,
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

    return all_quantizers, all_layer_losses, all_condition_numbers, quant_records


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

def _make_h_hook(acc):
    """Forward hook: feed the module input into a Hessian accumulator."""
    def _hook(module, inp, out):
        acc.add_batch(inp[0], out)
    return _hook


def _collect_hessians_grouped(
    model, layers, handler, calibration_data, device, cache,
    pending, layer_group_size, quant_format, nvfp4_block_size,
):
    """Collect Hessians for `pending` layers in memory-bounded groups (P0.4).

    For each group of <= layer_group_size layers:
      1. Attach lightweight _GptqH accumulators (standard linears) and the
         expert forward patch (MoE layers) to the GROUP ONLY.
      2. Run every calibration sample through the FULL, UNCHANGED model —
         statistics always come from clean source activations (no cascade).
      3. Detach, verify coverage, and stream the group's Hessians to the
         cache (atomic write + manifest entry) before touching the next group.

    Peak accumulator memory is bounded by one group instead of the whole
    model (~51 GB of expert Hessians for GPT-OSS-20B — the P0.4 OOM).
    Costs ceil(len(pending)/layer_group_size) full-model passes in exchange.

    Fail-closed behavior:
      - Any exception during a calibration forward aborts the run (a skipped
        sample would silently give different groups different sample sets).
      - A standard linear with zero samples after a full pass aborts (its
        hook cannot legitimately miss tokens — something is broken).
      - An MoE layer whose patched expert forward was never invoked aborts
        (e.g. a fused-kernel forward bypassed the patch).
      - NaN/Inf Hessians are rejected by the cache at save time.
    """
    import resource
    import time as _time

    from expert_dispatch import _GptqH

    n_passes = (len(pending) + layer_group_size - 1) // layer_group_size
    print(f"[GPTQ] Grouped Hessian collection: {len(pending)} pending layer(s), "
          f"group size {layer_group_size} → {n_passes} full-model pass(es), "
          f"{len(calibration_data)} samples each")

    model.eval()

    for pass_idx in range(n_passes):
        group = pending[pass_idx * layer_group_size:(pass_idx + 1) * layer_group_size]
        t0 = _time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # ── Attach accumulators/hooks to this group only ──────────────────────
        # MoE layers OUTSIDE the group get a passthrough patch pinning them to
        # the same expert-forward implementation the collection patch uses —
        # otherwise transformers' dispatched implementation (batched_mm etc.,
        # ULP-different from the loop) would make downstream activations, and
        # therefore Hessians, depend on which layers are in the group.
        group_state = {}
        handles = []
        passthrough_tokens = {}
        group_set = set(group)
        if handler is not None:
            for li, layer in enumerate(layers):
                if li not in group_set and handler.has_moe(layer):
                    passthrough_tokens[li] = handler.attach_passthrough(layer)

        for li in group:
            layer = layers[li]
            raw_subset = find_layers(layer)
            subset = (handler.filter_standard_layers(layer, raw_subset)
                      if handler is not None else raw_subset)

            acc_map = {}
            for name, linear in subset.items():
                acc = _GptqH(linear.in_features)
                handles.append(linear.register_forward_hook(_make_h_hook(acc)))
                acc_map[name] = acc

            has_experts = (handler is not None and handler.has_moe(layer))
            acc_state = None
            hook_token = None
            if has_experts:
                acc_state = handler.setup_accumulators(
                    layer, device, quant_format, nvfp4_block_size
                )
                hook_token = handler.attach_hooks(layer, acc_state)

            group_state[li] = {
                "acc_map":     acc_map,
                "has_experts": has_experts,
                "acc_state":   acc_state,
                "hook_token":  hook_token,
            }

        # ── One pass over ALL calibration samples, unchanged model ────────────
        try:
            with torch.no_grad():
                for i, (input_ids, _) in enumerate(calibration_data):
                    ids = input_ids.to(device)
                    # Fail closed: a raised sample must abort, not be skipped —
                    # otherwise different groups would accumulate over
                    # different effective sample sets.
                    model(ids, attention_mask=torch.ones_like(ids))
                    if (i + 1) % 64 == 0:
                        print(f"    group {pass_idx + 1}/{n_passes}: "
                              f"{i + 1}/{len(calibration_data)} samples")
        finally:
            for h in handles:
                h.remove()
            for li in group:
                gs = group_state[li]
                if gs["hook_token"] is not None:
                    handler.detach_hooks(layers[li], gs["hook_token"])
            for li, token in passthrough_tokens.items():
                if token is not None:
                    handler.detach_passthrough(layers[li], token)

        # ── Coverage checks (fail closed) ─────────────────────────────────────
        for li in group:
            gs = group_state[li]
            for name, acc in gs["acc_map"].items():
                if acc.nsamples == 0 or acc.H is None:
                    raise RuntimeError(
                        f"Layer {li} sublayer '{name}' received 0 calibration "
                        f"samples after a full-model pass — hook failure or "
                        f"dead module. Refusing to continue (fail closed)."
                    )
            if gs["has_experts"]:
                counter = gs["acc_state"].get("call_counter")
                if counter is not None and counter.get("n", 0) == 0:
                    raise RuntimeError(
                        f"Layer {li}: the patched expert forward was never "
                        f"invoked during calibration — a fused kernel forward "
                        f"(e.g. MegaBlocks via kernelize()) likely bypassed "
                        f"the patch. Expert Hessians would be silently empty. "
                        f"Disable kernelization for calibration."
                    )

        gpu_peak = (torch.cuda.max_memory_allocated()
                    if torch.cuda.is_available() else 0)

        # ── Stream this group to the cache, then free ─────────────────────────
        group_bytes = 0
        for li in group:
            gs = group_state[li]
            payload = {
                "attn": {
                    name: {"H": acc.H.cpu(), "nsamples": acc.nsamples}
                    for name, acc in gs["acc_map"].items()
                },
                "experts": (handler.hessian_state_to_save(gs["acc_state"])
                            if gs["has_experts"] else None),
            }
            cache.save_layer(li, payload)
            group_bytes += cache.manifest["layers"][str(li)]["bytes"]

            for acc in gs["acc_map"].values():
                acc.H = None
            if gs["has_experts"]:
                handler.free_hessians(gs["acc_state"])
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        elapsed = _time.perf_counter() - t0
        host_hwm_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        stats = {
            "layers": list(group),
            "seconds": round(elapsed, 2),
            "gpu_peak_bytes": int(gpu_peak),
            "host_maxrss_kb": int(host_hwm_kb),
            "cache_bytes_written": int(group_bytes),
        }
        cache.record_group_stats(stats)
        print(f"[GPTQ]   group {pass_idx + 1}/{n_passes} (layers {group}): "
              f"{elapsed:.1f}s, GPU peak {gpu_peak / 1024**3:.1f} GB, "
              f"cache +{group_bytes / 1024**2:.0f} MB")


def _run_parallel(
    model, layers, arch_type, handler, is_moe, embedding_modules,
    calibration_data, nsamples, device,
    quant_format, blocksize, percdamp, mode, log_condition,
    group_size, nvfp4_block_size,
    mixed_precision_threshold,
    hessian_cache_dir, model_name, dataset, nsamples_key, seqlen, seed,
    all_quantizers, all_layer_losses, all_condition_numbers,
    hessian_layer_group_size=1,
    artifact_dir=None, layer_prefixes=None, quant_records=None,
):
    """Parallel-Hessian pipeline: grouped collection (P0.4) + quantize-from-cache.

    Phase 1 collects Hessians from the unchanged model in memory-bounded layer
    groups, streaming each group to a manifest-verified on-disk cache (see
    _collect_hessians_grouped). Interrupted runs resume: layers whose cache
    entries are complete (file + SHA-256 verified against the manifest) are
    skipped, and the calibration token stream is reloaded from the immutable
    token cache so every pass sees identical data.

    Phase 2 quantizes layer-by-layer from the cache: GPTQ instances and expert
    shims are created lazily per layer, so at most one layer's Hessians are
    resident at a time.
    """
    from hessian_cache import HessianCache

    if hessian_cache_dir is None:
        raise ValueError(
            "parallel_hessian=True requires hessian_cache_dir: the grouped "
            "collection design streams each layer group to disk to bound "
            "memory (P0.4). Pass a cache directory, or use "
            "parallel_hessian=False (sequential, cascade-prone, debug only)."
        )

    cache_dir = _hessian_cache_dir(
        hessian_cache_dir, model_name, dataset, nsamples_key, seqlen, seed
    )
    cache = HessianCache(
        cache_dir,
        n_layers=len(layers),
        meta={
            "model_name": str(model_name),
            "dataset": dataset,
            "nsamples": int(nsamples_key),
            "seqlen": int(seqlen),
            "seed": int(seed),
        },
    )

    # Immutable, hashed token cache: the first run persists the tokens; any
    # resumed run reloads them (and verifies the hash) so all groups —
    # including ones collected after an interruption — see the same stream.
    calibration_data = cache.ensure_tokens(calibration_data)

    # ── Phase 1: grouped collection of pending layers ─────────────────────────
    pending = cache.pending_layers()
    if pending:
        for mod in embedding_modules:
            mod.to(device)
        _collect_hessians_grouped(
            model, layers, handler, calibration_data, device, cache,
            pending, hessian_layer_group_size, quant_format, nvfp4_block_size,
        )
    else:
        print(f"[GPTQ] Hessian cache complete at {cache_dir} — "
              f"skipping collection.")

    if not cache.is_complete():
        raise RuntimeError(
            f"Hessian cache at {cache_dir} is still incomplete after "
            f"collection — refusing to quantize from a partial cache."
        )
    print(f"[GPTQ] Hessian collection complete "
          f"({cache.total_bytes() / 1024**3:.1f} GB cached). "
          f"Starting quantization...")

    # ── Phase 2: sequential quantization from cache ───────────────────────────
    for layer_idx in range(len(layers)):
        layer = layers[layer_idx]
        layer.to(device)

        raw_subset = find_layers(layer)
        subset = (handler.filter_standard_layers(layer, raw_subset)
                  if handler is not None else raw_subset)
        has_experts = (handler is not None and handler.has_moe(layer))

        print(f"[GPTQ] Layer {layer_idx}/{len(layers)-1}: ", end="", flush=True)
        if is_moe:
            non_exp_cnt = sum(1 for n in subset if "experts" not in n)
            print(f"({non_exp_cnt} attn/gate) ", end="", flush=True)

        payload = cache.load_layer(layer_idx)

        # Lazy per-layer GPTQ instances: only this layer's Hessians are
        # resident. Transplanting (H, nsamples) into GPTQ is numerically
        # identical to native accumulation (test_hessian_canonical.py).
        gptq_map = {}
        for name, linear in subset.items():
            entry = payload["attn"].get(name)
            if entry is None or entry["H"] is None or entry["nsamples"] == 0:
                raise RuntimeError(
                    f"Layer {layer_idx} sublayer '{name}' has no cached "
                    f"Hessian — the cache is inconsistent with this model. "
                    f"Refusing to fall back silently (fail closed)."
                )
            g = GPTQ(linear)
            g.quantizer = _make_quantizer(
                quant_format, device,
                group_size=group_size, nvfp4_block_size=nvfp4_block_size
            )
            g.H = entry["H"].to(device)
            g.nsamples = entry["nsamples"]
            gptq_map[name] = g

        # ── Per-tensor global scales (D-010), fixed BEFORE quantization ───────
        # vLLM fuses q/k/v into one GEMM and applies max(weight_scale_2)
        # WITHOUT rescaling the fp8 group scales, so q/k/v must share one
        # global scale or the fused dequantization drifts.
        if quant_format == "nvfp4":
            qkv = [n for n in gptq_map
                   if n.rsplit(".", 1)[-1] in ("q_proj", "k_proj", "v_proj")]
            shared = None
            if qkv:
                amax = max(subset[n].weight.detach().abs().amax().item()
                           for n in qkv)
                shared = amax / (6.0 * 448.0)
            for name, g in gptq_map.items():
                if name in qkv:
                    g.quantizer.set_global_scale(shared)
                else:
                    g.quantizer.set_global_scale_from(subset[name].weight)

        acc_state = None
        if has_experts:
            if payload.get("experts") is None:
                raise RuntimeError(
                    f"Layer {layer_idx} has MoE experts but the cache holds "
                    f"no expert Hessians — cache/model mismatch (fail closed)."
                )
            acc_state = handler.setup_accumulators(
                layer, device, quant_format, nvfp4_block_size
            )
            handler.load_hessian_state(acc_state, payload["experts"])

        losses             = {}
        layer_cond_numbers = {}
        rtn_count          = 0
        bf16_count         = 0

        capture = artifact_dir is not None
        prefix = (layer_prefixes or {}).get(layer_idx, f"layers.{layer_idx}")
        layer_artifacts = {}      # (full_name, expert_idx|None) → artifact
        layer_records   = []      # manifest records for this layer

        for name, g in gptq_map.items():
            orig_weight = subset[name].weight.data.clone()
            out_f, in_f = g.rows, g.columns
            full_name = f"{prefix}.{name}"

            if capture:
                g.quantizer.begin_capture(out_f, in_f)
            try:
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
            except Exception:
                if capture:
                    g.quantizer.abort_capture()
                raise

            fell_back = (mixed_precision_threshold is not None
                         and _is_valid_loss(loss)
                         and loss > mixed_precision_threshold)

            record = {
                "name": full_name,
                "param": f"{full_name}.weight",
                "kind": "linear",
                "layer_index": layer_idx,
                "projection": name.rsplit(".", 1)[-1],
                "expert_index": None,
                "orig_shape": [out_f, in_f],
                "orientation": "out_in",
                "orig_dtype": str(subset[name].weight.dtype),
                "requested_format": quant_format,
                "disposition": None,
                "reason": None,
                "gptq_blocksize": blocksize,
                "scale_block_size": nvfp4_block_size,
                "loss": loss if _is_valid_loss(loss) else None,
                "normalized_loss": (loss / (out_f * in_f)
                                    if _is_valid_loss(loss) else None),
                "hessian_nsamples": g.nsamples,
                "artifact": None,
            }

            if fell_back:
                subset[name].weight.data.copy_(orig_weight)
                losses[name] = "BF16"
                bf16_count += 1
                if capture:
                    g.quantizer.abort_capture()
                record["disposition"] = "BF16_FALLBACK"
                record["reason"] = (f"loss {loss:.4f} > threshold "
                                    f"{mixed_precision_threshold}")
            else:
                losses[name] = loss
                all_quantizers[f"layer.{layer_idx}.{name}"] = g.quantizer
                record["disposition"] = "GPTQ_NVFP4" if capture else None
                if capture:
                    from quant_artifacts import verify_artifact_matches
                    art = g.quantizer.end_capture()
                    # P0.6 invariant, enforced immediately: the artifact must
                    # reproduce the QDQ weight bit-for-bit.
                    verify_artifact_matches(art, subset[name].weight.data,
                                            what=full_name)
                    layer_artifacts[(full_name, None)] = art
                    record["artifact"] = "pending"   # filled after shard write

            if capture:
                layer_records.append(record)
            del orig_weight
            g.free()

        # Expert quantization
        if has_experts and acc_state is not None:
            expert_losses = handler.quantize(
                layer, acc_state,
                quant_format=quant_format,
                device=device,
                nvfp4_block_size=nvfp4_block_size,
                blocksize=blocksize,
                percdamp=percdamp,
                threshold=mixed_precision_threshold,
                capture_artifacts=capture,
            )
            rtn_delta, bf16_delta = _process_expert_losses(
                losses, expert_losses, handler
            )
            rtn_count  += rtn_delta
            bf16_count += bf16_delta

            if capture:
                expert_records, expert_arts = handler.build_records(
                    layer, expert_losses,
                    layer_idx=layer_idx, prefix=prefix,
                    quant_format=quant_format, blocksize=blocksize,
                    nvfp4_block_size=nvfp4_block_size,
                    acc_state=acc_state,
                )
                layer_records.extend(expert_records)
                layer_artifacts.update(expert_arts)

        # ── Stream this layer's exact artifacts to disk (P0.6) ────────────────
        if capture:
            from quant_artifacts import save_layer_artifacts, artifact_keys
            if layer_artifacts:
                shard = save_layer_artifacts(artifact_dir, layer_idx,
                                             layer_artifacts)
            else:
                shard = None
            for rec in layer_records:
                if rec["artifact"] == "pending":
                    keys = artifact_keys(rec["name"], rec["expert_index"])
                    rec["artifact"] = {"file": shard, **keys}
            quant_records.extend(layer_records)

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

        # Free this layer's Hessians before moving to the next
        for g in gptq_map.values():
            g.H = None
        if handler is not None and acc_state is not None:
            handler.free_hessians(acc_state)

        if torch.cuda.is_available():
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
