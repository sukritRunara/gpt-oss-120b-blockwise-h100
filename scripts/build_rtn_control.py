#!/usr/bin/env python
"""Build the matched RTN NVFP4 control (arm C, handoff §15).

First-class RTN — NOT the "expert got no calibration" fallback. Uses the
exact same machinery as the GPTQ path so the ONLY difference is the
quantization algorithm:
  - same NVFP4Quantizer (E2M1 grid, fp8 block scales, D-010 per-tensor
    global scales, q/k/v sharing one global scale)
  - same exact-artifact capture + bit-exact verification (P0.6)
  - same manifest schema (dispositions RTN_NVFP4) and the same Stage 7
    exporter / vLLM packing path
  - same tensor mask: by default every eligible tensor; with
    --match_manifest, EXACTLY the mask of a GPTQ run's manifest (tensors
    that fell back to BF16 there stay BF16 here).

Run inside .venv-quant:
    python scripts/build_rtn_control.py \\
        --source /workspace/models/gpt-oss-20b-mxfp4-dequant-bf16 \\
        --output /workspace/models/gpt-oss-20b-mxfp4-dequant-rtn-nvfp4 \\
        [--match_manifest <gptq-qdq>/quant_artifacts/manifest.json] \\
        [--pack]
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

REPO = Path("/workspace/blockwise-gptq-main")
sys.path.insert(0, str(REPO / "opteam-blockwise-gptq"))


def _rtn_tensor(quantizer, W_out_in):
    """RTN-quantize an [out, in] fp32 tensor with capture. Returns (qdq, art)."""
    out_f, in_f = W_out_in.shape
    quantizer.set_global_scale_from(W_out_in)
    quantizer.begin_capture(out_f, in_f)
    quantizer.find_params(W_out_in)
    qdq = quantizer.quantize_dequantize(W_out_in)
    return qdq, quantizer.end_capture()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path,
                    default=Path("/workspace/models/gpt-oss-20b-mxfp4-dequant-bf16"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--match_manifest", type=Path, default=None,
                    help="GPTQ manifest whose tensor mask to replicate exactly")
    ap.add_argument("--pack", action="store_true",
                    help="Run Stage 7 into <output>-packed afterwards")
    ap.add_argument("--nvfp4_block_size", type=int, default=16)
    args = ap.parse_args()

    from quantizer import NVFP4Quantizer
    from quant_artifacts import (
        save_layer_artifacts, artifact_keys, verify_artifact_matches,
        write_quant_manifest,
    )
    from model_utils import get_model_layers, find_layers
    from expert_dispatch import get_handler

    # Mask from a GPTQ manifest, if matching
    forced_bf16 = set()
    if args.match_manifest:
        gptq_manifest = json.loads(args.match_manifest.read_text())
        for r in gptq_manifest["tensors"]:
            if r["disposition"] == "BF16_FALLBACK":
                forced_bf16.add((r["name"], r["expert_index"]))
        print(f"Matching GPTQ mask: {len(forced_bf16)} forced-BF16 tensors")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading source {args.source} …")
    model = AutoModelForCausalLM.from_pretrained(
        str(args.source), torch_dtype=torch.bfloat16, device_map="auto",
        low_cpu_mem_usage=True)
    model.eval()
    device = next(model.parameters()).device

    layers, arch = get_model_layers(model)
    handler = get_handler(arch)
    id_to_name = {id(m): n for n, m in model.named_modules()}

    art_dir = args.output / "quant_artifacts"
    records = []

    for li, layer in enumerate(layers):
        prefix = id_to_name.get(id(layer), f"layers.{li}")
        raw = find_layers(layer)
        subset = (handler.filter_standard_layers(layer, raw)
                  if handler is not None else raw)
        layer_arts = {}

        # Shared q/k/v global scale (D-010; vLLM fuses them)
        qkv = [n for n in subset
               if n.rsplit(".", 1)[-1] in ("q_proj", "k_proj", "v_proj")]
        shared = (max(subset[n].weight.detach().abs().amax().item()
                      for n in qkv) / (6.0 * 448.0)) if qkv else None

        for name, linear in subset.items():
            full = f"{prefix}.{name}"
            out_f, in_f = linear.weight.shape
            rec = {
                "name": full, "param": f"{full}.weight", "kind": "linear",
                "layer_index": li, "projection": name.rsplit(".", 1)[-1],
                "expert_index": None, "orig_shape": [out_f, in_f],
                "orientation": "out_in", "orig_dtype": str(linear.weight.dtype),
                "requested_format": "nvfp4", "gptq_blocksize": None,
                "scale_block_size": args.nvfp4_block_size,
                "loss": None, "normalized_loss": None,
                "hessian_nsamples": None, "reason": None, "artifact": None,
            }
            if (full, None) in forced_bf16:
                rec["disposition"] = "BF16_FALLBACK"
                rec["reason"] = "matched GPTQ mask (BF16 there)"
                records.append(rec)
                continue
            q = NVFP4Quantizer(block_size=args.nvfp4_block_size, device=device)
            if name in qkv:
                q.set_global_scale(shared)
                W = linear.weight.data.float()
                q.begin_capture(out_f, in_f)
                q.find_params(W)
                qdq = q.quantize_dequantize(W)
                art = q.end_capture()
            else:
                qdq, art = _rtn_tensor(q, linear.weight.data.float())
            linear.weight.data.copy_(qdq.to(linear.weight.dtype))
            verify_artifact_matches(art, linear.weight.data, what=full)
            layer_arts[(full, None)] = art
            rec["disposition"] = "RTN_NVFP4"
            rec["artifact"] = "pending"
            records.append(rec)

        if handler is not None and handler.has_moe(layer):
            experts = layer.mlp.experts
            for proj, attr in (("gate_up", "gate_up_proj"),
                               ("down", "down_proj")):
                batched = getattr(experts, attr)
                pname = f"{prefix}.mlp.experts.{attr}"
                E = batched.shape[0]
                for e in range(E):
                    w_in_out = batched.data[e]
                    out_f, in_f = w_in_out.shape[1], w_in_out.shape[0]
                    rec = {
                        "name": pname, "param": pname, "kind": "expert_slice",
                        "layer_index": li, "projection": proj,
                        "expert_index": e, "orig_shape": [out_f, in_f],
                        "orientation": "transposed_out_in",
                        "orig_dtype": str(batched.dtype),
                        "requested_format": "nvfp4", "gptq_blocksize": None,
                        "scale_block_size": args.nvfp4_block_size,
                        "loss": None, "normalized_loss": None,
                        "hessian_nsamples": None, "reason": None,
                        "artifact": None,
                    }
                    if (pname, e) in forced_bf16:
                        rec["disposition"] = "BF16_FALLBACK"
                        rec["reason"] = "matched GPTQ mask (BF16 there)"
                        records.append(rec)
                        continue
                    q = NVFP4Quantizer(block_size=args.nvfp4_block_size,
                                       device=device)
                    qdq, art = _rtn_tensor(q, w_in_out.T.float())
                    w_in_out.copy_(qdq.T.to(batched.dtype))
                    verify_artifact_matches(art, w_in_out.T,
                                            what=f"{pname}[{e}]")
                    layer_arts[(pname, e)] = art
                    rec["disposition"] = "RTN_NVFP4"
                    rec["artifact"] = "pending"
                    records.append(rec)

        if layer_arts:
            shard = save_layer_artifacts(art_dir, li, layer_arts)
            for rec in records:
                if rec["artifact"] == "pending":
                    keys = artifact_keys(rec["name"], rec["expert_index"])
                    rec["artifact"] = {"file": shard, **keys}
        print(f"  layer {li}: {len(layer_arts)} tensors RTN-quantized")

    print(f"Saving QDQ model to {args.output} …")
    model.cpu().save_pretrained(str(args.output), max_shard_size="4GB")
    tok = AutoTokenizer.from_pretrained(str(args.source))
    tok.save_pretrained(str(args.output))
    import shutil
    for extra in ("chat_template.jinja", "generation_config.json"):
        src = args.source / extra
        if src.exists() and not (args.output / extra).exists():
            shutil.copy2(src, args.output / extra)

    covered = {r["param"] for r in records}
    excluded = [{"param": n, "reason": "not an eligible Linear/expert weight"}
                for n, _ in model.named_parameters() if n not in covered]
    manifest = write_quant_manifest(
        art_dir, records,
        {"quant_format": "nvfp4", "algorithm": "RTN",
         "nvfp4_block_size": args.nvfp4_block_size,
         "source": str(args.source),
         "matched_manifest": str(args.match_manifest) if args.match_manifest
         else None},
        excluded)
    print(f"Manifest: {manifest} ({len(records)} records)")

    if args.pack:
        spec = importlib.util.spec_from_file_location(
            "stage7", REPO / "tests" / "stage7_save_modelopt.py")
        stage7 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stage7)
        packed_dir = args.output.parent / (args.output.name + "-packed")
        report = stage7.pack_from_manifest(args.output, packed_dir,
                                           Path(manifest))
        print(f"Packed: {packed_dir} ({report['counts']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
