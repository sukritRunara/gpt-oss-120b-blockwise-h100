#!/usr/bin/env python
"""Build a tiny GPT-OSS NVFP4 fixture checkpoint (P0.7/P0.8 proof).

Runs the REAL pipeline end to end on a small random GPT-OSS model:
  gptq_quantize_model (grouped Hessians, artifact capture, D-010 global
  scales) → save_pretrained + manifest → stage7 pack_from_manifest.

The output directory is a complete vLLM-loadable checkpoint (real gpt-oss
tokenizer files copied from the official download) used by the serve-side
fixture load test before any 20B export is trusted.

Run inside .venv-quant:
    python scripts/build_nvfp4_fixture.py --output /workspace/models/fixture-nvfp4
"""

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

import torch

REPO = Path("/workspace/blockwise-gptq-main")
sys.path.insert(0, str(REPO / "opteam-blockwise-gptq"))

OFFICIAL = Path("/workspace/models/gpt-oss-20b-official-mxfp4")

# Tiny but Marlin-friendly dims (all GEMM dims multiples of 128 where possible)
HIDDEN, INTERM, N_EXPERTS, TOP_K, N_LAYERS = 256, 256, 8, 2, 2
HEADS, KV_HEADS, HEAD_DIM = 4, 2, 64
SEQLEN, N_SAMPLES = 64, 4


def build_tiny_model():
    from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM

    cfg = GptOssConfig(
        hidden_size=HIDDEN,
        intermediate_size=INTERM,
        num_local_experts=N_EXPERTS,
        num_experts_per_tok=TOP_K,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        vocab_size=201088,           # real gpt-oss tokenizer vocab
    )
    torch.manual_seed(1234)
    model = GptOssForCausalLM(cfg).to(torch.bfloat16)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path,
                    default=Path("/workspace/models/fixture-nvfp4"))
    args = ap.parse_args()

    import apply as apply_mod

    work = args.output.parent / (args.output.name + "-build")
    qdq_dir = work / "qdq"
    art_dir = qdq_dir / "quant_artifacts"
    work.mkdir(parents=True, exist_ok=True)

    print("Building tiny GPT-OSS model (bf16)…")
    model = build_tiny_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    g = torch.Generator().manual_seed(7)
    calib = [(torch.randint(0, 201088, (1, SEQLEN), generator=g),) * 2
             for _ in range(N_SAMPLES)]
    apply_mod.get_calibration_data = lambda *a, **k: calib

    print("Quantizing (real pipeline, artifact capture, global scales)…")
    _, _, _, records = apply_mod.gptq_quantize_model(
        model, "fixture-gpt-oss",
        quant_format="nvfp4", dataset="synthetic",
        nsamples=N_SAMPLES, seqlen=SEQLEN,
        blocksize=128, percdamp=0.01, seed=0,
        device=device,
        mode="blockwise", parallel_hessian=True,
        mixed_precision_threshold=None,
        hessian_cache_dir=str(work / "hcache"),
        hessian_layer_group_size=N_LAYERS,
        artifact_dir=str(art_dir),
    )
    print(f"  {len(records)} tensor records")

    print("Saving QDQ model + manifest…")
    model.cpu().save_pretrained(str(qdq_dir))
    from quant_artifacts import write_quant_manifest
    covered = {r["param"] for r in records}
    excluded = [{"param": n, "reason": "not eligible"}
                for n, _ in model.named_parameters() if n not in covered]
    manifest = write_quant_manifest(
        art_dir, records,
        {"quant_format": "nvfp4", "nvfp4_block_size": 16}, excluded)

    print("Packing with stage 7…")
    spec = importlib.util.spec_from_file_location(
        "stage7", REPO / "tests" / "stage7_save_modelopt.py")
    stage7 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage7)
    report = stage7.pack_from_manifest(qdq_dir, args.output, Path(manifest),
                                       allow_hybrid=False)
    assert report["hybrid"] is False
    assert report["counts"]["experts_packed"] == N_LAYERS * N_EXPERTS * 2

    print("Copying real gpt-oss tokenizer files…")
    for fname in ("tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "chat_template.jinja",
                  "generation_config.json"):
        src = OFFICIAL / fname
        if src.exists():
            shutil.copy2(src, args.output / fname)

    print(f"\nFixture ready at {args.output}")
    print(f"Packing report: {report['counts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
