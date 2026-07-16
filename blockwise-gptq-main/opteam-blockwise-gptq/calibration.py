"""
Calibration data loading and layer input capture for GPTQ.

Provides dataset loading (WikiText-2, C4) and a hook-based mechanism
to capture hidden states entering the first transformer layer.
"""

import torch
import random


def get_calibration_data(model_name, dataset="wikitext2", nsamples=128, seqlen=2048, seed=0):
    """Load and tokenize calibration data.

    Args:
        model_name: HuggingFace model name (for tokenizer).
        dataset: "wikitext2" or "c4".
        nsamples: Number of calibration samples.
        seqlen: Sequence length per sample.
        seed: Random seed for reproducibility.

    Returns:
        List of (input_ids [1, seqlen], targets [1, seqlen]) tuples.
    """
    from transformers import AutoTokenizer
    from datasets import load_dataset

    random.seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if dataset == "wikitext2":
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n\n".join(data["text"])
    elif dataset == "c4":
        data = load_dataset(
            "allenai/c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )
        # Sample random documents and concatenate
        indices = random.sample(range(len(data)), min(nsamples * 5, len(data)))
        text = "\n\n".join(data[i]["text"] for i in indices)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Supported: wikitext2, c4")

    # Tokenize the full text
    enc = tokenizer(text, return_tensors="pt")
    all_ids = enc.input_ids[0]  # [total_tokens]

    # Slice into samples of seqlen
    samples = []
    total_tokens = all_ids.shape[0]
    if total_tokens < seqlen + 1:
        raise ValueError(f"Calibration text too short: {total_tokens} tokens < {seqlen + 1}")

    # Random starting positions
    starts = random.sample(range(total_tokens - seqlen - 1), min(nsamples, total_tokens - seqlen - 1))
    for start in starts[:nsamples]:
        input_ids = all_ids[start:start + seqlen].unsqueeze(0)
        targets = all_ids[start + 1:start + seqlen + 1].unsqueeze(0)
        samples.append((input_ids, targets))

    return samples


class LayerInputCatcher(torch.nn.Module):
    """Wraps the first transformer layer to capture calibration inputs.

    After running all calibration samples through the model's embeddings,
    this module stores the hidden states (inputs to layers[0]) in a buffer.
    It raises ValueError to abort the forward pass early (we only need
    the first-layer inputs, not full model output).

    Usage:
        catcher = LayerInputCatcher(layers[0], nsamples, hidden_size, seqlen, dtype, device)
        layers[0] = catcher
        for batch in calibration_data:
            try:
                model(batch)
            except ValueError:
                pass
        layers[0] = catcher.module  # restore original
        inps = catcher.inps  # [nsamples, seqlen, hidden_size]
    """

    def __init__(self, module, nsamples, hidden_size, seqlen, dtype, device):
        super().__init__()
        self.module = module
        self.nsamples = nsamples
        self.inps = torch.zeros(
            (nsamples, seqlen, hidden_size), dtype=dtype, device=device
        )
        self.cache = {"i": 0}
        self.kwargs = {}
    
    def __getattr__(self, name):
        # Proxy attribute access to the wrapped layer so the model's forward
        # loop can access layer attributes like attention_type, layer_idx, etc.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
    
    def forward(self, inp, *args, **kwargs):
        i = self.cache["i"]
        # inp may be [1, seqlen, hidden] or [seqlen, hidden]
        if inp.dim() == 3:
            self.inps[i] = inp[0].to(self.inps.device)
        else:
            self.inps[i] = inp.to(self.inps.device)

        self.cache["i"] += 1

        # Capture kwargs on first call (attention_mask, position_embeddings, etc.)
        if not self.kwargs:
            self.kwargs = {k: v for k, v in kwargs.items()}

        raise ValueError("LayerInputCatcher: early exit after capture")
