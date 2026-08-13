"""Generate reference logits so Day 1 has a correctness gate.

Your Rust forward pass is "working" long before it is correct. Fluent-looking
output survives a wrong RoPE convention, a missing q/k/v bias, or a swapped
gate/up projection. Comparing against real HF logits does not.

Run (pin 3.12 — torch wheels lag new Python releases):

    uv run --python 3.12 --with torch --with transformers --with numpy \
        python scripts/golden.py

Writes tests/golden/logits.npz with, for a fixed prompt:
    input_ids   [T]            the tokenization your Rust side must reproduce
    logits      [T, vocab]     full-sequence prefill logits, fp32
    hidden_0    [T, hidden]    embedding output, for bisecting a bad layer
    hidden_mid  [T, hidden]    after layer 13
    hidden_last [T, hidden]    after the final norm
"""

import pathlib

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
PROMPT = "The capital of France is"
OUT = pathlib.Path(__file__).resolve().parent.parent / "tests" / "golden"


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    # float32 on CPU: this is the reference, so precision matters more than speed.
    # No device_map= — it would drag in `accelerate`, and CPU is the default anyway.
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()

    ids = tok(PROMPT, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)

    # hidden_states is a tuple of length num_layers+1: [embeddings, ...layers].
    hs = out.hidden_states
    mid = len(hs) // 2

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "logits.npz"
    np.savez(
        path,
        input_ids=ids[0].numpy().astype(np.int64),
        logits=out.logits[0].float().numpy(),
        hidden_0=hs[0][0].float().numpy(),
        hidden_mid=hs[mid][0].float().numpy(),
        hidden_last=hs[-1][0].float().numpy(),
    )

    top = out.logits[0, -1].topk(5)
    print(f"wrote {path}")
    print(f"prompt: {PROMPT!r}")
    print(f"input_ids: {ids[0].tolist()}")
    print("top-5 next tokens:")
    for score, tid in zip(top.values.tolist(), top.indices.tolist()):
        print(f"  {tid:>7}  {score:8.4f}  {tok.decode([tid])!r}")


if __name__ == "__main__":
    main()
