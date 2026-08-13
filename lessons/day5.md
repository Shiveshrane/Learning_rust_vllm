# Day 5 — YaRN, long context, and shipping it

**Goal:** context beyond what the model was trained on, and a server the official
OpenAI client can talk to.

**Concepts:** RoPE as frequencies · wavelength vs dimension · interpolation vs
extrapolation · attention entropy

---

## 1. The problem

Day 1's RoPE: `θ_i = 1 / 10000^(2i/d)`, and position `p` rotates dimension pair
`i` by `p · θ_i`.

The model saw positions `0..L` in training. Feed it position `4 · L` and every
rotation angle is four times larger than anything it has ever seen. Attention
scores go haywire — not gracefully, but off a cliff. Output degenerates into
repetition or noise.

You want to extend context **without retraining**. Every method below is a
different answer to: *what do we do with the angles?*

## 2. Wavelength — the idea everything hangs on

Rewrite the rotation as a wave. Dimension pair `i` has wavelength

$$\lambda_i = \frac{2\pi}{\theta_i} = 2\pi \cdot 10000^{2i/d}$$

With `d = 128`, `θ = 10000`:

| pair `i` | `λ_i` (tokens) | what it encodes |
|---|---|---|
| 0 | ~6 | fine local ordering — adjacent words |
| 16 | ~63 | phrase scale |
| 32 | ~630 | paragraph scale |
| 48 | ~6,300 | longer than most training contexts |
| 63 | ~57,000 | never completes a cycle in training |

**Two different situations in one tensor.** Low-`i` pairs complete thousands of
full rotations during training — the model has seen every angle they can
produce, at every relative offset it cares about. High-`i` pairs never complete
even one cycle, so the model has only ever seen a narrow arc of their range.

That asymmetry is why a single global fix can't be right.

## 3. Four methods, in order

Implement them in this order. Each one's failure is the next one's motivation,
and YaRN reads as arbitrary if you skip to it.

### Position Interpolation (PI)

Divide positions by the scale factor `s`: `p → p/s`. Position 8192 at `s=2` is
treated as 4096 — in range, no unseen angles.

**Works, and degrades local resolution.** Adjacent tokens now differ by `1/s` of
a rotation instead of a full one. You've squeezed the fine-grained pairs
— the ones the model relies on for local word order — into half their
resolution. Perplexity gets measurably worse even *inside* the original context.

### NTK-aware

Scale the base instead: `θ = 10000 → 10000 · s^(d/(d−2))`.

This spreads the change unevenly: high-frequency pairs barely move, low-frequency
pairs stretch a lot. Local resolution is preserved. **But** the very highest
pairs now get pushed slightly out of their trained range — the fix is applied by
a smooth formula rather than by asking which pairs actually need it.

### NTK-by-parts

Stop being smooth. **Ask each dimension what it needs.**

- λ_i ≪ trained context → the model has seen all these angles → **extrapolate** (leave it alone)
- λ_i ≫ trained context → never completed a cycle → **interpolate** (scale by `1/s`)
- in between → **ramp** linearly between the two

Concretely: for interpolated pairs use `θ_i/s`, for extrapolated pairs use `θ_i`,
and blend with a per-dimension ramp mask `γ_i ∈ [0,1]`:

$$\theta_i^{new} = (1 - \gamma_i)\cdot\frac{\theta_i}{s} + \gamma_i \cdot \theta_i$$

The boundaries come from `β_fast` and `β_slow` (typically 32 and 1), expressed as
"how many full rotations does this pair complete in the trained context":
`r_i = L / λ_i`. Pairs with `r_i > β_fast` are safe to extrapolate; `r_i < β_slow`
must be interpolated; between them, ramp.

### YaRN = NTK-by-parts + attention temperature

One more effect. Longer context means more tokens in the softmax, which spreads
attention thinner — entropy rises and the distribution flattens, independent of
any RoPE issue.

YaRN corrects it by scaling attention:

$$\text{mscale} = 0.1 \cdot \ln(s) + 1$$

applied to the softmax scale (so `scale = mscale / sqrt(head_dim)`). An empirical
constant, not derived — worth knowing, because papers rarely flag which parts
are theory and which are fitted.

Cheapest implementation: fold `mscale` into your precomputed `cos`/`sin` tables.
Then attention code doesn't change at all — it's still `1/sqrt(128)` — and the
temperature rides along in the rotation. Convince yourself that's equivalent
before you do it.

## 4. Building it

Drive it from `rope_scaling` in the config (your checkpoint has none, so `None`
is the identity path and Day 1's behaviour must be bit-identical when it's absent
— assert that).

The pieces, all operating on the inverse-frequency vector before you build
cos/sin tables:

- `find_correction_dim(rotations, dim, base, max_pos)` — invert `r_i = L/λ_i` to get the dimension index where a given rotation count occurs
- `find_correction_range(β_fast, β_slow, ...)` → `(low, high)`, floor and ceil respectively
- `linear_ramp_mask(low, high, dim)` → `γ` clamped to `[0,1]`; guard `high == low` or you divide by zero
- `get_mscale(scale, mscale)` → `1.0` when `s ≤ 1`, else `0.1·mscale·ln(s) + 1`

Reference implementation to check yourself against **after** you've written
yours: `candle-transformers-0.11.0/src/models/deepseek2.rs:362-440`.

## 5. Proving it works

Code that compiles is not a feature that works, and long-context bugs are
invisible in short prompts.

**Passkey retrieval.** Bury a secret in filler:

```
[~N tokens of repetitive filler]
The passkey is 48291. Remember it.
[~N tokens more filler]
What is the passkey?
```

Sweep total length 4k / 8k / 16k / 32k, and vary where the passkey sits (10%,
50%, 90% through). Run each config 5 times with different passkeys and report
accuracy.

Compare: **no scaling** vs **PI** vs **YaRN**. Expect unscaled to fall off a
cliff past its trained window while YaRN degrades gracefully. Also check
perplexity *inside* the original context under each method — this is where PI
should show the local-resolution damage from §3, and where you'll see YaRN
mostly avoid it.

If YaRN doesn't beat PI, you have a bug. Most likely candidates: ramp inverted
(interpolating the wrong end), or `mscale` applied twice.

Watch the KV cost — 32k × 28 KB is 940 MB for one sequence. Days 3 and 4 are
what make this experiment runnable at all; note in your writeup how many
concurrent 32k sequences int4 KV buys you versus bf16.

**Cross-check:** `Qwen2.5-1.5B-Instruct` ships a real YaRN `rope_scaling` block
in its config. Parsing it and matching upstream's numbers validates your config
handling independently of your math.

## 6. OpenAI compatibility

`POST /v1/chat/completions` and `GET /v1/models`.

The chat template lives in `tokenizer_config.json` (your snapshot has no
`chat_template.jinja`). It's a Jinja string — render it with `minijinja`, or
hand-roll it since ChatML is a fixed, simple format. Rendering the real template
is more honest: it handles the `add_generation_prompt` flag and this model's
`<think>` tag convention, and getting the special tokens wrong degrades output
in ways that look like a model problem.

Get `usage` right (`prompt_tokens`, `completion_tokens`, `total_tokens`) and
return OpenAI-shaped error bodies.

**The test:** point the official `openai` Python client at
`http://localhost:8080/v1` with a dummy key, streaming and non-streaming. If it
works unmodified, you're compatible. If you hand-rolled your own curl tests only,
you're not — the client is stricter than you are about SSE framing, the
`data: [DONE]` sentinel, and required response fields.

## 7. Write it up

README with: architecture diagram, the Day 4 benchmark matrix, the passkey
results, and an explicit **known gaps** section —

- gather-based paged attention, not a real paged-attention kernel (Day 3)
- dequantize-on-read, not fused dequant (Day 4)
- recompute-only preemption, no swap (Day 3)

Being precise about what you didn't build is what makes the rest credible.

## 8. Gate

- [ ] `rope_scaling: None` reproduces Day 1 logits exactly
- [ ] PI, NTK-aware, and YaRN all implemented and switchable
- [ ] Passkey retrieval at 32k works with YaRN, fails without — with numbers
- [ ] Perplexity inside the original context measured for each method
- [ ] Official `openai` Python client works unmodified, streaming and not
- [ ] README with benchmarks and known gaps

## 9. Where to go next

In rough order of how much you'd learn:

1. **A real MSL paged-attention kernel.** Removes the Day 3 gather and the Day 4 dequant copy in one move — you'd be implementing what vLLM actually does. Needs `sudo xcode-select -s /Applications/Xcode.app`.
2. **Speculative decoding.** A draft model proposes `k` tokens, the target verifies them in one forward pass. The rejection-sampling correctness proof — that the output distribution is *provably unchanged* — is the interesting part, not the speedup.
3. **Weight quantization / GGUF.** `candle-core`'s `quantized` module: `k_quants.rs`, `gguf_file.rs`. The 3.5 GB you didn't touch this week.
4. **Structured output.** Constrain sampling to a grammar or JSON schema by masking logits. Mechanically simple, surprisingly deep once you hit tokenizer-boundary problems.
