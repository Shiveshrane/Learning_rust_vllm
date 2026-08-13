# Day 1 — From weights to a token

**Goal:** a forward pass you wrote, producing logits that match HuggingFace within 1e-2.

**Concepts:** decoder-only data flow · RMSNorm · RoPE · GQA · SwiGLU · why naive generation is quadratic

---

## 1. The shape table

Write this on paper and keep it next to you. Almost every Day 1 bug is a shape
bug wearing a disguise, and the compiler cannot help you — candle shapes are
checked at runtime.

For a prompt of `T` tokens, batch 1:

| Step | Shape | Notes |
|---|---|---|
| `input_ids` | `[T]` | u32 |
| `embed_tokens` | `[1, T, 1536]` | row lookup into `[151936, 1536]` |
| `input_layernorm` | `[1, T, 1536]` | RMSNorm, no shape change |
| `q_proj` | `[1, T, 1536]` | weight `[1536, 1536]`, **+ bias** |
| `k_proj` / `v_proj` | `[1, T, 256]` | weight `[256, 1536]`, **+ bias** — 2 heads × 128 |
| reshape q | `[1, T, 12, 128]` | then transpose → `[1, 12, T, 128]` |
| reshape k/v | `[1, T, 2, 128]` | then transpose → `[1, 2, T, 128]` |
| RoPE(q), RoPE(k) | unchanged | v is **not** rotated |
| repeat_kv(k, v) | `[1, 12, T, 128]` | each KV head serves 6 Q heads |
| `q @ k^T` | `[1, 12, T, T]` | × `1/sqrt(128)` |
| + causal mask | `[1, 12, T, T]` | |
| softmax(dim=-1) | `[1, 12, T, T]` | |
| `@ v` | `[1, 12, T, 128]` | |
| transpose + reshape | `[1, T, 1536]` | back to model dim |
| `o_proj` | `[1, T, 1536]` | **no bias** |
| + residual | `[1, T, 1536]` | |
| `post_attention_layernorm` | `[1, T, 1536]` | |
| `gate_proj`, `up_proj` | `[1, T, 8960]` | no bias |
| `silu(gate) * up` | `[1, T, 8960]` | |
| `down_proj` | `[1, T, 1536]` | no bias |
| + residual | `[1, T, 1536]` | |
| × 28 layers | | |
| `model.norm` | `[1, T, 1536]` | final RMSNorm |
| `lm_head` | `[1, T, 151936]` | untied — its own `[151936, 1536]` |

The transpose-then-reshape pair at the end is where people lose an hour. You
cannot `reshape` straight from `[1, 12, T, 128]` to `[1, T, 1536]` — that
interleaves head data into the wrong positions. Transpose to `[1, T, 12, 128]`
first, and candle will make you call `.contiguous()` before the reshape because
the transposed tensor isn't laid out linearly. When it errors, that's the
memory layout telling you something true, not an annoyance to work around.

## 2. The math you're implementing

### RMSNorm

$$y = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \odot w$$

Mean over the **last** dim, `eps = 1e-6`, elementwise multiply by the learned
weight. No mean-subtraction and no bias — that's what separates it from
LayerNorm, and it's most of why it's faster.

Do the reduction in **f32 even when the tensor is bf16**. bf16 has 8 bits of
mantissa; summing 1536 squared values in it loses real precision. HF does the
same upcast, and if you don't, your error against the golden file will be
suspiciously large from layer 1.

### RoPE

Rotary embeddings encode position by *rotating* pairs of dimensions, at a rate
that differs per pair.

$$\theta_i = \frac{1}{10000^{2i/d}}, \quad i \in [0, d/2), \quad d = 128$$

Low `i` → high frequency → rotates fast → encodes fine-grained local position.
High `i` → wavelength longer than the whole context → nearly static → encodes
coarse position. **Hold onto this**; it is the entire basis of YaRN on Day 5.

Build a `[T, 64]` table of `pos × θ_i`, take `cos` and `sin`, then rotate. The
rotation itself has two conventions, and Qwen2 uses the **half-split** one:

```
x1 = x[..., :64]      x2 = x[..., 64:]
out = concat(x1·cos − x2·sin,  x2·cos + x1·sin)
```

The other convention pairs *adjacent* elements `(x0,x1), (x2,x3), …`. Both are
valid rotary embeddings; the weights were trained with exactly one of them.

- `candle_nn::rotary_emb::rope` — half-split. **This is yours.**
- `candle_nn::rotary_emb::rope_i` — interleaved. Not yours.
- `rope_slow` (`candle-nn-0.11.0/src/rotary_emb.rs:590`) — readable reference. Read it.

Wrong convention produces text that is grammatical and completely wrong, with
no error anywhere. This is the highest-value thing the golden file catches.

### GQA — and the head-mapping trap

12 query heads, 2 KV heads. Six Q heads share each KV head. **Which six matters:**

```
KV head 0  serves  Q heads 0,1,2,3,4,5
KV head 1  serves  Q heads 6,7,8,9,10,11
```

That is `q_head / 6` (integer division), **not** `q_head % 2`. Repeat-interleave
semantics, not tile semantics. Both produce a `[1, 12, T, 128]` tensor of the
right shape, so shape checks pass and output is garbage.

Getting there in candle: `unsqueeze` a dim after the head axis, `expand` it to
the group size, then `reshape` to fold it in. Think about which axis you're
expanding and convince yourself it gives repeat-interleave, don't just make the
shapes line up.

### SwiGLU

$$\text{silu}(x) = x \cdot \sigma(x), \qquad \text{MLP}(x) = W_{down}\big(\text{silu}(W_{gate}x) \odot W_{up}x\big)$$

Two parallel up-projections to 8960, one gated by the other, then back down.
`gate` and `up` are separate weights — swapping them is silent, since the shapes
are identical. Check the tensor names twice.

### Causal mask

`mask[i,j] = 0 if j <= i else −inf`, added to the scores before softmax. Build it
once per prefill, not per layer. Adding `−inf` then softmaxing gives exactly 0
attention weight; using a merely large negative like `−1e4` leaks a little
probability, which shows up as a small persistent error in your golden diff.

## 3. What to build

**Block 1 (20 min) — `device.rs`.** `pick(force_cpu: bool) -> Result<Device>`.
You want the CPU escape hatch because when logits disagree, "my math or the
Metal kernel?" is the first question, and a flag answers it in seconds.

**Block 2 (70 min) — `config.rs` + weight loading.**

- `Qwen2Config` via `#[derive(Deserialize)]`, one field per `config.json` key.
  Add derived methods (`head_dim()`, `kv_groups()`, `kv_bytes_per_token()`)
  instead of recomputing `hidden_size / num_attention_heads` in eight places.
  Leave room for `rope_scaling: Option<...>` — Day 5 fills it.
- `VarBuilder::from_mmaped_safetensors` (`candle-nn-0.11.0/src/var_builder.rs:642`).
  It's `unsafe` because mmap gives you a `&[u8]` view of a file another process
  can truncate underneath you. You're asserting nobody edits the checkpoint mid-run.
- `vb.pp("model.layers.0.self_attn")` for prefix scoping.
- **First test, two lines, do it before anything else:** tokenize
  `"The capital of France is"` and assert you get `[785, 6722, 315, 9625, 374]`.
  Five tokens, no BOS. If this fails, every later comparison is meaningless.

**Block 3 (2h) — `model.rs`.** RMSNorm, RoPE tables, attention, MLP, block, model.
Build bottom-up and unit-test each piece against a hand-computed value before
assembling. A `[1,1,4]` tensor you can normalize on paper is worth more than a
debugger here.

**Block 4 (1h) — greedy loop + the gate.** Re-run the whole prefix every step.
Deliberately quadratic; it's your Day 2 baseline and the reason KV cache exists.

## 4. Gate

- [ ] Tokenizer gives `[785, 6722, 315, 9625, 374]`
- [ ] `argmax(logits[-1]) == 12095` (` Paris`, logit 11.3769)
- [ ] Full logits match `tests/golden/logits.npz` within 1e-2
- [ ] 20 tokens of coherent text
- [ ] tok/s recorded (expect single digits — that's correct for now)

Read the golden file with `Tensor::read_npz_by_name`
(`candle-core-0.11.0/src/npy.rs:304`) — candle tensors directly, no numpy plumbing.

**Bisecting a failure.** Don't stare at logits. Compare in order:
`hidden_0` (embedding — catches wrong token IDs or a bad lookup) → `hidden_mid`
(after layer 13 — catches RoPE, attention, MLP) → `hidden_last` (after final
norm). The golden is fp32 and you're in bf16, so error grows with depth
naturally; judge relative error. A genuinely broken layer jumps by orders of
magnitude, not percent.

## 5. The gotcha list

1. `q/k/v_proj` have **bias**; `o_proj` and all MLP projections do **not**
2. `rope` (half-split), never `rope_i`
3. GQA mapping is `q_head / 6`, not `q_head % 2`
4. RMSNorm reduction in f32, not bf16
5. `lm_head` is untied — its own weight, not `embed_tokens` transposed
6. `.contiguous()` before reshaping a transposed tensor
7. candle `Linear` holds `[out, in]` and computes `x @ w.T` — matches safetensors, no transpose on load
8. v is not rotated; only q and k

## 6. After the gate passes — not before

Open `candle-transformers-0.11.0/src/models/qwen2.rs` (402 lines) and diff it
against yours. Where you differ, work out which is right. Reading it earlier
costs you the entire day's learning, and it is the only irreversible mistake
available to you today.
