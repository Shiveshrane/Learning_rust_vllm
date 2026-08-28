# A Rust inference server for Qwen2 on Apple Silicon

A from-scratch LLM inference engine — paged KV cache, continuous batching, KV
quantization, YaRN — built on `candle-core`'s Metal backend, targeting
`DeepSeek-R1-Distill-Qwen-1.5B` (Qwen2 architecture, 1.5B).

Built as a 5-day learning project.

## Lessons

- [x] **Day 0** — workspace, Metal smoke test, checkpoint verified
- [x] [**Day 1**](lessons/day1.md) — forward pass, greedy decode, golden-logit gate
- [x] [**Day 2**](lessons/day2.md) — KV cache, sampling, SSE streaming server
- [x] [**Day 3**](lessons/day3.md) — paged KV cache, continuous batching
- [ ] [**Day 4**](lessons/day4.md) — KV quantization, prefix caching, benchmarks
- [ ] [**Day 5**](lessons/day5.md) — YaRN, long context, OpenAI compatibility

## Layout

| Crate | Contents |
|---|---|
| `crates/qwen` | Config, weight loading, RMSNorm, RoPE, GQA attention, SwiGLU, forward |
| `crates/engine` | KV cache → block allocator → scheduler; sampling; quantized KV; YaRN |
| `crates/server` | axum, OpenAI-compatible routes, SSE |
| `crates/bench` | Concurrency load generator, TTFT/ITL/throughput |

## Commands

```sh
cargo run --release --bin smoke      # Day 0 gate: Metal + checkpoint
cargo clippy --workspace --all-targets -- -D warnings

# Reference logits for the Day 1 correctness gate (pin 3.12; torch wheels lag)
uv run --python 3.12 --with torch --with transformers --with numpy \
    python scripts/golden.py
```

## Target architecture

```
Qwen2ForCausalLM   28 layers   hidden 1536   intermediate 8960 (SwiGLU)
12 Q heads / 2 KV heads (GQA 6:1)   head_dim 128   vocab 151936
rms_norm_eps 1e-6   rope_theta 10000   tie_word_embeddings false
q/k/v_proj have bias; o_proj and all MLP projections do not
```

KV per token = 2 × 28 × 2 × 128 × 2 bytes = **28,672 bytes**. A 32k context is
0.94 GB of KV for a single sequence — which is why paging and KV quantization
exist, and the thread running through Days 3–4.

## Day 0 measurements

M5 Pro, 24 GB unified memory, 16-core GPU, macOS 26.5.2, rustc 1.96.1.

| 2048² matmul | TFLOP/s |
|---|---|
| f32 | 5.40 |
| bf16 | 5.96 |
| f16 | 5.91 |

bf16 being only ~10% ahead of f32 says this size is launch/bandwidth bound
rather than compute bound in candle's Metal GEMM — worth re-measuring at 4096²
and at real prefill shapes before drawing conclusions about dtype choice.

Checkpoint: 339 tensors (28 × 12 + embed + norm + lm_head), 3.55 GB bf16, q/k/v
biases present and o_proj bias absent as expected.

## Day 2 measurements

F32 weights on Metal, single stream, measured over HTTP/SSE against a warm
server. Median of three runs.

| | value |
|---|---|
| TTFT (5-token prompt) | **57.7 ms** |
| Prefill (480-token prompt) | **1083 tok/s** |
| Decode (short context) | **40.8 tok/s** |
| Decode (480-token context) | 35.3 tok/s |

Measure warm. The first request after startup shows a TTFT of ~858 ms — Metal
kernel compilation and paging in the mmap'd weights, not prefill.

### What the KV cache bought

Same prompt, 20 tokens, byte-identical output at every stage:

| | 20 tok | tok/s | vs Day 1 |
|---|---|---|---|
| Day 1, recompute the whole prefix each step | 7.34 s | 2.7 | 1× |
| KV cache, hand-rolled attention | 1.20 s | 16.6 | 6.1× |
| + `sdpa`, `repeat_kv` deleted | 0.54 s | 37.3 | **13.7×** |

Day 1 was `O(N²)`; extrapolating it to 512 tokens gives roughly 80 minutes
against 13 seconds measured.

### Prefill and decode are different machines

Taking 1.78e9 parameters and `2·P·N` FLOPs, against the 5.96 TFLOP/s ceiling
measured on Day 0:

| | achieved | % of peak |
|---|---|---|
| Prefill | 3.86 TFLOP/s | 65% |
| Decode | 0.145 TFLOP/s | 2.4% |

A 27× gap on identical weights. Prefill does `T` tokens' work per pass over the
weights and is compute-bound; decode does one token's work per pass over the
same 3.5 GB and is bandwidth-bound at ~2 FLOP/byte. That ratio is the argument
for batching — 16 sequences read those weights once — and it is what Day 3 cashes
in.

Decode falls from 40.8 to 35.3 tok/s as context grows to 480 tokens: attention
is `O(context)` per step, and the cache readback copies the whole span with
`.contiguous()` on every layer.

Sampling costs ~10 ms/token when enabled (37.3 → 27.4 tok/s), almost entirely the
608 KB device→host `to_vec1` of the logits row. The fix is on-device top-k so
only survivors cross the bus; not yet done.

## Day 3 measurements

Paged KV cache (16-token blocks) plus continuous batching. Same machine, F32
weights on Metal, 40 tokens per request over HTTP/SSE.

### Throughput against concurrency

| concurrency | tok/s | vs 1x | efficiency | TTFT p50 |
|---|---|---|---|---|
| 1 | 37.6 | 1.00x | 100% | 51 ms |
| 2 | 43.4 | 1.15x | 58% | 96 ms |
| 4 | 73.9 | 1.96x | 49% | 188 ms |
| 8 | 114.4 | 3.04x | 38% | 370 ms |
| 16 | 150.4 | 4.00x | 25% | 729 ms |
| 32 | 181.1 | 4.81x | 15% | 1429 ms |

Batching is worth **4.8x** at concurrency 32, and the curve bends immediately —
efficiency is already 58% at two concurrent requests. Three costs scale with
batch size instead of amortising:

1. **The KV gather.** Every decode step copies each sequence's blocks into a
   contiguous tensor, pads to the longest, and concatenates. That is `O(B x L)`
   per layer per step. vLLM avoids it entirely by walking the block table inside
   the attention kernel; candle has no Metal equivalent (see gaps below).
2. **`repeat_kv` in the batched path.** GQA expansion is materialised 6x because
   the batched decode cannot use `sdpa` (see gaps).
3. **Sampling.** `Sampler::sample` copies a 151,936-float logits row to the host
   per sequence per step — 608 KB each, so 19 MB per step at concurrency 32,
   plus a CPU pass per sequence. On-device top-k would move only the survivors.

Weight reads *do* amortise, which is where the 4.8x comes from: one pass over
3.5 GB now serves the whole batch instead of one sequence.

TTFT grows roughly linearly with concurrency because prefill and decode
alternate: a step is either one or the other, so an arriving request stalls
every running sequence for one long step. Chunked prefill is the fix and is
deliberately not implemented (see below).

### Against Day 2

| | Day 2 | Day 3 |
|---|---|---|
| KV per sequence | 4096 tokens preallocated | 16-token blocks on demand |
| Waste on a 50-token generation | 99% | <0.4% |
| Concurrent sequences in budget | ~63 | thousands |
| Aggregate tok/s | 40.8 (single stream) | 181.1 |

### Known gaps

**Gather, not a kernel.** `KVPool::gather` uses `index_select` to make scattered
blocks contiguous before `sdpa`. The memory-management win is intact — no
fragmentation, no over-reservation, and prefix sharing becomes possible — but
part of the bandwidth win is given up, because KV is copied every step rather
than read in place. A hand-written MSL paged-attention kernel is the fix.

**`sdpa` ignores masks during decode.** On Metal, `q_seq == 1` selects the
vectorised kernel and `call_sdpa_vector` takes no mask argument. Batched decode
pads ragged sequence lengths and *must* mask the padding — a zero key is not
neutral, `exp(q.0) = 1` gives it real attention weight — so the batched path
uses hand-rolled attention instead.

**KV pool capped at 2 GB.** Above roughly 4 GB, candle's Metal backend hands out
buffers that alias ones still in use: logits stay bit-exact while `argmax`
returns the contents of the gather index tensor. Not a paging bug — verified
`max|diff| = 0` against the contiguous cache at every pool size.

**Alternate prefill/decode, not chunked.** Chosen for simplicity; the TTFT cost
is visible in the table above.

### Correctness

Paged KV is bit-identical to the contiguous cache: `max|diff| = 0.000000` across
prefill and 20 decode steps, with deliberately non-adjacent block tables.
Deliberate over-subscription forces preemption (95 scheduler steps versus 61 with
a roomy pool) and the output is byte-identical, so recompute is invisible to the
client. The block invariant — `free + held == total`, two independently
maintained numbers — is asserted every scheduler iteration.

## Day 4 measurements — int8 KV

Symmetric int8, stored as `u8` with a `+128` offset because candle has no `I8`.
Per-channel scales for K, per-token for V, following KIVI. Runtime-selectable:

```sh
KV_DTYPE=int8 cargo run --release -p server
```

### The scorecard

| | F32 | int8 | delta |
|---|---|---|---|
| Tokens resident in a 2 GB pool | 34,864 | **135,280** | **3.88x** |
| Perplexity (512-token held-out passage) | 50.9099 | 51.3057 | **+0.78%** |
| Aggregate tok/s at concurrency 32 | 187.9 | 135.2 | **-28%** |

Bytes per token: `57,344` (F32) against `14,784` (int8 = 14,336 codes + 448
scales; scales are 3% overhead).

### Grouping is the whole question

Error is roughly `scale/2` per element, and the scale is set by the largest
magnitude in the group — so one outlier makes every other value in that group
noise. Measured on a synthetic group, values sharing a scale with a 400x outlier
came back **73x** worse than the same values in a clean group.

Key and value tensors need different grouping. Keys carry persistent
per-channel outliers, and K errors are amplified because they pass through
`q.k^T` and then softmax; V errors pass through a linear weighted sum. Measured
on real keys at layer granularity, K's absolute reconstruction error is **32x**
V's, at identical relative error — K values are simply larger.

Switching K from per-token to per-channel, on the real checkpoint:

| K grouping | logit max\|diff\| | relative to peak logit | next-token argmax |
|---|---|---|---|
| per-token (naive) | 2.3487 | 16.19% | `" "` — wrong |
| per-channel (KIVI) | 0.8268 | 5.70% | `" Paris"` — correct |

Perplexity says the remaining 5.70% logit error costs **0.78%** of predictive
quality. Greedy decoding still diverges from the F32 trajectory after ~10
tokens, because argmax amplifies small logit differences at genuinely uncertain
choices — the text differs, the model is not meaningfully worse.

### The throughput result, which contradicts the premise

The usual argument is that halving KV bytes buys throughput directly, because
decode is memory-bandwidth-bound. Measured here it does the opposite, and the
gap widens with concurrency:

| concurrency | F32 tok/s | int8 tok/s |
|---|---|---|
| 1 | 37.9 | 35.2 |
| 4 | 76.8 | 66.4 |
| 8 | 120.5 | 96.8 |
| 16 | 153.4 | 115.8 |
| 32 | 187.9 | 135.2 |

Bytes saved on the pool read are paid for twice in compute:

1. **`gather` dequantizes on every read** — `to_dtype`, `affine`,
   `broadcast_mul`, plus a second `index_select` for K's block-indexed scales.
   Three extra full-size tensor ops per layer per step.
2. **`write` requantizes a whole 16-slot block per token.** Per-channel K scales
   span a block, so a decode write dequantizes 16 slots, overwrites one row,
   recomputes the scale, and writes all 16 back — ~16x the quantize work, on
   every token, on every one of 28 layers.

Cost (2) scales with batch size, which is why the penalty grows from 7% to 28%.
The premise holds only when dequantization is **fused into the attention
kernel**; with dequant-on-gather it is the same trade as Day 3's `index_select`
— the capacity win is kept in full and the bandwidth win is lost rather than
merely forgone.

So int8 here buys **4x the concurrent sequences for 0.8% quality and a 28%
throughput penalty**. Worth it if capacity-bound, not if throughput-bound.

### Partial blocks

A per-channel K scale spans all 16 tokens of a block, so it is not final until
the block fills — and decode writes one token at a time. Rather than staging
partial blocks in F32 (which would need per-sequence state that `KVPool`, being
shared, does not have), a write requantizes the whole block: dequantize its 16
slots, overwrite the incoming rows, recompute the scale, write back. Unwritten
slots are zeros and cannot raise a max, so the scale is correct without tracking
fill level, and error does not compound — once a value sits on the quantization
grid, requantizing at an unchanged scale returns it exactly.
