# Day 2 — KV cache, sampling, streaming

**Goal:** a server that streams tokens over HTTP, 10–30× faster than Day 1.

**Concepts:** prefill vs decode as different machines · memory-bandwidth-bound
decoding · arithmetic intensity · incremental detokenization

---

## 1. Why Day 1 was slow

Your Day 1 loop recomputed the entire prefix every step. Generating token `n`
re-ran attention over tokens `0..n`, so producing `N` tokens cost `O(N²)` work.

But look at what actually changes between steps. For a token at position `j`,
its key and value vectors depend only on that token's hidden state and its
position — **neither depends on anything that comes after it.** Causal masking
guarantees it. So `k_j` and `v_j` computed at step 5 are bit-identical to `k_j`
recomputed at step 500.

That's the whole idea. Cache them. Generating token `n` becomes: compute `q, k,
v` for **one** token, append `k,v` to the cache, attend over all cached keys.
`O(N)` total instead of `O(N²)`.

### The two regimes

This split is the most important intuition of the week — Days 3 and 4 are both
consequences of it.

|  | Prefill | Decode |
|---|---|---|
| Tokens processed | `T` at once | 1 |
| `q @ k^T` shape | `[12, T, T]` | `[12, 1, T]` |
| Weight bytes read | 3.5 GB | 3.5 GB |
| FLOPs done | ~2·3.5e9·T | ~2·3.5e9·1 |
| Arithmetic intensity | ~`T` FLOP/byte | **~2 FLOP/byte** |
| Bound by | compute | **memory bandwidth** |

Decode reads all 3.5 GB of weights to produce a *single* token. The GPU's ALUs
sit idle waiting on memory. Your M5 Pro's bandwidth, not its TFLOP/s, sets your
tok/s ceiling — and you measured 5.9 TFLOP/s on Day 0 that you will never touch
during decode.

Two consequences you'll cash in later: batching is nearly free (16 sequences read
the same weights once — that's Day 3), and shrinking the KV cache directly buys
throughput rather than just capacity (Day 4).

## 2. Building the cache

**Preallocate. Do not `Tensor::cat`.** The obvious implementation appends with
`cat` each step. `cat` allocates a new tensor and copies the whole cache — so
you've rebuilt an `O(N²)` algorithm out of memcpy instead of matmul, and it will
quietly dominate your runtime while looking correct.

Allocate `[1, kv_heads, max_seq, head_dim]` per layer up front and write the new
token into slot `pos` with `slice_set`. Track `current_len` yourself.

Per layer that's `2 × 2 × max_seq × 128 × 2` bytes. For `max_seq = 4096`: 4 MB
per layer, 114 MB across 28 layers. Note how much you're reserving for one
sequence regardless of how long it actually gets — that waste is exactly what
Day 3 kills.

**The position bug.** Your RoPE now receives one token, but it is not at
position 0. Thread a `pos_offset` through so decode step `n` rotates by `n`.
This is bug #3 on the classic list and it's insidious: output stays fluent and
slowly loses the plot, because the model thinks every new token is the first one.

Verify by asserting cached-decode logits equal Day 1's recompute logits for the
same prefix. Same numbers, different cost — that's the whole claim.

Read `candle-nn-0.11.0/src/kv_cache.rs` (lines 6–150) **after** yours works.

## 3. Prefill and decode are different functions

Split them: `forward_prefill(&[u32]) -> Tensor` and `forward_decode(u32) -> Tensor`.
Resist the urge to unify — they want different masks (causal matrix vs none at
all, since one query attends to everything cached), different shapes, and on
Day 3 they'll want different scheduling.

Then swap your hand-rolled attention for
`candle_nn::ops::sdpa(q, k, v, mask, do_causal, scale, softcapping)`
(`candle-nn-0.11.0/src/ops.rs:1308`). Per its own docs, on Metal it takes a
vectorized kernel when `seq == 1` — your decode path — and **handles GQA
natively when `qhead` is a multiple of `kv_head`**.

So delete your `repeat_kv` call and pass the `[1, 2, T, 128]` K/V straight in.
Materializing the 6× repeat was allocating and writing 6× the KV bytes every
step, in the exact regime where bytes are the bottleneck.

Benchmark before and after. Then benchmark prefill separately from decode, and
compare each against the 5.9 TFLOP/s from Day 0. Prefill should get within
striking distance; decode should be nowhere near. That gap is the lesson.

## 4. Sampling

Argmax is deterministic and dull. Implement, in this order — each operates on
the `[151936]` logits vector before softmax unless noted:

| Knob | Effect |
|---|---|
| **repetition penalty** | divide (or multiply, if negative) logits of already-seen tokens by `p`. Applied to logits, before temperature |
| **temperature** | `logits / t`. `t → 0` approaches argmax, `t > 1` flattens |
| **top-k** | keep the `k` highest, `−inf` the rest |
| **top-p** (nucleus) | sort desc, keep the shortest prefix whose cumulative probability ≥ `p` |
| **min-p** | keep tokens with `prob ≥ min_p × max_prob`. Scales with confidence — better than top-p in practice |

Order matters and is a real source of "why is my output different from
llama.cpp": penalties → temperature → truncation (k/p) → softmax → sample.

Take a seed and use `rand::SeedableRng`. An unreproducible sampler makes every
later quality comparison — int8 vs bf16 KV on Day 4, YaRN on Day 5 — worthless.

**Stopping** is subtler than it looks. EOS is `151643` and `max_tokens` is
trivial, but stop-strings are not: you cannot emit a token until you know it
isn't the start of a stop string. Buffer a tail of decoded text, and when you do
stop, truncate the stop string out of the output rather than shipping it.

## 5. The server

`axum`, `POST /v1/completions`. Non-streaming first — get the request/response
shapes right against something you can `curl` — then SSE with
`axum::response::Sse` and `tokio_stream`.

**The detokenization trap.** Qwen uses byte-level BPE. A token is a sequence of
*bytes*, not necessarily a complete character. Decode token-by-token and you
will emit `U+FFFD` in the middle of every emoji and CJK character, and mangle
leading spaces (Qwen encodes them into the token, not between tokens).

Keep a buffer: decode `all_tokens_so_far`, compare against what you've already
emitted, and emit only the newly-valid UTF-8 suffix. Test with an emoji prompt
and a Chinese prompt — if those render clean, you've got it.

The generation loop is blocking and Metal work isn't `Send`-friendly across
await points. Simplest structure that survives Day 3: generation on a dedicated
thread, tokens out over an `mpsc` channel, the axum handler turning that
receiver into an SSE stream. Building it this way today means Day 3's scheduler
is a change of what's on the far end of the channel, not a rewrite.

## 6. Gate

- [ ] Cached decode logits == Day 1 recompute logits, same prefix
- [ ] ≥10× tok/s over Day 1
- [ ] `curl -N localhost:8080/v1/completions` streams token by token
- [ ] Emoji and CJK prompts render with no replacement characters
- [ ] Same seed + same params → identical output, twice
- [ ] **Recorded:** single-stream TTFT, decode tok/s, prefill tok/s

Write those three numbers in the README. Everything on Days 3–4 is measured
against them, and you cannot reconstruct them later.
