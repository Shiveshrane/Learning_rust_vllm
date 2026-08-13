# Day 4 — KV quantization and prefix caching

**Goal:** twice the tokens in the same memory, and measurements that prove what
it cost you.

**Concepts:** activation outliers · per-channel vs per-token quantization ·
why quantizing KV buys *throughput* · content-addressed blocks

---

## 1. Quantization in one page

Store a tensor in fewer bits by mapping its float range onto integers.

Symmetric int8, the simplest form:

```
scale = max(|x|) / 127
q     = round(x / scale)        # i8
x̂     = q · scale               # dequantized
```

One f32 scale per group of values, plus one byte per value instead of two (from
bf16). The error is roughly `scale/2` per element — so **the scale is everything**,
and the scale is set by the largest magnitude in the group. One outlier 100×
larger than its neighbours makes every other value in that group garbage.

Which means the real question is never "how many bits" but **"what do I group
together?"** Small groups = more scales = better fidelity, more overhead. The
group's *shape* matters as much as its size.

## 2. Why the KV cache is the right target

From Day 2: decode is memory-bandwidth-bound. From Day 3: your block pool holds
419k tokens at 28 KB/token.

Halve KV to int8 and you get **both**: 838k tokens resident (more sequences in
the batch) *and* half the bytes read per attention step. On a bandwidth-bound
workload, memory saved converts directly into throughput. That's the
non-obvious payoff, and it's why this comes after Day 3 rather than before.

Weights are the bigger 3.5 GB target, but they're a solved, boring problem
(load a GGUF). KV is where the interesting structure is.

## 3. The interesting part: K and V want different schemes

Here's the finding worth internalizing (from the KIVI paper, 2024).

**Key and Value tensors have different structure.**

Key vectors have strong **per-channel outliers** — specific dimensions of the
head are consistently, massively larger than others, across all tokens. And K
errors are amplified: keys go through `q·kᵀ` and then **softmax**, which is
exponential. A small logit error becomes a large probability error.

Value vectors are better behaved, with no strong channel structure, and V errors
pass through only a linear weighted sum — no amplification.

So:

| | Group along | Why |
|---|---|---|
| **K** | per-channel (one scale per head dim, across tokens) | isolates the outlier channels so they don't poison well-behaved ones |
| **V** | per-token (one scale per token, across head dims) | no channel structure to exploit; per-token is simpler and streams naturally |

Per-channel K has an awkward consequence worth thinking through: the scale spans
*all tokens in a block*, so you can't finalize it until the block is full.
Handle the partial trailing block separately — that's a real design problem, not
an oversight.

**Do it in this order, and measure between steps:**

1. int8, per-token for both K and V — the naive version
2. switch K to per-channel — watch the quality gap close
3. int4 with group-wise scales (groups of 32 or 64 along head dim)

Skipping step 1 means you never see the difference, and the difference *is* the
lesson. This is a day for measurement, not for arriving at the right answer fast.

## 4. Implementation shape

Store quantized blocks as `i8` (or packed `u8` for int4, two values per byte)
with scale tensors alongside. **Dequantize on read**, in the gather step you
already wrote on Day 3 — you're gathering blocks into a contiguous tensor
anyway, so dequantize there and hand `sdpa` normal bf16.

Yes, a fused dequant-in-attention kernel is the production answer, and no,
you're not writing one. Same honest tradeoff as Day 3's gather: you keep the
capacity win in full and part of the bandwidth win. Note it in the README.

Keep the KV dtype a runtime flag — `bf16 | int8 | int4`. You need to A/B it in
the same binary or the benchmark matrix is a nightmare.

## 5. Measuring quality honestly

"Looks fine to me" is not a measurement. Three levels:

**Perplexity** — the real check. Run a held-out passage (~2000 tokens) through
the model and compute `exp(mean(−log P(actual_next_token)))`. Compare bf16 vs
int8 vs int4. A well-implemented int8 KV should be within ~1% of baseline. If
it's 20% worse, you have a bug, not a quantization limit — most likely your K
grouping axis.

**Logit MSE** — against the same golden file from Day 1. Fast, catches gross
errors, doesn't tell you whether output quality actually degrades.

**Eyeball** — same seed, same prompt, bf16 vs int4. Where do they diverge? Often
the first ~50 tokens are identical and they split at a genuinely uncertain
choice. That's informative about *how* the degradation works.

## 6. Prefix caching

Ten chat requests share a 500-token system prompt. Day 3 prefills all 500 tokens
ten times, computing bit-identical KV each time. Waste.

Blocks are immutable once full, and a block's KV depends only on the tokens in
it *and every token before it*. So content-address them:

```
block_hash = hash(parent_block_hash, token_ids_in_block)
```

Chaining the parent hash is essential — the same 16 tokens after different
prefixes produce completely different KV, and a collision here yields silently
wrong output.

Then: a hash table from block hash → physical block id, a **refcount** per
block, and blocks with refcount 0 kept in an **LRU** list rather than freed
immediately (they may be reused). When a sequence needs to write into a block
with refcount > 1, **copy-on-write**: allocate a fresh block, copy, decrement.
Only full blocks get hashed; a partial trailing block is private.

The demo: fire 10 requests sharing a long system prompt. First pays full TTFT,
the rest should collapse. Prefix caching is often a larger real-world win than
everything else this week combined — production traffic is overwhelmingly shared
system prompts and multi-turn conversations replaying their own history.

## 7. The benchmark harness

A `bench` binary that sweeps concurrency `1/2/4/8/16/32` and reports:

- **TTFT** p50/p99 — time to first token, the latency users feel
- **ITL** p50/p99 — inter-token latency, whether streaming feels smooth
- **Output tok/s** — aggregate throughput
- **Peak blocks used** — how close you ran to preemption

p99 matters more than p50 and is where scheduling bugs hide: a mean that looks
fine can hide one sequence being starved every time.

Emit CSV. Run the matrix:

```
KV dtype {bf16, int8, int4} × prefix caching {on, off} × concurrency {1..32}
```

These numbers are the substance of your README. Measuring your own optimizations
— including the ones that turn out not to help — is the part self-taught engine
work almost always skips.

## 8. Gate

- [ ] int8 KV: ~2× tokens resident in the same pool, perplexity delta quantified (not guessed)
- [ ] Per-channel K measurably beats per-token K — you have both numbers
- [ ] int4 works, with its cost stated
- [ ] Shared-prefix TTFT collapses after the first request
- [ ] Block invariant still holds with refcounting and CoW in play
- [ ] `bench` CSV produced for the full matrix

One prediction to check yourself against: does int8 KV improve throughput at
concurrency 1? It shouldn't, much — you're not memory-bound on KV with one short
sequence. The win should grow with concurrency and context length. If your
numbers say otherwise, find out why before you believe them.
