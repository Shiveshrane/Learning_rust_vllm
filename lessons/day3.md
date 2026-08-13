# Day 3 — Paged KV cache and continuous batching

**Goal:** many concurrent requests, sharing the GPU, with no memory waste.

This is the heart of vLLM and the most valuable day of the week. If a day slips,
protect this one.

**Concepts:** fragmentation · block tables as page tables · iteration-level
scheduling · preemption · head-of-line blocking

---

## 1. The problem with Day 2's cache

You preallocated `[1, kv_heads, max_seq, head_dim]` per sequence. Consider
serving with `max_seq = 4096`:

- A request that generates 50 tokens holds 4096 tokens' worth of memory. **99% wasted.**
- You can't know the output length in advance, so you can't size it smaller.
- With 12 GB of KV budget at 28 KB/token, you fit `12e9 / (28672 × 4096)` ≈ **102 sequences**, no matter how short they are.

This is *internal fragmentation* — space reserved inside an allocation that's
never used. There's *external fragmentation* too: free sequences leave holes,
and a new sequence needs one contiguous run.

The 2023 vLLM paper measured real systems wasting **60–80%** of KV memory this
way. The fix is the oldest idea in operating systems.

## 2. Paging

An OS doesn't give a process one contiguous run of physical RAM. It hands out
fixed-size pages and keeps a page table mapping virtual → physical. Fragmentation
collapses because any free page serves any request.

Do exactly that to the KV cache:

- **Physical pool**, per layer: `[num_blocks, block_size, kv_heads, head_dim]`, `block_size = 16`
- **Block table**, per sequence: `Vec<u32>` mapping logical block index → physical block id
- **Free list**: allocator hands out block ids and takes them back

A sequence's KV is now scattered across the pool. Logical position `p` lives at
physical block `table[p / 16]`, slot `p % 16`.

Waste drops to at most 15 tokens per sequence (the partly-filled last block) —
under 0.4% instead of 99%. You allocate a block only when the sequence actually
grows into it.

**Sizing the pool** — do this at startup and log it, the way vLLM's
`gpu_memory_utilization` does:

```
24 GB unified − 3.5 GB weights − ~2 GB activations/OS ≈ 12 GB for KV
12e9 / 28672 bytes-per-token ≈ 419,000 tokens ≈ 26,000 blocks of 16
```

Compare to 102 sequences before. That is the entire ballgame.

**The invariant.** Assert it every scheduler iteration, from the first line of
code you write:

```
free_blocks + allocated_blocks == total_blocks
```

Paging bugs otherwise surface three layers downstream as garbled text, and you
will spend two hours in the attention kernel looking for a bug in the allocator.

## 3. Attention over scattered memory

Here's the honest part. vLLM's speed comes from a custom CUDA kernel that walks
the block table *inside* the kernel, so scattered KV costs nothing extra. candle
has no such kernel for Metal, and writing one is a week on its own.

So: **gather the blocks into a contiguous tensor with `index_select`, then call
`sdpa`.**

Be clear about what this trades. You keep the memory-management win — no
fragmentation, no over-reservation, and prefix sharing becomes possible tomorrow.
You give up part of the bandwidth win, because you're copying KV each step
instead of reading it in place.

Write it down in the README as a known gap. Understanding precisely which half
of an idea you implemented is worth more than pretending you got both, and a
hand-written MSL paged-attention kernel is the natural project for next week.
(You'd need `sudo xcode-select -s /Applications/Xcode.app` first — your
`xcode-select` points at CommandLineTools and `metal` isn't on PATH.)

## 4. Continuous batching

### Why batching is nearly free

From Day 2: decode is memory-bandwidth-bound at ~2 FLOP/byte. Running one
sequence reads 3.5 GB of weights to make 1 token. Running 16 sequences reads the
same 3.5 GB once and makes 16 tokens. Nearly 16× the throughput for nearly the
same time.

That's why an inference server exists as a distinct thing from a `generate()`
function.

### Why *continuous*, not static

Static batching collects 8 requests, runs them to completion, then takes the next
8. If seven finish in 20 tokens and one runs to 500, seven slots idle for 480
steps. Worse, a request arriving one step after the batch forms waits for the
whole batch — head-of-line blocking.

Continuous batching schedules at the **iteration** level. Every step, the batch
is whatever's currently running. A sequence finishes → its slot is refilled that
same step. A request arrives → it joins at the next step.

### What to build

A `Sequence` state machine — `Waiting → Running → (Preempted | Finished)` — and
a loop that, each iteration:

1. Admits waiting sequences if the free-block budget allows their prompt
2. Runs one batched forward step over all running sequences
3. Appends a block to any sequence that just crossed a 16-token boundary
4. Frees blocks and closes channels for finished sequences
5. Asserts the block invariant

Engine on its own thread; requests arrive over an `mpsc`, each carrying its own
response `Sender`. If you built Day 2's server this way, this is a change of
what sits on the far end of the channel.

**Prefill vs decode in one step.** A new request needs a 500-token prefill; the
15 running sequences need 1 token each. Options:

- *Alternate* — prefill-only steps and decode-only steps. Simple. New arrivals
  stall every running sequence for one long step (a TTFT/ITL tradeoff you can measure).
- *Chunked prefill* — split the prefill into 256-token chunks, mix one chunk in
  with the decodes each step. Smoother inter-token latency, more bookkeeping.

Pick one, write down why in the README, and measure the other on Day 4 if
there's time.

**Preemption.** The pool will run out — sequences grow unpredictably. Two options:

- *Recompute* — drop the victim's KV, return it to `Waiting`, re-prefill later. Simple, and prefill is compute-bound and fast.
- *Swap* — copy its KV to host memory and back. More code, and on unified memory the win is questionable.

Take recompute. Choose the victim by last-arrived-first-preempted so you don't
starve anyone.

## 5. Gate

- [ ] 8 concurrent `curl -N` streams all advancing simultaneously
- [ ] Aggregate tok/s meaningfully above single-stream tok/s from Day 2
- [ ] After all requests finish, `free_blocks == total_blocks` — no leak
- [ ] A request arriving mid-batch starts within one iteration, not after the batch drains
- [ ] Deliberately over-subscribe until preemption fires; output still correct

That last one is worth engineering on purpose: set a tiny pool, fire 20 long
requests, and confirm the preempted ones come back with coherent output. Preemption
paths that have never run are preemption paths that don't work.

## 6. Measure the shape of it

Plot tok/s against concurrency 1/2/4/8/16/32. You should see near-linear scaling
early — that's the bandwidth argument paying out — then a bend as you become
compute-bound or run out of blocks.

Find your bend. Knowing *why* your particular curve bends where it does is the
thing you actually learned today.
