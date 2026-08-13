//! The inference engine: everything between "a model that can do a forward
//! pass" and "a server that can serve many users at once".
//!
//! Empty by design — this is the crate you fill in. Roadmap:
//!
//!   Day 2 Block 3  `sampling.rs`   temperature, top-k, top-p, min-p, rep penalty
//!   Day 2 Block 3  `stop.rs`       EOS / max_tokens / stop-string tail buffer
//!   Day 3 Block 1  `blocks.rs`     BlockAllocator, block tables, the KV pool
//!   Day 3 Block 2  `paged_attn.rs` gather blocks -> contiguous -> sdpa
//!   Day 3 Block 3  `scheduler.rs`  Sequence state machine, continuous batching
//!   Day 4 Block 1  `quant_kv.rs`   int8 (per-channel K, per-token V), then int4
//!   Day 4 Block 2  `prefix.rs`     block hashing, refcounts, copy-on-write, LRU
//!
//! The invariant to assert every iteration once `blocks.rs` exists:
//!     allocator.free() + allocator.allocated() == allocator.total()
//! It catches nearly every paging bug the moment it happens rather than three
//! layers downstream as garbled output.
pub mod sampling;
pub mod stop; // WRITTEN BY CLAUDE — Day 2 Block 3