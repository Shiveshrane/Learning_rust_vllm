//! Qwen2 model: config, weights, layers, forward pass.
//!
//! Day 0 gives you plumbing only (model-path resolution and a timing macro).
//! Everything that teaches you something is a TODO — see the plan:
//!
//!   Day 1 Block 2  `config.rs`  Qwen2Config via serde, VarBuilder mmap load
//!   Day 1 Block 3  `model.rs`   RMSNorm, RoPE, GQA attention, SwiGLU, forward
//!   Day 2 Block 1  `cache.rs`   your own preallocated KV cache
//!   Day 5 Block 1  `rope.rs`    PI -> NTK-aware -> YaRN
pub mod config;  
pub mod cache;
pub mod device;
pub mod paths;
pub mod model;

/// Time an expression, returning `(value, elapsed)`.
///
/// GPU work is queued asynchronously, so timing a candle op that you never read
/// back measures the enqueue, not the compute. Force a sync before stopping the
/// clock — e.g. `t.sum_all()?.to_scalar::<f32>()?` — or your numbers are fiction.
#[macro_export]
macro_rules! timeit {
    ($e:expr) => {{
        let __start = std::time::Instant::now();
        let __v = $e;
        (__v, __start.elapsed())
    }};
}
