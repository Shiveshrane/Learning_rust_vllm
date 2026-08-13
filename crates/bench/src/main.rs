//! Load generator. Roadmap (Day 4 Block 3):
//!
//!   sweep concurrency 1/2/4/8/16/32, emit CSV with
//!     TTFT p50/p99, inter-token latency p50/p99, output tok/s, peak KV blocks
//!
//!   then run the matrix that becomes your README:
//!     KV dtype {bf16, int8, int4} x prefix caching {on, off}
//!
//! Record single-stream TTFT and tok/s here from Day 2 onward, so you have a
//! baseline to compare against before the engine gets complicated.

fn main() -> anyhow::Result<()> {
    println!("bench: not built yet — see Day 4 Block 3.");
    Ok(())
}
