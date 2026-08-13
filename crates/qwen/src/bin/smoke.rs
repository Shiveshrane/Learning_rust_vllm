//! Day 0 smoke test. Answers three questions before Day 1 starts:
//!
//!   1. Does candle's Metal backend build and run on this machine?
//!   2. How fast is it, in f32 vs bf16?
//!   3. Is the model checkpoint present, complete, and shaped how we think?
//!
//! Run with: cargo run --release --bin smoke

use anyhow::{Context, Result};
use candle_core::{safetensors::MmapedSafetensors, DType, Device, Tensor};
use qwen::{paths::ModelPaths, timeit};

const N: usize = 2048;
const ITERS: usize = 20;

/// Benchmark an NxN matmul. Returns effective TFLOP/s.
///
/// The `to_scalar` forces a device sync: Metal command buffers are queued
/// asynchronously, so without it we would time the enqueue and report a
/// gloriously wrong number.
fn bench_matmul(dev: &Device, dtype: DType) -> Result<f64> {
    let a = Tensor::randn(0f32, 1f32, (N, N), dev)?.to_dtype(dtype)?;
    let b = Tensor::randn(0f32, 1f32, (N, N), dev)?.to_dtype(dtype)?;

    // Warm up: the first call compiles/loads kernels and allocates buffers.
    let warm = a.matmul(&b)?;
    warm.sum_all()?.to_dtype(DType::F32)?.to_scalar::<f32>()?;

    let (out, elapsed) = timeit!({
        let mut last = None;
        for _ in 0..ITERS {
            last = Some(a.matmul(&b)?);
        }
        let last = last.unwrap();
        last.sum_all()?.to_dtype(DType::F32)?.to_scalar::<f32>()?;
        anyhow::Ok(last)
    });
    out?;

    let flops = 2.0 * (N as f64).powi(3) * ITERS as f64;
    Ok(flops / elapsed.as_secs_f64() / 1e12)
}

fn main() -> Result<()> {
    // --- 1. Metal ---------------------------------------------------------
    let dev = Device::new_metal(0).context(
        "Metal device unavailable. Is candle-core built with features = [\"metal\"]?",
    )?;
    println!("device: {dev:?}");

    for dtype in [DType::F32, DType::BF16, DType::F16] {
        match bench_matmul(&dev, dtype) {
            Ok(tflops) => println!("  {N}x{N} matmul {dtype:?}: {tflops:.2} TFLOP/s"),
            Err(e) => println!("  {N}x{N} matmul {dtype:?}: UNSUPPORTED ({e})"),
        }
    }

    // --- 2. Checkpoint ----------------------------------------------------
    let paths = ModelPaths::from_cache()?;
    println!("\nmodel: {}", paths.root.display());
    println!("  weights: {:.2} GB", paths.weights_bytes()? as f64 / 1e9);

    // Reading the safetensors header validates the file is not the truncated
    // `.incomplete` blob sitting next to it in the HF cache.
    let st = unsafe { MmapedSafetensors::multi(&paths.weights)? };
    let tensors = st.tensors();
    println!("  tensors: {}", tensors.len());
    for name in [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.bias",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "lm_head.weight",
    ] {
        match tensors.iter().find(|(n, _)| n == name) {
            Some((_, v)) => println!("    {name}: {:?} {:?}", v.shape(), v.dtype()),
            None => println!("    {name}: ABSENT"),
        }
    }
    // Qwen2's signature asymmetry, worth confirming with your own eyes on Day 0
    // because it is silent when you get it wrong on Day 1.
    let has = |n: &str| tensors.iter().any(|(name, _)| name == n);
    println!(
        "  qkv bias present: {}   o_proj bias present: {}",
        has("model.layers.0.self_attn.q_proj.bias"),
        has("model.layers.0.self_attn.o_proj.bias"),
    );

    // --- 3. The number the rest of the week is about ----------------------
    let cfg: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&paths.config)?).context("parsing config.json")?;
    let u = |k: &str| cfg[k].as_u64().unwrap_or(0) as usize;
    let (layers, kv_heads, heads, hidden) = (
        u("num_hidden_layers"),
        u("num_key_value_heads"),
        u("num_attention_heads"),
        u("hidden_size"),
    );
    let head_dim = hidden / heads;
    let kv_per_token = 2 * layers * kv_heads * head_dim * 2; // K and V, bf16

    println!("\narch: {layers} layers, {heads}Q/{kv_heads}KV heads, head_dim {head_dim}");
    println!("  KV per token (bf16): {} bytes", kv_per_token);
    println!(
        "  32k context = {:.2} GB of KV for ONE sequence",
        (kv_per_token * 32_768) as f64 / 1e9
    );
    println!(
        "  a 12 GB block pool holds {:.0}k tokens",
        12e9 / kv_per_token as f64 / 1e3
    );

    println!("\nDay 0 green. Do not open candle-transformers' qwen2.rs yet.");
    Ok(())
}
