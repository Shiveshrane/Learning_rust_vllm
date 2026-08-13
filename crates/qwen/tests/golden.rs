//! TEMPORARY — Day 1 correctness gate. Delete this whole file when Day 1 closes.
//!
//! Nothing in `src/` depends on it: it drives the public API only, so removal is
//! `rm crates/qwen/tests/golden.rs` and nothing else.
//!
//! Compares our prefill logits against HuggingFace's, produced by
//! `scripts/golden.py` into `tests/golden/logits.npz` (fp32, CPU).
//!
//! Run on CPU to answer "my math or the Metal kernel?":
//!     QWEN_BACKEND=cpu cargo test -p qwen --test golden -- --nocapture

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::VarBuilder;
use qwen::cache::KVCache;
use qwen::config::QwenConfig;
use qwen::device::{from_env, pick};
use qwen::model::Qwen2;
use qwen::paths::ModelPaths;

const GOLDEN: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden/logits.npz");
const PROMPT: &str = "The capital of France is";

/// The gate's tolerance. Note this was written assuming bf16 weights; we load
/// f32 against an f32 reference, so passing anywhere near 1e-2 means something
/// is wrong and is merely hiding under a loose threshold. Watch the printed
/// numbers, not just the pass/fail.
const TOL: f32 = 1e-2;

/// Prefill and decode run identical math in different groupings, on the same
/// device and dtype — so they should agree far more tightly than we agree with
/// HuggingFace. Reusing TOL here would let a real position bug slip through.
const DECODE_TOL: f32 = 1e-3;

/// Max absolute elementwise difference.
fn max_abs_diff(ours: &Tensor, gold: &Tensor) -> Result<f32> {
    Ok((ours - gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?)
}

#[test]
fn logits_match_hf_golden() -> Result<()> {
    let cpu = Device::Cpu;
    let golden = Tensor::read_npz_by_name(GOLDEN, &["input_ids", "logits"])?;
    let (gold_ids, gold_logits) = (&golden[0], &golden[1]);

    // --- 1. tokenization must match, or every later number is meaningless ----
    let path = ModelPaths::from_cache()?;
    let tok = tokenizers::Tokenizer::from_file(&path.tokenizer).map_err(anyhow::Error::from_boxed)?;
    let ids: Vec<u32> = tok
        .encode(PROMPT, false)
        .map_err(anyhow::Error::from_boxed)?
        .get_ids()
        .to_vec();
    let gold_ids: Vec<u32> = gold_ids.to_vec1::<i64>()?.iter().map(|&i| i as u32).collect();
    assert_eq!(ids, gold_ids, "tokenization diverges from the reference");
    let t = ids.len();

    // --- 2. our prefill ------------------------------------------------------
    let device = pick(from_env()?)?;
    let cfg = QwenConfig::from_path(&path.config)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&path.weights, DType::F32, &device)? };
    let model = Qwen2::load(&cfg, 4096, vb)?;

    let mut cache = KVCache::new(&cfg, 4096, DType::F32, &device)?;
    let input = Tensor::new(ids.as_slice(), &device)?.unsqueeze(0)?;
    let ours = model
        .forward_prefill(&input, &mut cache)?
        .i(0)?
        .to_dtype(DType::F32)?
        .to_device(&cpu)?;

    assert_eq!(ours.dims(), gold_logits.dims(), "logits shape mismatch");

    // --- 3. per-position error, so a bad tail is visible ---------------------
    println!("\n  pos   max|diff|");
    for p in 0..t {
        println!("  {p:>3}   {:>9.5}", max_abs_diff(&ours.i(p)?, &gold_logits.i(p)?)?);
    }

    // --- 4. the argmax the gate names ---------------------------------------
    let top = ours.i(t - 1)?.argmax(D::Minus1)?.to_scalar::<u32>()?;
    let gold_top = gold_logits.i(t - 1)?.argmax(D::Minus1)?.to_scalar::<u32>()?;
    let text = tok.decode(&[top], false).map_err(anyhow::Error::from_boxed)?;
    println!("\n  argmax: ours {top} ({text:?}), golden {gold_top}");
    assert_eq!(top, gold_top, "next-token argmax disagrees with the reference");
    assert_eq!(top, 12095, "expected ' Paris' (12095)");

    // --- 5. the gate ---------------------------------------------------------
    let worst = max_abs_diff(&ours, gold_logits)?;
    println!("  max|diff| over all {t}x{} logits: {worst:.6}", cfg.vocab_size);
    assert!(worst < TOL, "logits differ by {worst:.6}, tolerance is {TOL}");

    // --- 6. Day 2 gate: the cached decode path must equal the recompute path --
    //
    // Same token, two routes. Path A above prefilled all `t` tokens and read
    // row t-1. Path B prefills t-1 of them, then pushes the last one through
    // `forward_decode` — offset t-1, RoPE position t-1, no mask, attending over
    // the cached prefix plus itself.
    //
    // A miscounted `advance`, an offset read after advancing, or a `narrow` to
    // seq_len instead of end all move this number off zero.
    cache.reset();
    let head = Tensor::new(&ids[..t - 1], &device)?.unsqueeze(0)?;
    model.forward_prefill(&head, &mut cache)?;

    let tail = Tensor::new(&ids[t - 1..], &device)?.unsqueeze(0)?;
    let decoded = model
        .forward_decode(&tail, &mut cache)?
        .i((0, 0))?
        .to_dtype(DType::F32)?
        .to_device(&cpu)?;

    let vs_prefill = max_abs_diff(&decoded, &ours.i(t - 1)?)?;
    let vs_golden = max_abs_diff(&decoded, &gold_logits.i(t - 1)?)?;
    println!("\n  decode vs our prefill: {vs_prefill:.6}");
    println!("  decode vs HF golden:   {vs_golden:.6}\n");

    assert_eq!(
        decoded.argmax(D::Minus1)?.to_scalar::<u32>()?,
        12095,
        "decode path picked a different next token"
    );
    assert!(
        vs_prefill < DECODE_TOL,
        "cached decode differs from recompute by {vs_prefill:.6}, tolerance is {DECODE_TOL}"
    );
    assert!(vs_golden < TOL, "decode differs from HF by {vs_golden:.6}");
    Ok(())
}
