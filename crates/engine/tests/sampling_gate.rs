// ===========================================================================
// WRITTEN BY CLAUDE — Day 2 Block 3 gate.
//
// Two claims that the unit tests cannot make, because they need the real model:
//
//   1. Sampler with temperature 0.0 == the on-device argmax path. Proves the
//      logits -> to_vec1 -> truncate -> argmax plumbing is faithful, before any
//      randomness is involved.
//   2. Same seed + same params => byte-identical output, twice. This is the
//      gate item; without it every Day 4 quality comparison is meaningless.
//
// One model load covers both (F32 weights are ~7GB, and cargo runs tests in
// parallel — two #[test] fns would try to hold 14GB).
// ===========================================================================

use anyhow::Result;
use candle_core::{DType, IndexOp, Tensor, D};
use candle_nn::VarBuilder;
use engine::sampling::{Params, Sampler};
use qwen::cache::KVCache;
use qwen::config::QwenConfig;
use qwen::device::{from_env, pick};
use qwen::model::Qwen2;
use qwen::paths::ModelPaths;

const PROMPT: &str = "The capital of France is";
const N: usize = 20;

/// Generate `N` tokens. `sampler = None` uses candle's on-device argmax, which
/// is the Day 1 reference path.
fn generate(
    model: &Qwen2,
    cache: &mut KVCache,
    ids: &[u32],
    device: &candle_core::Device,
    mut sampler: Option<&mut Sampler>,
) -> Result<Vec<u32>> {
    cache.reset();
    let mut out: Vec<u32> = ids.to_vec();

    let input = Tensor::new(ids, device)?.unsqueeze(0)?;
    let logits = model.forward_prefill(&input, cache, 0)?;
    // Position is the caller's job since the KVStore refactor.
    let mut pos = ids.len();
    let mut last = logits.i((0, ids.len() - 1))?.to_dtype(DType::F32)?;

    for _ in 0..N {
        let top = match sampler.as_deref_mut() {
            Some(s) => s.sample(&last, &out)?,
            None => last.argmax(D::Minus1)?.to_scalar::<u32>()?,
        };
        out.push(top);
        let inp = Tensor::new(&[top], device)?.unsqueeze(0)?;
        last = model
            .forward_decode(&inp, cache, pos)?
            .i((0, 0))?
            .to_dtype(DType::F32)?;
        pos += 1;
    }
    Ok(out[ids.len()..].to_vec())
}

#[test]
fn greedy_matches_argmax_and_seeded_runs_reproduce() -> Result<()> {
    let device = pick(from_env()?)?;
    let path = ModelPaths::from_cache()?;
    let cfg = QwenConfig::from_path(&path.config)?;
    let tok =
        tokenizers::Tokenizer::from_file(&path.tokenizer).map_err(anyhow::Error::from_boxed)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&path.weights, DType::F32, &device)? };
    let model = Qwen2::load(&cfg, 4096, vb)?;
    let mut cache = KVCache::new(&cfg, 4096, DType::F32, &device)?;

    let ids: Vec<u32> = tok
        .encode(PROMPT, false)
        .map_err(anyhow::Error::from_boxed)?
        .get_ids()
        .to_vec();
    let vocab = tok.get_vocab_size(true);

    // --- 1. temperature 0.0 must reproduce the on-device argmax path ---------
    let reference = generate(&model, &mut cache, &ids, &device, None)?;

    let mut greedy = Sampler::new(
        Params {
            temperature: 0.0,
            ..Params::default()
        },
        vocab,
    );
    let sampled = generate(&model, &mut cache, &ids, &device, Some(&mut greedy))?;

    println!("\n  argmax   : {reference:?}");
    println!("  temp=0.0 : {sampled:?}");
    assert_eq!(
        reference, sampled,
        "temperature 0.0 diverged from the argmax path"
    );
    assert_eq!(reference[0], 12095, "expected ' Paris' as the first token");

    // --- 2. the gate: same seed => identical output --------------------------
    let params = || Params {
        temperature: 0.8,
        top_k: Some(50),
        top_p: Some(0.95),
        min_prob: None,
        repetition_penalty: Some(1.1),
        seed: Some(42),
    };

    let mut a = Sampler::new(params(), vocab);
    let run_a = generate(&model, &mut cache, &ids, &device, Some(&mut a))?;

    let mut b = Sampler::new(params(), vocab);
    let run_b = generate(&model, &mut cache, &ids, &device, Some(&mut b))?;

    println!("  seed=42 run A: {run_a:?}");
    println!("  seed=42 run B: {run_b:?}");
    assert_eq!(run_a, run_b, "same seed must produce identical output");

    // A different seed should diverge, or the seed is not being used at all.
    let mut c = Sampler::new(
        Params {
            seed: Some(43),
            ..params()
        },
        vocab,
    );
    let run_c = generate(&model, &mut cache, &ids, &device, Some(&mut c))?;
    println!("  seed=43 run C: {run_c:?}\n");
    assert_ne!(run_a, run_c, "different seeds must diverge");

    // Sampling with truncation on must stay inside the real vocab.
    assert!(
        run_a.iter().all(|&t| (t as usize) < vocab),
        "sampled a token id past the tokenizer's vocab"
    );
    Ok(())
}
