// TESTS WRITTEN BY CLAUDE — Day 4 gate: "perplexity delta quantified, not guessed".
//
// Logit error is a proxy. Perplexity is the question actually being asked: how
// surprised is the model by text it did not generate? Runs a held-out passage
// through the model once per KV dtype and reports
//
//     ppl = exp( mean( -log P(actual next token) ) )
//
// The lesson's bar: a well-implemented int8 KV lands within ~1% of baseline.
// 20% worse means a bug, not a quantization limit.
use anyhow::Result;
use candle_core::{DType, IndexOp, Tensor, D};
use candle_nn::VarBuilder;
use engine::block::BlockTable;
use engine::paged_attn::{KVPool, PagedStore};
use engine::quant_kv::KVDType;
use qwen::config::QwenConfig;
use qwen::device::{from_env, pick};
use qwen::model::Qwen2;
use qwen::paths::ModelPaths;

const BS: usize = 16;
// 512 rather than the lesson's 2000: logits are [T, 151936], so each extra
// token costs 608KB and log_softmax doubles it. 512 keeps the transient under
// a gigabyte, well clear of the pool-size aliasing threshold found on Day 3.
const MAX_TOKENS: usize = 512;

fn passage() -> String {
    // Held-out technical English the model did not produce: the project's own
    // lesson notes. Deterministic, no download, and not in the training set in
    // this form.
    let mut s = String::new();
    for f in ["lessons/day2.md", "lessons/day3.md", "lessons/day4.md"] {
        let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../");
        if let Ok(t) = std::fs::read_to_string(format!("{p}{f}")) {
            s.push_str(&t);
        }
    }
    s
}

fn perplexity(model: &Qwen2, cfg: &QwenConfig, device: &candle_core::Device,
              ids: &[u32], dt: KVDType) -> Result<f64> {
    let need = ids.len().div_ceil(BS) + 4;
    let pool = KVPool::new(cfg, need, BS, dt, device)?;
    let mut table = BlockTable::new(BS);
    for b in 0..need {
        table.append_block(b as u32);
    }
    let store = PagedStore::new(&pool, &table);

    let input = Tensor::new(ids, device)?.unsqueeze(0)?;
    let logits = model.forward_prefill(&input, &store, 0)?.i(0)?.to_dtype(DType::F32)?;

    // Position t predicts token t+1, so drop the last row and the first target.
    let t = ids.len() - 1;
    let lp = candle_nn::ops::log_softmax(&logits.narrow(0, 0, t)?, D::Minus1)?;
    let targets = Tensor::new(&ids[1..], device)?.reshape((t, 1))?;
    let picked = lp.gather(&targets, D::Minus1)?;
    let nll = picked.mean_all()?.to_scalar::<f32>()? as f64;
    Ok((-nll).exp())
}

#[test]
fn int8_kv_perplexity_delta() -> Result<()> {
    let device = pick(from_env()?)?;
    let path = ModelPaths::from_cache()?;
    let cfg = QwenConfig::from_path(&path.config)?;
    let tok =
        tokenizers::Tokenizer::from_file(&path.tokenizer).map_err(anyhow::Error::from_boxed)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&path.weights, DType::F32, &device)? };
    let model = Qwen2::load(&cfg, 4096, vb)?;

    let text = passage();
    assert!(!text.is_empty(), "no held-out passage found");
    let mut ids: Vec<u32> = tok
        .encode(text.as_str(), false)
        .map_err(anyhow::Error::from_boxed)?
        .get_ids()
        .to_vec();
    ids.truncate(MAX_TOKENS);
    println!("\n  held-out passage: {} tokens", ids.len());

    let base = perplexity(&model, &cfg, &device, &ids, KVDType::F32)?;
    let q = perplexity(&model, &cfg, &device, &ids, KVDType::Int8)?;
    let delta = (q - base) / base * 100.0;

    println!("  perplexity  F32  : {base:.4}");
    println!("  perplexity  int8 : {q:.4}");
    println!("  delta            : {delta:+.2}%");
    println!(
        "  verdict          : {}",
        if delta.abs() < 1.0 { "within the ~1% bar" }
        else if delta.abs() < 20.0 { "above 1% — scheme cost, worth recording" }
        else { "20%+ — suspect a bug, not a quantization limit" }
    );

    assert!(base.is_finite() && q.is_finite());
    assert!(delta.abs() < 20.0, "int8 perplexity delta {delta:.2}% suggests a bug");
    Ok(())
}
