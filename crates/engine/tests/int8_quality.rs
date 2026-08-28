// TESTS WRITTEN BY CLAUDE — Day 4 Block 1, int8 KV quality at real dimensions.
//
// The unit tests run a synthetic config (head_dim 3) on CPU. This runs the real
// checkpoint on the real device, because that is where head_dim 128, genuine
// activation outliers, and Metal kernels actually live.
//
// Reports rather than asserts a tight bound: the point is to KNOW what int8
// costs before deciding whether it is a bug or the scheme.
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

#[test]
fn int8_kv_logit_error_at_real_dimensions() -> Result<()> {
    let device = pick(from_env()?)?;
    let path = ModelPaths::from_cache()?;
    let cfg = QwenConfig::from_path(&path.config)?;
    let tok =
        tokenizers::Tokenizer::from_file(&path.tokenizer).map_err(anyhow::Error::from_boxed)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&path.weights, DType::F32, &device)? };
    let model = Qwen2::load(&cfg, 4096, vb)?;

    let ids: Vec<u32> = tok
        .encode("The capital of France is", false)
        .map_err(anyhow::Error::from_boxed)?
        .get_ids()
        .to_vec();
    let input = Tensor::new(ids.as_slice(), &device)?.unsqueeze(0)?;

    let mut out = Vec::new();
    for dt in [KVDType::F32, KVDType::Int8] {
        let pool = KVPool::new(&cfg, 32, BS, dt, &device)?;
        let mut table = BlockTable::new(BS);
        table.append_block(31);
        let store = PagedStore::new(&pool, &table);
        let logits = model.forward_prefill(&input, &store, 0)?;
        let row = logits.i((0, ids.len() - 1))?.to_dtype(DType::F32)?;
        let top = row.argmax(D::Minus1)?.to_scalar::<u32>()?;
        let piece = tok.decode(&[top], false).map_err(anyhow::Error::from_boxed)?;
        println!("\n  {dt:?}: argmax {top} -> {piece:?}");
        out.push(row);
    }

    let diff = (&out[1] - &out[0])?.abs()?;
    let max = diff.max(0)?.to_scalar::<f32>()?;
    let mean = diff.mean(0)?.to_scalar::<f32>()?;
    let scale = out[0].abs()?.max(0)?.to_scalar::<f32>()?;
    println!("  logit max|diff| = {max:.4}   mean|diff| = {mean:.4}");
    println!("  relative to peak logit {scale:.4}: {:.2}%", 100.0 * max / scale);
    Ok(())
}

/// Is K worse than V, or are they equally bad?
///
/// The KIVI claim is that key tensors carry strong per-channel outliers while
/// values do not, so per-token grouping (over head_dim) hurts K far more. If
/// both reconstruct equally well, a 16% logit error is a bug elsewhere. If K is
/// markedly worse, this is the naive scheme behaving exactly as advertised and
/// per-channel K is the fix.
///
/// Uses REAL K/V: prefill through an F32 pool, read them back, then round-trip
/// the same tensors through an int8 pool.
#[test]
fn is_it_k_or_v_that_quantizes_badly() -> Result<()> {
    let device = pick(from_env()?)?;
    let path = ModelPaths::from_cache()?;
    let cfg = QwenConfig::from_path(&path.config)?;
    let tok =
        tokenizers::Tokenizer::from_file(&path.tokenizer).map_err(anyhow::Error::from_boxed)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&path.weights, DType::F32, &device)? };
    let model = Qwen2::load(&cfg, 4096, vb)?;

    let ids: Vec<u32> = tok
        .encode("The capital of France is a city with a long history", false)
        .map_err(anyhow::Error::from_boxed)?
        .get_ids()
        .to_vec();
    let input = Tensor::new(ids.as_slice(), &device)?.unsqueeze(0)?;
    let len = ids.len();

    let f32_pool = KVPool::new(&cfg, 32, BS, KVDType::F32, &device)?;
    let mut table = BlockTable::new(BS);
    table.append_block(31);
    table.append_block(30);
    {
        let store = PagedStore::new(&f32_pool, &table);
        model.forward_prefill(&input, &store, 0)?;
    }

    let q_pool = KVPool::new(&cfg, 32, BS, KVDType::Int8, &device)?;
    let mut worst_k = 0f32;
    let mut worst_v = 0f32;
    let mut rel_k = 0f32;
    let mut rel_v = 0f32;

    for layer in 0..cfg.num_hidden_layers {
        let (k, v) = f32_pool.gather(layer, &table, len)?;
        q_pool.write(layer, &table, 0, &k, &v)?;
        let (kq, vq) = q_pool.gather(layer, &table, len)?;

        let ke = (&kq - &k)?.abs()?.max(0)?.max(0)?.max(0)?.max(0)?.to_scalar::<f32>()?;
        let ve = (&vq - &v)?.abs()?.max(0)?.max(0)?.max(0)?.max(0)?.to_scalar::<f32>()?;
        let kmax = k.abs()?.max(0)?.max(0)?.max(0)?.max(0)?.to_scalar::<f32>()?;
        let vmax = v.abs()?.max(0)?.max(0)?.max(0)?.max(0)?.to_scalar::<f32>()?;

        worst_k = worst_k.max(ke);
        worst_v = worst_v.max(ve);
        rel_k = rel_k.max(ke / kmax);
        rel_v = rel_v.max(ve / vmax);
    }

    println!("\n  K: max abs err {worst_k:.4}   worst relative {:.3}%", rel_k * 100.0);
    println!("  V: max abs err {worst_v:.4}   worst relative {:.3}%", rel_v * 100.0);
    println!(
        "  K/V error ratio: {:.1}x",
        (worst_k / worst_v.max(1e-9)).max(0.0)
    );
    Ok(())
}
