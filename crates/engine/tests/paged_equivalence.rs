// ===========================================================================
// WRITTEN BY CLAUDE — Day 3 Block 2 gate.
//
// Paging must not change the arithmetic. Same model, same prompt, same tokens,
// two KV backends:
//
//   A) KVCache   — Day 2's contiguous [1, kv_heads, max_seq, head_dim]
//   B) PagedStore — Day 3's pool, blocks handed out by BlockAllocator
//
// The logits should agree far more tightly than either agrees with HuggingFace:
// identical kernels on identical numbers, differing only in where the bytes
// live. Anything above 1e-5 means a real bug, not float noise.
//
// The allocator is deliberately churned first so the sequence's blocks are
// NON-CONTIGUOUS. With a tidy table like [0,1,2,...] the flat slot index equals
// the logical position and every paging bug hides.
// ===========================================================================

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::VarBuilder;
use engine::block::{BlockAllocator, BlockTable};
use engine::quant_kv::KVDType;
use engine::paged_attn::{KVPool, PagedStore};
use qwen::cache::{KVCache, KVStore};
use qwen::config::QwenConfig;
use qwen::device::{from_env, pick};
use qwen::model::Qwen2;
use qwen::paths::ModelPaths;

const PROMPT: &str = "The capital of France is";
const BLOCK_SIZE: usize = 16;
const NUM_BLOCKS: usize = 64;
/// 20 tokens on a 5-token prompt reaches position 25, so the sequence crosses
/// the block boundary at 16 — where needs_block/append_block have to cooperate.
const N: usize = 20;

fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    Ok((a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?)
}

/// Grow the table until it can hold `needed` tokens, holding a decoy block
/// between every real one so the sequence's blocks are guaranteed NOT adjacent.
/// Relying on incidental churn is not enough — the first version of this test
/// produced the table [40, 41] and proved nothing.
fn ensure_capacity(
    table: &mut BlockTable,
    alloc: &mut BlockAllocator,
    decoys: &mut Vec<u32>,
    needed: usize,
) {
    while table.capacity() < needed {
        decoys.push(alloc.allocate().expect("pool exhausted"));
        table.append_block(alloc.allocate().expect("pool exhausted"));
    }
}

/// No two logically adjacent blocks may be physically adjacent.
fn assert_scattered(blocks: &[u32]) {
    assert!(blocks.len() >= 2, "need >= 2 blocks to test scattering");
    for w in blocks.windows(2) {
        assert!(
            w[0].abs_diff(w[1]) > 1,
            "blocks {:?} are physically adjacent — paging is not exercised",
            w
        );
    }
}

#[test]
fn paged_kv_matches_contiguous_kv() -> Result<()> {
    let device = pick(from_env()?)?;
    let path = ModelPaths::from_cache()?;
    let cfg = QwenConfig::from_path(&path.config)?;
    let tok =
        tokenizers::Tokenizer::from_file(&path.tokenizer).map_err(anyhow::Error::from_boxed)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&path.weights, DType::F32, &device)? };
    let model = Qwen2::load(&cfg, 4096, vb)?;

    let ids: Vec<u32> = tok
        .encode(PROMPT, false)
        .map_err(anyhow::Error::from_boxed)?
        .get_ids()
        .to_vec();
    let t = ids.len();

    // --- backend A: Day 2's contiguous cache --------------------------------
    let cache = KVCache::new(&cfg, 4096, DType::F32, &device)?;

    // --- backend B: the paged pool ------------------------------------------
    let pool = KVPool::new(&cfg, NUM_BLOCKS, BLOCK_SIZE, KVDType::F32, &device)?;
    let mut alloc = BlockAllocator::new(NUM_BLOCKS);
    let mut decoys: Vec<u32> = Vec::new();
    let mut table = BlockTable::new(BLOCK_SIZE);
    // Two blocks up front so the prompt already spans a scattered table.
    ensure_capacity(&mut table, &mut alloc, &mut decoys, BLOCK_SIZE + 1);
    println!("\n  block table for the prompt: {:?}", table.blocks());
    assert_scattered(table.blocks());

    // --- prefill both -------------------------------------------------------
    let input = Tensor::new(ids.as_slice(), &device)?.unsqueeze(0)?;
    let a = model.forward_prefill(&input, &cache, 0)?;
    let b = {
        let store = PagedStore::new(&pool, &table);
        model.forward_prefill(&input, &store, 0)?
    };

    let prefill_diff = max_abs_diff(&a, &b)?;
    println!("  prefill  max|diff| = {prefill_diff:.9}");
    assert!(
        prefill_diff < 1e-5,
        "paged prefill diverged by {prefill_diff:.9}"
    );

    // --- decode N tokens through both, in lockstep --------------------------
    //
    // The same token is fed to both backends every step, so any divergence is
    // the KV layout rather than the two paths picking different tokens.
    let mut last_a = a.i((0, t - 1))?.to_dtype(DType::F32)?;
    let mut last_b = b.i((0, t - 1))?.to_dtype(DType::F32)?;
    let mut worst = prefill_diff;

    for step in 0..N {
        let top = last_a.argmax(D::Minus1)?.to_scalar::<u32>()?;
        let from_b = last_b.argmax(D::Minus1)?.to_scalar::<u32>()?;
        assert_eq!(top, from_b, "backends chose different tokens at step {step}");

        let pos = t + step;
        let inp = Tensor::new(&[top], &device)?.unsqueeze(0)?;

        last_a = model
            .forward_decode(&inp, &cache, pos)?
            .i((0, 0))?
            .to_dtype(DType::F32)?;

        // Grow the block table only when the sequence actually crosses a
        // boundary — this is the scheduler's step 3, in miniature.
        ensure_capacity(&mut table, &mut alloc, &mut decoys, pos + 1);
        last_b = {
            let store = PagedStore::new(&pool, &table);
            model
                .forward_decode(&inp, &store, pos)?
                .i((0, 0))?
                .to_dtype(DType::F32)?
        };

        let d = max_abs_diff(&last_a, &last_b)?;
        worst = worst.max(d);
        if step < 3 || d > 1e-5 {
            println!("  step {step:>2} pos {pos:>2}  max|diff| = {d:.9}");
        }
    }

    println!("  blocks after {N} tokens: {:?}", table.blocks());
    println!("  worst max|diff| over prefill + {N} decodes: {worst:.9}\n");
    assert!(
        table.len_blocks() > 1,
        "sequence never crossed a block boundary — lower BLOCK_SIZE or raise N"
    );
    assert_scattered(table.blocks());
    assert!(worst < 1e-5, "paged KV diverged by {worst:.9}");
    Ok(())
}
