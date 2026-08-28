// ===========================================================================
// TESTS WRITTEN BY CLAUDE — Day 3 Block 3 gate.
//
// Drives Scheduler::step directly, no HTTP. Two claims:
//
//   1. Continuous batching produces correct output for several concurrent
//      sequences sharing one pool.
//   2. Deliberately over-subscribing the pool forces preemption, and the
//      preempted sequences still come back with the SAME text they would have
//      produced with room to spare. Recompute must be invisible in the output.
//
// The pool sizes are chosen so that (2) cannot pass without preemption firing:
// three sequences need ~15 blocks between them and are given 8.
// ===========================================================================

use anyhow::Result;
use candle_core::DType;
use candle_nn::VarBuilder;
use engine::quant_kv::KVDType;
use engine::paged_attn::KVPool;
use engine::sampling::Params;
use engine::scheduler::{Event, Job, Request, Scheduler};
use qwen::config::QwenConfig;
use qwen::device::{from_env, pick};
use qwen::model::Qwen2;
use qwen::paths::ModelPaths;
use tokio::sync::mpsc;

const PROMPT: &str = "The capital of France is";
const BLOCK_SIZE: usize = 16;
const MAX_STEPS: usize = 4000;

struct Harness {
    model: Qwen2,
    tok: tokenizers::Tokenizer,
    device: candle_core::Device,
    cfg: QwenConfig,
}

fn load() -> Result<Harness> {
    let device = pick(from_env()?)?;
    let path = ModelPaths::from_cache()?;
    let cfg = QwenConfig::from_path(&path.config)?;
    let tok =
        tokenizers::Tokenizer::from_file(&path.tokenizer).map_err(anyhow::Error::from_boxed)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&path.weights, DType::F32, &device)? };
    let model = Qwen2::load(&cfg, 4096, vb)?;
    Ok(Harness { model, tok, device, cfg })
}

/// Run `n` identical greedy requests through a pool of `num_blocks`, stepping
/// until every stream reports Done. Returns the text each stream received.
fn run(h: &Harness, num_blocks: usize, n: usize, max_tokens: usize) -> Result<(Vec<String>, usize)> {
    let pool = KVPool::new(&h.cfg, num_blocks, BLOCK_SIZE, KVDType::F32, &h.device)?;
    let mut sched = Scheduler::new(pool, BLOCK_SIZE);
    let vocab = h.tok.get_vocab_size(true);
    let eos = h.cfg.eos_token_id;

    let mut rxs = Vec::new();
    for _ in 0..n {
        let (tx, rx) = mpsc::unbounded_channel();
        let job = Job {
            req: Request {
                prompt: PROMPT.to_string(),
                max_tokens,
                params: Params::default(), // temperature 0.0 => greedy, reproducible
                stop: vec![],
            },
            tx,
        };
        sched.admit(job, &h.tok, vocab, eos)?;
        rxs.push(rx);
    }

    let mut texts = vec![String::new(); n];
    let mut done = vec![false; n];
    let mut steps = 0;
    while !done.iter().all(|d| *d) {
        sched.step(&h.model, &h.tok, &h.device)?;
        steps += 1;
        for (i, rx) in rxs.iter_mut().enumerate() {
            while let Ok(ev) = rx.try_recv() {
                match ev {
                    Event::Token(s) => texts[i].push_str(&s),
                    Event::Done { .. } => done[i] = true,
                    Event::Error(e) => panic!("stream {i} errored: {e}"),
                }
            }
        }
        assert!(steps < MAX_STEPS, "no progress after {MAX_STEPS} steps — livelock?");
    }
    Ok((texts, steps))
}

#[test]
fn concurrent_sequences_share_a_pool_correctly() -> Result<()> {
    let h = load()?;
    let (texts, steps) = run(&h, 256, 4, 20)?;
    println!("\n  4 concurrent, roomy pool: {steps} steps");
    println!("  {:?}", texts[0]);
    for (i, t) in texts.iter().enumerate() {
        assert!(!t.is_empty(), "stream {i} produced nothing");
        assert_eq!(*t, texts[0], "greedy streams diverged: {i}");
    }
    assert!(texts[0].starts_with(" Paris"), "expected ' Paris', got {:?}", texts[0]);
    Ok(())
}

/// The gate item: over-subscribe until preemption fires; output still correct.
#[test]
fn preemption_does_not_change_output() -> Result<()> {
    let h = load()?;
    let n = 3;
    let max_tokens = 60;

    // Roomy: every sequence fits at once, so nothing is ever preempted.
    let (baseline, roomy_steps) = run(&h, 256, n, max_tokens)?;

    // Tight: 3 sequences need ~5 blocks each once grown; they get 8 total.
    let (tight, tight_steps) = run(&h, 8, n, max_tokens)?;

    println!("\n  roomy pool (256 blocks): {roomy_steps} steps");
    println!("  tight pool (  8 blocks): {tight_steps} steps");
    println!("  baseline text: {:?}", baseline[0]);
    println!("  tight    text: {:?}", tight[0]);

    assert!(
        tight_steps > roomy_steps,
        "tight pool took no extra steps ({tight_steps} vs {roomy_steps}) — \
         preemption never fired, so this test proves nothing"
    );
    for i in 0..n {
        assert_eq!(
            tight[i], baseline[i],
            "stream {i} differs after preemption:\n  want {:?}\n  got  {:?}",
            baseline[i], tight[i]
        );
    }
    Ok(())
}
