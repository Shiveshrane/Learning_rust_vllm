use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use engine::quant_kv::KVDType;
use engine::paged_attn::KVPool;
use engine::scheduler::{Job, Scheduler};
use qwen::config::QwenConfig;
use qwen::device::{pick, Backend};
use qwen::model::Qwen2;
use qwen::paths::ModelPaths;
use tokio::sync::mpsc;

pub fn spawn()->mpsc::UnboundedSender<Job>{
    let (job_tx, mut job_rx)=mpsc::unbounded_channel::<Job>();
    std::thread::spawn(move ||{
        let mut state=match load(){
            Ok(s)=>s,
            Err(e)=>{
                eprintln!("Failed to load model: {e:#}");
                return;
            }
        };
        loop{
            while let Ok(job)=job_rx.try_recv(){
                let _=state.scheduler.admit(job, &state.tok, state.vocab_size, state.eos);
            }
            if state.scheduler.is_idle(){
                match job_rx.blocking_recv(){
                    Some(job)=>{
                        let _=state.scheduler.admit(job, &state.tok, state.vocab_size, state.eos);
                    }
                    None=>break,
                }
                continue;
            }
            if let Err(e)=state.scheduler.step(&state.model, &state.tok, &state.device){
                eprintln!("scheduler step failed: {e:#}");
            }
        }
    });
    job_tx
}

struct WorkerState{
    model:Qwen2,
    scheduler:Scheduler,
    device:Device,
    tok: tokenizers::Tokenizer,
    eos:u32,
    vocab_size:usize,
}

fn load()->Result<WorkerState>{
    let device=pick(Backend::Auto)?;
    let paths=ModelPaths::from_cache()?;
    let cfg=QwenConfig::from_path(&paths.config)?;
    let tok=tokenizers::Tokenizer::from_file(&paths.tokenizer)
    .map_err(anyhow::Error::from_boxed)?;
    let t0=std::time::Instant::now();
    let vb=unsafe{VarBuilder::from_mmaped_safetensors(&paths.weights, DType::F32, &device)?};
    let model=Qwen2::load(&cfg,4096, vb)?;
    //let cache=KVCache::new(&cfg, 4096, DType::F32, &device)?;

    const BLOCK_SIZE:usize=16;
    let budget:usize=std::env::var("KV_BUDGET_BYTES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2_000_000_000);
    let num_blocks=engine::block::blocks_for_budget(budget, &cfg, BLOCK_SIZE, 4);
    let pool=KVPool::new(&cfg, num_blocks, BLOCK_SIZE, KVDType::F32, &device)?;
    println!(
        "KV pool: {num_blocks} blocks x {BLOCK_SIZE} tokens = {} tokens ({:.1} GB)",
        num_blocks * BLOCK_SIZE,
        (num_blocks * BLOCK_SIZE * cfg.kv_bytes_per_token(4)) as f64 / 1e9
    );
    let scheduler=Scheduler::new(pool, BLOCK_SIZE);
    let vocab_size=tok.get_vocab_size(true);
    println!("Loaded model in {:.2}s", t0.elapsed().as_secs_f32());
    Ok(WorkerState{
        model,
        scheduler,
        device,
        tok,
        eos: cfg.eos_token_id,
        vocab_size,
    })
}
