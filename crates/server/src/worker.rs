use crate:: api::CompletionRequest;
use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use engine::sampling::{Params, Sampler};
use engine::stop::{StopReason, Stopper};
use qwen::cache::KVCache;
use qwen::config::QwenConfig;
use qwen::device::{pick, Backend};
use qwen::model::Qwen2;
use qwen::paths::ModelPaths;
use tokio::sync::mpsc;
use engine::detokenize::Detokenizer;

pub enum Event{
    Token(String),
    Done {reason:StopReason, prompt_tokens:usize, completion_tokens:usize},
    Error(String),
}


pub struct Job{
    pub req:CompletionRequest,
    pub tx:mpsc::UnboundedSender<Event>,
}


pub fn spawn()->mpsc::UnboundedSender<Job>{
    let (job_tx, mut job_rx)=mpsc::unbounded_channel::<Job>();
    std::thread::spawn(move ||{
        let mut state=match load(){
            Ok(s)=>s,
            Err(e)=>{
                eprintln!("Failed to load model: {}", e);
                return;
            }
        };
        while let Some(job)=job_rx.blocking_recv(){
            let tx=job.tx.clone();
            if let Err(e)=run_job(&mut state, job){
                let _=tx.send(Event::Error(format!("Failed to run job: {}", e)));
            }
        }
    });
    job_tx
}

struct WorkerState{
    model:Qwen2,
    cache:KVCache,
    device:Device,
    tok: tokenizers::Tokenizer,
    eos:u32,
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
    let cache=KVCache::new(&cfg, 4096, DType::F32, &device)?;
    println!("Loaded model in {:.2}s", t0.elapsed().as_secs_f32());
    Ok(WorkerState{
        model,
        cache,
        device,
        tok,
        eos: cfg.eos_token_id,
    })
}

fn run_job(st: &mut WorkerState, job:Job)->Result<()>{
    let req=job.req;

    let mut sampler=Sampler::new(Params{
        temperature: req.temperature,
        top_k: req.top_k,
        top_p: req.top_p,
        min_prob: req.min_prob,
        repetition_penalty: req.repeat_penalty,
        seed: req.seed,

    },
    st.tok.get_vocab_size(true)
);
    let mut stopper=Stopper::new(st.eos, req.max_tokens, req.stop.clone());
    let ids:Vec<u32>=st.tok.encode(req.prompt.as_str(), false)
    .map_err(anyhow::Error::from_boxed)?
    .get_ids().to_vec();
    let prompt_tokens=ids.len();
    st.cache.reset();
    let mut input=Tensor::new(ids.as_slice(), &st.device)?.unsqueeze(0)?;
    // let logits=st.model.forward_prefill(&input, &mut st.cache)?;
    let logits=st.model.forward_prefill(&input, &st.cache, 0)?;
    let mut last=logits.i((0, prompt_tokens-1))?.to_dtype(DType::F32)?.to_device(&st.device)?;

    // Position is the caller's job now: the cache no longer tracks it.
    // The prompt occupies 0..T, so the first generated token sits at T.
    let mut pos=prompt_tokens;
    let mut all=ids;
    let mut detok=Detokenizer::new(&st.tok);
    loop{
        let top=sampler.sample(&last, &all)?;
        //let piece=st.tok.decode(&[top], false).map_err(anyhow::Error::from_boxed)?;
        let piece=detok.push(top)?;
        let step=stopper.push(top, &piece);
        if !step.text.is_empty(){
            let _=job.tx.send(Event::Token(step.text));
        }
        if let Some(reason)=step.stop{
            let _=job.tx.send(Event::Done{
                reason,
                prompt_tokens,
                completion_tokens:stopper.generated(),
            });
            return Ok(());
        }
        all.push(top);
        let inp=Tensor::new(&[top], &st.device)?.unsqueeze(0)?;
        // last = st.model.forward_decode(&inp, &mut st.cache)?.i((0, 0))?.to_dtype(DType::F32)?;
        last = st.model.forward_decode(&inp, &st.cache, pos)?.i((0, 0))?.to_dtype(DType::F32)?;
        pos+=1;

    }
}
