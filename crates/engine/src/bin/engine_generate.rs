use anyhow::Result;
use candle_core::{DType, Device, Tensor, IndexOp, D};
use candle_nn::VarBuilder;
use qwen::config::QwenConfig;
use qwen::model::Qwen2;
use qwen::device::{pick, Backend};
use qwen::paths::ModelPaths;
use qwen::cache::KVCache;
use engine::sampling::{Params, Sampler};
use engine::stop::{Stopper, StopReason};

const PROMPT:&str="Which city is the capital of France?";

fn greedy_decode(model: &Qwen2, tok:&tokenizers::Tokenizer, input:&[u32], device:&Device, max_new_tokens:usize, cache: &mut KVCache, sampler: &mut Sampler, stopper: &mut Stopper)->Result<String>{
    let mut ip_ids=input.to_vec();
    cache.reset();
    let input=Tensor::new(ip_ids.as_slice(), device)?.unsqueeze(0)?;
    // let logits=model.forward_prefill(&input, cache)?;
    let logits=model.forward_prefill(&input, cache, 0)?;
    let mut last_tok=logits.i((0, ip_ids.len()-1))?.to_dtype(DType::F32)?;
    let mut pos=ip_ids.len();
    // for i in 0..max_new_tokens{
    //     //let top=last_tok.argmax(D::Minus1)?.to_scalar::<u32>()?;
    //     let top=sampler.sample(&last_tok, &ip_ids)?;
    //     if top==eos{
    //         println!("Step {}: Generated EOS token ID {}, stopping.", i+1, top);
    //         break;
    //     }
    //     ip_ids.push(top);
    //     let text=tok.decode(&[top], false).map_err(anyhow::Error::from_boxed)?;
    //     println!("Step {}: Generated token ID {} -> {:?}", i+1, top, text);
    //     let inp=Tensor::new(&[top], device)?.unsqueeze(0)?;
    //     let logits=model.forward_decode(&inp, cache)?;
    //     last_tok=logits.i((0, 0))?.to_dtype(DType::F32)?;

    // }
    // Ok(tok.decode(&ip_ids, false).map_err(anyhow::Error::from_boxed)?)

    loop{
        let top=sampler.sample(&last_tok, &ip_ids)?;
        let piece=tok.decode(&[top], false).map_err(anyhow::Error::from_boxed)?;
        let step=stopper.push(top, &piece);
       eprintln!("Step {}: token {} -> {:?}", stopper.generated(), top, piece);
        if let Some(reason)=step.stop{
            eprintln!("Stopping due to {:?}", reason);
            break;

    }
    ip_ids.push(top);
    let inp=Tensor::new(&[top], device)?.unsqueeze(0)?; 
    // let logits=model.forward_decode(&inp, cache)?;
    let logits=model.forward_decode(&inp, cache, pos)?;
    pos+=1;
    last_tok=logits.i((0, 0))?.to_dtype(DType::F32)?;
    }
    Ok(stopper.text().to_string())
}
fn main()->Result<()>{

    let params=Params{
    temperature:0.5,
    top_k: None,
    top_p: None,
    min_prob: None,
    repetition_penalty: None,
    seed: None,
    };
    let device=pick(Backend::Auto)?;
    let path=ModelPaths::from_cache()?;
    let cfg=QwenConfig::from_path(&path.config)?;
    let tok=tokenizers::Tokenizer::from_file(&path.tokenizer).map_err(anyhow::Error::from_boxed)?;
    let mut sampler=Sampler::new(params, tok.get_vocab_size(true));
    let mut stopper=Stopper::new(cfg.eos_token_id, 512, vec![]);
    let mut cache=KVCache::new(&cfg, 4096, DType::F32, &device)?;
    let t0=std::time::Instant::now();
    let vb=unsafe{
        VarBuilder::from_mmaped_safetensors(&path.weights, DType::F32,  &device)?
    };
    let model=Qwen2::load(&cfg, 4096, vb)?;
    println!("Loaded weights in {:?}", t0.elapsed());

    let ids=tok
    .encode(PROMPT, false)
    .map_err(anyhow::Error::from_boxed)?
    .get_ids()
    .to_vec();
    println!("Prompt: {:?}", PROMPT);
    println!("Token IDs: {:?}", ids);

    let t0=std::time::Instant::now();
    let out = greedy_decode(&model, &tok, &ids, &device, 512, &mut cache, &mut sampler, &mut stopper)?;
    println!("Generated in {:.3}s", t0.elapsed().as_secs_f32());
    println!("Output: {}", out);
    Ok(())

}