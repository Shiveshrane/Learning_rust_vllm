use anyhow::{Context, Result};
use serde ::{Deserialize};
use std::path::Path;
#[derive(Debug, Clone, Deserialize)]

pub struct QwenConfig{
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub tie_word_embeddings: bool,
    pub eos_token_id: u32,
}

impl QwenConfig{
    // parse config.json from a path
    pub fn from_path(path:&Path)-> Result<Self>{
        let bytes=std::fs::read(path).with_context(|| format!("Failed to read config file from path: {:?}", path.display()))?;
        serde_json::from_slice(&bytes).with_context(|| format!("Failed to parse config file from path: {:?}", path.display()))

    }
    pub fn head_dim(&self)-> usize{
        self.hidden_size/self.num_attention_heads
    }
    pub fn kv_groups(&self)-> usize{
        self.num_attention_heads/self.num_key_value_heads
    }


    pub fn kv_bytes_per_token(&self, bytes_per_elem:usize)-> usize{
        2*self.num_hidden_layers*self.num_key_value_heads*self.head_dim()*bytes_per_elem
    }
}


#[cfg(test)]
mod tests{
use super::*;
use crate::paths::ModelPaths;
use candle_core::DType;
use candle_nn::VarBuilder;
use crate::device::{Backend};


#[test]

fn parse_checkpoint_config()->Result<()>{
    let model_path=ModelPaths::from_cache()?;
    let cfg=QwenConfig::from_path(&model_path.config)?;
    assert_eq!(cfg.num_hidden_layers, 28);
    assert_eq!(cfg.head_dim(), 128);
    assert_eq!(cfg.kv_bytes_per_token(2), 28_672);
    assert_eq!(cfg.kv_groups(), 6);
    assert!(!cfg.tie_word_embeddings);
    Ok(())
}

#[test]
fn tokenizer_match_golden()->Result<()>{
    let path=ModelPaths::from_cache()?;
    let tok=tokenizers::Tokenizer::from_file(&path.tokenizer).map_err(anyhow::Error::from_boxed)?;
    let enc=tok.encode("The capital of France is", false).map_err(anyhow::Error::from_boxed)?;
    assert_eq!(enc.get_ids(), &[785, 6722, 315, 9625, 374]);
    Ok(())
}
#[test]

fn load_weights()->Result<()>{
    let device=crate::device::pick(Backend::Auto)?;
    let path=ModelPaths::from_cache()?;
    let cfg=QwenConfig::from_path(&path.config)?;
    let vb=unsafe{
        VarBuilder::from_mmaped_safetensors(&path.weights, DType::BF16,  &device)?
    };
    let attn=vb.pp("model.layers.0.self_attn");
    let q=attn.get((cfg.hidden_size, cfg.hidden_size), "q_proj.weight")?;
    let k=attn.get((
        cfg.num_key_value_heads*cfg.head_dim(),
        cfg.hidden_size,
    ), "k_proj.weight")?;
    assert_eq!(q.dims(), &[1536, 1536]);
    assert_eq!(k.dims(), &[256, 1536]);
    Ok(())
}
}

