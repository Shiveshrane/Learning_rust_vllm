use anyhow::Result;
use candle_core::{DType, Device, Tensor, D, IndexOp};
use candle_nn::VarBuilder;
use crate::config::QwenConfig;


pub struct KVCache{
    keys:Vec<Tensor>,
    values: Vec<Tensor>,
    max_seq: usize,
    seq_len: usize,
}

impl KVCache{
    pub fn new(cfg:&QwenConfig, max_seq:usize, dtype:DType, device:&Device)->Result<Self>{
        let n_layers=cfg.num_hidden_layers;
        let n_heads=cfg.num_key_value_heads;
        let head_dim=cfg.head_dim();
        let mut keys=Vec::with_capacity(n_layers);
        let mut values=Vec::with_capacity(n_layers);
        for _ in 0..n_layers{
            keys.push(Tensor::zeros(&[1, n_heads, max_seq, head_dim], dtype, device)?);
            values.push(Tensor::zeros(&[1, n_heads, max_seq, head_dim], dtype, device)?);
        }
        Ok(Self{
            keys,
            values,
            max_seq,
            seq_len:0,
        })
    }

    pub fn append(&mut self, layer:usize, new_keys:&Tensor, new_values:&Tensor)->Result<(Tensor, Tensor)>{
        let s=new_keys.dim(2)?;
        let end=self.seq_len+s;
        if end>self.max_seq{
            anyhow::bail!("KVCache overflow: max_seq={} but trying to append {} new
    tokens at seq_len={}", self.max_seq, s, self.seq_len);
        }
        self.keys[layer].slice_set(new_keys, 2, self.seq_len)?;
        self.values[layer].slice_set(new_values, 2, self.seq_len)?;
        let k=self.keys[layer].narrow(2,0,end)?.contiguous()?;
        let v=self.values[layer].narrow(2,0,end)?.contiguous()?;
        Ok((k, v))
    }


    pub fn advance(&mut self, n:usize){
        self.seq_len+=n;
    }


    pub fn len(&self)->usize{
        self.seq_len
    }
    pub fn reset(&mut self){
        self.seq_len=0;
        // self.keys.iter_mut().for_each(|k| k.zero_().unwrap());
        // self.values.iter_mut().for_each(|v| v.zero_().unwrap());
    }
}