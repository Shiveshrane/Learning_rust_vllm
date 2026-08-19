use qwen::config::QwenConfig;
use crate::block::BlockTable;
use candle_core::{DType, Device, Tensor};
use anyhow::Result;

pub struct KVPool{
    keys: Vec<Tensor>,
    values: Vec<Tensor>,
    block_size: usize,
    num_blocks: usize,
    device: Device,
}


impl KVPool{
    pub fn new(cfg:&QwenConfig, num_blocks:usize, block_size:usize, dtype: DType, device: &Device)->Result<Self>{
        let mut keys=Vec::with_capacity(cfg.num_hidden_layers);
        let mut values=Vec::with_capacity(cfg.num_hidden_layers);
        let slots=num_blocks*block_size;
        let kv_heads=cfg.num_key_value_heads;
        let head_dim=cfg.head_dim();
        for _ in 0..cfg.num_hidden_layers{
            let k=Tensor::zeros(&[slots, kv_heads, head_dim], dtype, device)?;
            let v=Tensor::zeros(&[slots, kv_heads, head_dim], dtype, device)?;
            keys.push(k);
            values.push(v);
        }
        Ok(Self { keys, values, block_size, num_blocks, device: device.clone() })
    }
    fn slot_of(&self, table:&BlockTable, pos:usize)->usize{
        let (block, slot)=table.locate(pos);
        block as usize*self.block_size+slot as usize
    }

    pub fn write(&self, layer:usize, table:&BlockTable, start_pos:usize, k:&Tensor, v:&Tensor)->Result<()>{
        let s=k.dim(2)?;
        let k3=k.squeeze(0)?.transpose(0,1)?.contiguous()?;
        let v3=v.squeeze(0)?.transpose(0,1)?.contiguous()?;
        for i in 0..s{
            let dst=self.slot_of(table, start_pos+i);
            self.keys[layer].slice_set(&k3.narrow(0,i,1)?.contiguous()?,0, dst)?;
            self.values[layer].slice_set(&v3.narrow(0,i,1)?.contiguous()?,0, dst)?;
        }
        Ok(())
    }
    pub fn gather(&self, layer:usize, table:&BlockTable, len:usize)->Result<(Tensor, Tensor)>{
        let idx:Vec<u32>=(0..len).map(|i| self.slot_of(table, i) as u32).collect();
        let idx=Tensor::new(idx.as_slice(), &self.device)?;

        let pick=|pool:&Tensor|->Result<Tensor>{
            Ok(pool
            .index_select(&idx, 0)?
            .transpose(0,1)?
            .contiguous()?
            .unsqueeze(0)?)
        };
        Ok((pick(&self.keys[layer])?, pick(&self.values[layer])?))
    }
    pub fn num_blocks(&self)->usize{
        self.num_blocks
    }
    pub fn block_size(&self)->usize{
        self.block_size
    }
}