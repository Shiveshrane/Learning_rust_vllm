use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{embedding, linear, linear_no_bias, Embedding, Linear, Module, VarBuilder};
use candle_nn::ops::sdpa;
use crate::cache::{BatchedKvStore, KVCache, KVStore};
use crate::config::QwenConfig;

pub struct RMSNorm{
    pub eps: f64,
    pub weight: Tensor,
}

impl RMSNorm{
    pub fn load(size: usize, eps: f64, vb: VarBuilder)-> Result<Self>{
        Ok(Self{
            eps,
            weight: vb.get(size, "weight")?,
        })
    }
    pub fn forward(&self, x: &Tensor)-> Result<Tensor>{
        let dtype=x.dtype();
        let x32=x.to_dtype(DType::F32)?;
        let ms=x32.sqr()?.mean_keepdim(D::Minus1)?;
        let inv=ms.affine(1.0, self.eps)?.sqrt()?;
        let normed=x32.broadcast_div(&inv)?.to_dtype(dtype)?;
        Ok(normed.broadcast_mul(&self.weight)? )
    }
}

pub struct Mlp {
    gate_proj: Linear, 
    down_proj: Linear,
    up_proj: Linear,
}

impl Mlp {
    pub fn load(cfg: &QwenConfig, vb: VarBuilder)->Result<Self>{
        let (h, i)= (cfg.hidden_size, cfg.intermediate_size);
        Ok(Self{
            gate_proj:linear_no_bias(h, i, vb.pp("gate_proj"))?,
            down_proj:linear_no_bias(i, h, vb.pp("down_proj"))?,
            up_proj:linear_no_bias(h, i, vb.pp("up_proj"))?,
        })
    }

    pub fn forward(&self, x:&Tensor)-> Result<Tensor>{
        let gate=self.gate_proj.forward(x)?.silu()?;
        let up=self.up_proj.forward(x)?;
        Ok(self.down_proj.forward(&gate.mul(&up)?)?)
    }
}

pub struct RoPE{
    pub cos: Tensor,
    pub sin: Tensor,
}

impl RoPE{
    pub fn new(cfg: &QwenConfig, max_seq: usize, dtype: DType, device: &Device)->Result<Self>{
        let head_dim=cfg.head_dim();
        let theta=cfg.rope_theta as f32;
        let inv_freq:Vec<f32>=(0..head_dim/2).map(|i| 1f32/theta.powf(2.0*i as f32/head_dim as f32)).collect();
        let inv_freq=Tensor::from_vec(inv_freq, (1, head_dim/2), device)?;

        let pos: Vec<f32>=(0..max_seq).map(|i| i as f32).collect();
        let pos=Tensor::from_vec(pos, (max_seq, 1), device)?;
        let freqs=pos.broadcast_mul(&inv_freq)?;
        Ok(Self{
            // cos: freqs.cos()?.to_dtype(dtype)?,
            // sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?,
            sin: freqs.sin()?,
        })
    }

    pub fn apply_at(&self, x:&Tensor, pos:&Tensor)-> Result<Tensor>{
        let (b, _h, s, d)=x.dims4()?;
        debug_assert_eq!(s, 1, "apply_at is the single-token decode path");
        let half=d/2;
        // [B, half] -> [B, 1, 1, half] so it broadcasts across heads.
        let cos=self.cos.index_select(pos, 0)?.reshape((b, 1, 1, half))?;
        let sin=self.sin.index_select(pos, 0)?.reshape((b, 1, 1, half))?;
        let x1=x.narrow(D::Minus1, 0, half)?.to_dtype(DType::F32)?;
        let x2=x.narrow(D::Minus1, half, half)?.to_dtype(DType::F32)?;
        let r1=((x1.broadcast_mul(&cos)?) - (x2.broadcast_mul(&sin)?))?;
        let r2=((x1.broadcast_mul(&sin)?) + (x2.broadcast_mul(&cos)?))?;
        Ok(Tensor::cat(&[r1, r2], D::Minus1)?.to_dtype(x.dtype())?)
    }

    pub fn apply(&self, x:&Tensor, offset: usize)-> Result<Tensor>{
        let (_b, _h, s, d)=x.dims4()?;
        let half=d/2;
        let cos=self.cos.narrow(0, offset, s)?;
        let sin=self.sin.narrow(0, offset, s)?;
        let x1=x.narrow(D::Minus1, 0, half)?.to_dtype(DType::F32)?;
        let x2=x.narrow(D::Minus1, half, half)?.to_dtype(DType::F32)?;
        let r1=((x1.broadcast_mul(&cos)?) - (x2.broadcast_mul(&sin)?))?;
        let r2=((x1.broadcast_mul(&sin)?) + (x2.broadcast_mul(&cos)?))?;
        Ok(Tensor::cat(&[r1, r2], D::Minus1)?.to_dtype(x.dtype())?)
    }
}

pub struct Attention{
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_groups: usize,
    scale: f64,
}

impl Attention{
    pub fn load(cfg:&QwenConfig, vb: VarBuilder)->Result<Self>{
        let h=cfg.hidden_size;
        let head_dim=cfg.head_dim();
        let kv_dim=cfg.num_key_value_heads*head_dim;

        Ok(Self{
            q_proj:linear(h, h, vb.pp("q_proj"))?,
            k_proj:linear(h, kv_dim, vb.pp("k_proj"))?,
            v_proj:linear(h, kv_dim, vb.pp("v_proj"))?,
            o_proj:linear_no_bias(h, h, vb.pp("o_proj"))?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim, 
            kv_groups: cfg.kv_groups(),
            scale: 1.0/(head_dim as f64).sqrt(),
        })
    }

    pub fn repeat_kv(&self, x: &Tensor)->Result<Tensor>{
        if self.kv_groups==1{
            return Ok(x.clone());
        }
        let (b, kv_h, s, d)=x.dims4()?;
        Ok(x.unsqueeze(2)?
        .expand((b, kv_h, self.kv_groups, s, d))?
        .contiguous()?
        .reshape((b, kv_h*self.kv_groups, s, d))?)
    }

    pub fn forward_batch(
        &self,
        x:&Tensor,                       // [B, 1, hidden]
        rope:&RoPE,
        store:&dyn BatchedKvStore,
        layer:usize,
        pos_ids:&Tensor,                 // [B] u32
        mask:&Tensor,                    // [B, num_heads, 1, max_len]
    )->Result<Tensor>{
        let (b, s, _)=x.dims3()?;
        let q=self.q_proj.forward(x)?
            .reshape((b, s, self.num_heads, self.head_dim))?
            .transpose(1,2)?.contiguous()?;
        let k=self.k_proj.forward(x)?
            .reshape((b, s, self.num_kv_heads, self.head_dim))?
            .transpose(1,2)?.contiguous()?;
        let v=self.v_proj.forward(x)?
            .reshape((b, s, self.num_kv_heads, self.head_dim))?
            .transpose(1,2)?.contiguous()?;

        let q=rope.apply_at(&q, pos_ids)?;
        let k=rope.apply_at(&k, pos_ids)?;

        store.write_batch(layer, &k, &v)?;
        let (k, v)=store.gather_batch(layer)?;

        let k=repeat_kv(&k, self.kv_groups)?;
        let v=repeat_kv(&v, self.kv_groups)?;
        let scores=q.matmul(&k.transpose(2,3)?.contiguous()?)?;
        let scores=scores.affine(self.scale, 0.0)?;
        let scores=scores.broadcast_add(mask)?;
        let probs=candle_nn::ops::softmax_last_dim(&scores.to_dtype(DType::F32)?)?
            .to_dtype(scores.dtype())?;
        let out=probs.matmul(&v)?;
        let out=out.transpose(1,2)?.contiguous()?
            .reshape((b, s, self.num_heads*self.head_dim))?;
        Ok(self.o_proj.forward(&out)?)
    }

    pub fn forward(&self, 
        x:&Tensor, 
        rope: &RoPE, 
        mask: Option<&Tensor>, 
        offset: usize, 
        //cache: &mut KVCache,
        store: &dyn KVStore,
        layer: usize)->Result<Tensor>{
        let (b, s, _)=x.dims3()?;
        let q=self.q_proj.forward(x)?
        .reshape((b,s,self.num_heads,self.head_dim))?
        .transpose(1,2)?.contiguous()?;
        let k=self.k_proj.forward(x)?
        .reshape((b,s,self.num_kv_heads,self.head_dim))?
        .transpose(1,2)?.contiguous()?;
        let v=self.v_proj.forward(x)?
        .reshape((b,s,self.num_kv_heads,self.head_dim))?
        .transpose(1,2)?.contiguous()?;

        let q=rope.apply(&q, offset)?;
        let k=rope.apply(&k, offset)?;

        store.write(layer, offset, &k, &v)?;
        let (k, v) = store.gather(layer, offset + s)?;

        // let (k, v)=cache.append(layer, &k, &v)?;

        // let k=self.repeat_kv(&k)?; //Remove after SDPA
        // let v=self.repeat_kv(&v)?; //Remove after SDPA

        // let scores=q.matmul(&k.transpose(2,3)?.contiguous()?)?;
        // let scores=scores.affine(self.scale, 0.0)?;
        // let scores=match mask{
        //     Some(m)=>scores.broadcast_add(m)?,
        //     None=>scores,
        // };
        // let probs=candle_nn::ops::softmax_last_dim(&scores.to_dtype(DType::F32)?)?
        // .to_dtype(scores.dtype())?;
        // let out=probs.matmul(&v)?;
        let out=sdpa(&q, &k, &v, mask, false, self.scale as f32, 1.0)?; //used SDPA instead of manual matmul, softmax, and matmul for better performance
        let out=out.transpose(1,2)?.contiguous()?
        .reshape((b,s,self.num_heads*self.head_dim))?;
        Ok(self.o_proj.forward(&out)?)
    }
}

pub struct DecoderLayer{
    input_layernorm: RMSNorm, 
    self_attention: Attention,
    post_attention_layernorm: RMSNorm,
    mlp: Mlp,
}

impl DecoderLayer{
    pub fn load(cfg: &QwenConfig, vb:VarBuilder)->Result<Self>{
        Ok(Self{
            input_layernorm: RMSNorm::load(
                cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            self_attention: Attention::load(cfg, vb.pp("self_attn"))?,
            post_attention_layernorm: RMSNorm::load(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?,
            mlp: Mlp::load(cfg, vb.pp("mlp"))?,
        })

    }
    pub fn forward_batch(
        &self,
        x:&Tensor,
        rope:&RoPE,
        store:&dyn BatchedKvStore,
        layer:usize,
        pos_ids:&Tensor,
        mask:&Tensor,
    )->Result<Tensor>{
        let residual=self.input_layernorm.forward(x)?;
        let residual=self.self_attention.forward_batch(&residual, rope, store, layer, pos_ids, mask)?;
        let x=(x+residual)?;
        let residual=self.post_attention_layernorm.forward(&x)?;
        let residual=self.mlp.forward(&residual)?;
        Ok((x+residual)?)
    }

    pub fn forward(&self, 
        x: &Tensor, 
        rope: &RoPE, 
        mask:Option<&Tensor>, 
        offset:usize, 
        //cache: &mut KVCache,
        store: &dyn KVStore, 
        layer: usize)->Result<Tensor>{
        let residual=self.input_layernorm.forward(x)?;
        let residual=self.self_attention.forward(&residual, rope, mask, offset, store, layer)?;
        let x=(x+residual)?;
        let residual=self.post_attention_layernorm.forward(&x)?;
        let residual=self.mlp.forward(&residual)?;
        Ok((x+residual)?)
    }

}

fn repeat_kv(x:&Tensor, groups:usize)->Result<Tensor>{
    if groups==1{
        return Ok(x.clone());
    }
    let (b, kv_h, s, d)=x.dims4()?;
    Ok(x.unsqueeze(2)?
        .expand((b, kv_h, groups, s, d))?
        .contiguous()?
        .reshape((b, kv_h*groups, s, d))?)
}

fn causal_mask(s:usize, offset:usize, dtype: DType, device:&Device)->Result<Tensor>{
    let kv_len=offset+s;
    let data: Vec<f32>=(0..s).flat_map(|i|{
        (0..kv_len).map(move |j| if j>i+offset {f32::NEG_INFINITY} else {0.0})
    })
    .collect();
    Ok(Tensor::from_vec(data, (s, kv_len), device)?.to_dtype(dtype)?)
}

pub struct Qwen2{
    embedding: Embedding,
    layers: Vec<DecoderLayer>,
    final_layernorm: RMSNorm,
    lm_head: Linear,
    rope: RoPE,
    device: Device,
    dtype: DType,
}

impl Qwen2{
    pub fn load(cfg: &QwenConfig, max_seq:usize, vb:VarBuilder)->Result<Self>{
        let device=vb.device().clone();
        let dtype=vb.dtype();
        let m=vb.pp("model");
        let embedding=embedding(cfg.vocab_size, cfg.hidden_size, m.pp("embed_tokens"))?;
        let mut layers=Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers{
            layers.push(DecoderLayer::load(cfg, m.pp(format!("layers.{}", i)))?);

        }
        let final_layernorm=RMSNorm::load(cfg.hidden_size, cfg.rms_norm_eps, m.pp("norm"))?;
        let lm_head=linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
        let rope=RoPE::new(cfg, max_seq, dtype, &device)?;
        Ok(Self{
            embedding,
            layers,
            final_layernorm,
            lm_head,
            rope,
            device,
            dtype,
        })
    }

    // pub fn forward(&self, input_ids:&Tensor, offset:usize)->Result<Tensor>{
    //     let (_b, s)=input_ids.dims2()?;
    //     let mut h=self.embedding.forward(input_ids)?;
    //     let mask=if s==1{
    //         None
    //     }else{
    //         Some(causal_mask(s, offset, self.dtype, &self.device)?)
    //     };
    //     for (i, layer) in self.layers.iter().enumerate(){
    //         h=layer.forward(&h, &self.rope, mask.as_ref(), offset)?;

    //         //-----------------------------------------------------------------
    //         // Debug: per-layer activation magnitude. Run with TRACE=1 in bf16
    //         // and again in f32, then diff the columns. A jump at one layer
    //         // means a bug there; smooth growth across all 28 means accumulation.
    //         // if std::env::var("TRACE").is_ok(){
    //         //     let absmax=h.to_dtype(DType::F32)?
    //         //         .abs()?
    //         //         .flatten_all()?
    //         //         .max(0)?
    //         //         .to_scalar::<f32>()?;
    //         //     eprintln!("layer {i:2}  absmax {absmax:.4}");
    //         // }
    //         // //-----------------------------------------------------------------
    //     }
    //     let h=self.final_layernorm.forward(&h)?;
    //     Ok(self.lm_head.forward(&h)?)
    // }

    pub fn forward_prefill(
        &self, 
        input_ids:&Tensor, 
        //cache: &mut KVCache
        store: &dyn KVStore,
        start_pos: usize
    )->Result<Tensor>{
        let (_b, s)=input_ids.dims2()?;
        let mut h=self.embedding.forward(input_ids)?;
        // let mask=if s==1{
            // None
        // }else{
        let mask=Some(causal_mask(s, start_pos, self.dtype, &self.device)?
        .reshape((1,1,s,start_pos+s))?
        .expand((1, self.layers[0].self_attention.num_heads, s, start_pos+s))?);
        // };
        
        for (i, layer) in self.layers.iter().enumerate(){
            h=layer.forward(&h, &self.rope, mask.as_ref(), start_pos, store, i)?;
        }
        //cache.advance(s);
        let h=self.final_layernorm.forward(&h)?;
        Ok(self.lm_head.forward(&h)?)
    }

    pub fn forward_decode_batch(
        &self,
        input_ids:&Tensor,
        store:&dyn BatchedKvStore,
    )->Result<Tensor>{
        let (b, _s)=input_ids.dims2()?;
        let lens=store.lens();
        let max_len=lens.iter().copied().max().unwrap_or(0);
        let num_heads=self.layers[0].self_attention.num_heads;

        let mut m=Vec::with_capacity(b*max_len);
        for &l in lens{
            for j in 0..max_len{
                m.push(if j<l {0f32} else {f32::NEG_INFINITY});
            }
        }
        let mask=Tensor::from_vec(m, (b, 1, 1, max_len), &self.device)?
            .to_dtype(self.dtype)?
            .expand((b, num_heads, 1, max_len))?
            .contiguous()?;

        let pos_ids=Tensor::new(store.positions(), &self.device)?;

        let mut h=self.embedding.forward(input_ids)?;
        for (i, layer) in self.layers.iter().enumerate(){
            h=layer.forward_batch(&h, &self.rope, store, i, &pos_ids, &mask)?;
        }
        let h=self.final_layernorm.forward(&h)?;
        Ok(self.lm_head.forward(&h)?)
    }

    pub fn forward_decode(
        &self, 
        input_ids:&Tensor, 
        //cache: &mut KVCache
        store: &dyn KVStore,
        start_pos: usize
    )->Result<Tensor>{
        let offset=start_pos;
        let mut h=self.embedding.forward(input_ids)?;
        for (i, layer) in self.layers.iter().enumerate(){
            h=layer.forward(&h, &self.rope, None, offset, store, i)?;
        }
        //cache.advance(1);
        let h=self.final_layernorm.forward(&h)?;
        Ok(self.lm_head.forward(&h)?)
    }
}