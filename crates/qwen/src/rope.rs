use std::f64::consts::PI;
use crate::config::{QwenConfig, RopeScaling};


pub fn find_correction_dim(rotations:f64, dim:usize, base:f64, max_pos:usize)->f64{
    (dim as f64*(max_pos as f64/(rotations*2.0*PI)).ln())/(2.0*base.ln())
}

pub fn find_correction_range(
    beta_fast:f64, beta_slow:f64, dim:usize, base:f64, max_pos:usize
)->(usize, usize){
   let low=find_correction_dim(beta_fast, dim, base, max_pos).floor();
   let high=find_correction_dim(beta_slow, dim, base, max_pos).ceil();
   let top=(dim-1) as f64;
   (low.clamp(0.0, top) as usize, high.clamp(0.0, top) as usize)
}


pub fn linear_ramp_mask(low:f64, high:f64, n:usize)->Vec<f64>{
    let high=if high==low{high+0.001} else{high};
    (0..n).map(
        |i| (
            (i as f64-low)/(high-low)
        ).clamp(0.0, 1.0)
    ).collect()
}


pub fn get_mscale(s:f64)->f64{
    if s<=1.0{
        1.0
    }
    else{
        0.1*s.ln()+1.0
    }
}

pub fn scaled_inv_freq(cfg:&QwenConfig)->(Vec<f32>, f32){
    let dim=cfg.head_dim();
    let theta=cfg.rope_theta as f32;
    let base:Vec<f32>=(0..dim/2)
        .map(|i| 1f32/theta.powf(2.0*(i as f32)/dim as f32))
        .collect();
    match &cfg.rope_scaling{
        None=>(base, 1.0),
        Some(RopeScaling::Pi{factor})=>{
            let s=*factor as f32;
            (base.iter().map(|f| f/s).collect(), 1.0)
        },
        Some(RopeScaling::NtkAware{factor})=>{
            let nb=(cfg.rope_theta*factor.powf(dim as f64/(dim as f64-2.0))) as f32;
            ((0..dim/2).map(|i| 1f32/nb.powf(2.0*i as f32/dim as f32)).collect(), 1.0)
        },

        Some(RopeScaling::Yarn{factor, original_max_position_embeddings, beta_fast, beta_slow})=>{
            let (low, high)=find_correction_range(
                *beta_fast, *beta_slow, dim, cfg.rope_theta, *original_max_position_embeddings
            );
            let ramp=linear_ramp_mask(low as f64, high as f64, dim/2);
            let s=*factor as f32;
            let inv=base.iter().zip(&ramp).map(|(f,r)| {
                let r=*r as f32;
                f*(1.0-r)+f/s*r
            }).collect();
            (inv, get_mscale(*factor) as f32)
        }
    }
}