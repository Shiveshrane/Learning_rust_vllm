use anyhow::{bail, Context, Result};
use candle_core::{DType, Device};


#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend{
    #[default]
    Auto, 
    Cpu, 
    Metal, 
    Cuda,
}

impl std::str::FromStr for Backend{
    type Err = anyhow::Error;

    fn from_str(s: &str)->Result<Self>{
        match s.to_lowercase().as_str(){
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal),
            "cuda" => Ok(Self::Cuda),
            _ => bail!("Unknown backend: {}", s)
        }
    }
}

pub fn pick(backend:Backend)->Result<Device>{
    match backend{
        Backend::Cpu=> Ok(Device::Cpu),
        Backend::Metal=> Device::new_metal(0).context("Failed to create Metal device"),
        Backend::Cuda=> Device::new_cuda(0).context("Failed to create CUDA device"),
        Backend::Auto=> match Device::new_metal(0){
            Ok(dev)=>Ok(dev),
            Err(e)=>{
            eprintln!("Failed to create Metal device: {e}. Falling back to CPU.");
            Ok(Device::Cpu)
        }
    },

    }
}

pub fn from_env()->Result<Backend>{
    match std::env::var("QWEN_BACKEND"){
        Ok(s)=>s.parse(),
        Err(_)=>Ok(Backend::default()),
    }
}



pub fn pick_dtype(force_f32:bool)->DType{
    if force_f32{
        DType::F32
    }else{
        DType::F16
    }
}