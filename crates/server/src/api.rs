use serde::{Serialize, Deserialize};
#[derive(Debug, Deserialize)]

pub struct CompletionRequest{
    pub prompt:String,
    #[serde(default="default_max_tokens")]
    pub max_tokens:usize,
    #[serde(default)]
    pub temperature:f32,
    #[serde(default)]
    pub top_k:Option<usize>,
    #[serde(default)]
    pub top_p:Option<f32>,
    #[serde(default)]
    pub min_prob:Option<f32>,
    #[serde(default)]
    pub repeat_penalty:Option<f32>,
    #[serde(default)]
    pub seed:Option<u64>,
    #[serde(default)]
    pub stop:Vec<String>,
    #[serde(default)]
    pub stream:bool,
}

fn default_max_tokens()->usize{
    128
}


#[derive(Debug, Serialize)]
pub struct CompletionResponse{
    pub text:String,
    pub finish_reason:String,
    pub usage:Usage,
}

#[derive(Debug, Serialize)]
pub struct Usage{
    pub prompt_tokens:usize,
    pub completion_tokens:usize,
}
