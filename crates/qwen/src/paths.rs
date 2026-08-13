//! Resolving files out of the local HuggingFace cache.
//!
//! Pure plumbing, deliberately written for you so Day 1 starts on the model
//! rather than on directory walking. Nothing here is load-bearing for learning.

use anyhow::{anyhow, Context, Result};
use std::path::{Path, PathBuf};
/// The model this project targets: Qwen2 architecture, 1.5B, already cached.
pub const MODEL_REPO: &str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B";

/// A resolved model snapshot on disk.
#[derive(Debug, Clone)]
pub struct ModelPaths {
    pub root: PathBuf,
    pub config: PathBuf,
    pub tokenizer: PathBuf,
    pub tokenizer_config: PathBuf,
    /// One or more safetensors shards. This checkpoint is a single 3.3 GB file,
    /// but `VarBuilder::from_mmaped_safetensors` takes a slice, so keep it a Vec.
    pub weights: Vec<PathBuf>,
}

fn hub_root() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("HF_HUB_CACHE") {
        return Ok(PathBuf::from(p));
    }
    if let Ok(p) = std::env::var("HF_HOME") {
        return Ok(PathBuf::from(p).join("hub"));
    }
    let home = std::env::var("HOME").context("HOME is not set")?;
    Ok(PathBuf::from(home).join(".cache/huggingface/hub"))
}

/// Locate a cached repo's snapshot directory.
///
/// The hub layout is `models--{org}--{name}/snapshots/{commit}/`, where the
/// files are symlinks into `blobs/`. `refs/main` names the current commit; we
/// fall back to whatever single snapshot exists if it is missing.
pub fn snapshot_dir(repo: &str) -> Result<PathBuf> {
    let repo_dir = hub_root()?.join(format!("models--{}", repo.replace('/', "--")));
    if !repo_dir.is_dir() {
        return Err(anyhow!(
            "{repo} is not in the HF cache at {}",
            repo_dir.display()
        ));
    }

    let snapshots = repo_dir.join("snapshots");
    if let Ok(commit) = std::fs::read_to_string(repo_dir.join("refs/main")) {
        let dir = snapshots.join(commit.trim());
        if dir.is_dir() {
            return Ok(dir);
        }
    }

    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&snapshots)
        .with_context(|| format!("reading {}", snapshots.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    dirs.pop()
        .ok_or_else(|| anyhow!("no snapshots under {}", snapshots.display()))
}

fn require(dir: &Path, name: &str) -> Result<PathBuf> {
    let p = dir.join(name);
    if !p.exists() {
        return Err(anyhow!("missing {name} in {}", dir.display()));
    }
    Ok(p)
}

impl ModelPaths {
    /// Resolve the default target model from the HF cache.
    pub fn from_cache() -> Result<Self> {
        Self::from_repo(MODEL_REPO)
    }

    pub fn from_repo(repo: &str) -> Result<Self> {
        Self::from_dir(&snapshot_dir(repo)?)
    }

    /// Resolve from an explicit directory, for a model you downloaded yourself.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let mut weights: Vec<PathBuf> = std::fs::read_dir(dir)
            .with_context(|| format!("reading {}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
            .collect();
        weights.sort();
        if weights.is_empty() {
            return Err(anyhow!("no .safetensors files in {}", dir.display()));
        }

        Ok(Self {
            config: require(dir, "config.json")?,
            tokenizer: require(dir, "tokenizer.json")?,
            tokenizer_config: require(dir, "tokenizer_config.json")?,
            weights,
            root: dir.to_path_buf(),
        })
    }

    /// Total size of the weight shards, in bytes.
    ///
    /// You need this on Day 3 to size the paged-KV block pool: pool budget is
    /// (memory budget - weights - activation headroom).
    pub fn weights_bytes(&self) -> Result<u64> {
        self.weights
            .iter()
            .map(|p| Ok(std::fs::metadata(p)?.len()))
            .sum()
    }
}
