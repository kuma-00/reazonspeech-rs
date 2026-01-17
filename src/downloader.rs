use anyhow::{Context, Result};
use hf_hub::{api::sync::ApiBuilder, Cache, Repo, RepoType};
use std::path::PathBuf;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Precision {
    #[default]
    Fp32,
    Int8,
    Int8Fp32,
}

impl FromStr for Precision {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "fp32" => Ok(Precision::Fp32),
            "int8" => Ok(Precision::Int8),
            "int8-fp32" => Ok(Precision::Int8Fp32),
            _ => anyhow::bail!("Unknown precision: {}", s),
        }
    }
}

impl Precision {
    pub fn as_str(&self) -> &str {
        match self {
            Precision::Fp32 => "fp32",
            Precision::Int8 => "int8",
            Precision::Int8Fp32 => "int8-fp32",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Language {
    #[default]
    Ja,
    JaEn,
    JaEnMls5k,
}

impl FromStr for Language {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "ja" => Ok(Language::Ja),
            "ja-en" => Ok(Language::JaEn),
            "ja-en-mls-5k" => Ok(Language::JaEnMls5k),
            _ => anyhow::bail!("Unknown language: {}", s),
        }
    }
}

impl Language {
    pub fn as_str(&self) -> &str {
        match self {
            Language::Ja => "ja",
            Language::JaEn => "ja-en",
            Language::JaEnMls5k => "ja-en-mls-5k",
        }
    }
}

pub fn download_models(
    cache_dir: Option<PathBuf>,
    precision: Option<Precision>,
    language: Option<Language>,
) -> Result<(PathBuf, PathBuf, PathBuf, PathBuf)> {
    let precision = precision.unwrap_or_default();
    let language = language.unwrap_or_default();

    let (repo_id, epochs) = match language {
        Language::Ja => ("reazon-research/reazonspeech-k2-v2".to_string(), 99),
        Language::JaEn => ("reazon-research/reazonspeech-k2-v2-ja-en".to_string(), 35),
        Language::JaEnMls5k => {
            ("reazon-research/reazonspeech-k2-v2-ja-en-mls-5k-corrected".to_string(), 21)
        }
    };

    let cache = if let Some(dir) = cache_dir {
        Cache::new(dir)
    } else {
        Cache::default()
    };
    let repo = Repo::new(repo_id.clone(), RepoType::Model);
    let repo_cache = cache.repo(repo);

    let encoder_file = match precision {
        Precision::Fp32 => format!("encoder-epoch-{}-avg-1.onnx", epochs),
        Precision::Int8 | Precision::Int8Fp32 => format!("encoder-epoch-{}-avg-1.int8.onnx", epochs),
    };
    let decoder_file = match precision {
        Precision::Fp32 | Precision::Int8Fp32 => format!("decoder-epoch-{}-avg-1.onnx", epochs),
        Precision::Int8 => format!("decoder-epoch-{}-avg-1.int8.onnx", epochs),
    };
    let joiner_file = match precision {
        Precision::Fp32 => format!("joiner-epoch-{}-avg-1.onnx", epochs),
        Precision::Int8 | Precision::Int8Fp32 => format!("joiner-epoch-{}-avg-1.int8.onnx", epochs),
    };
    let tokens_file = "tokens.txt".to_string();

    let files = [
        encoder_file,
        decoder_file,
        joiner_file,
        tokens_file,
    ];

    // Check if all files exist in cache
    let mut paths = Vec::new();
    for file in &files {
        if let Some(path) = repo_cache.get(file) {
            paths.push(path);
        }
    }

    if paths.len() == 4 {
        return Ok((
            paths[0].clone(),
            paths[1].clone(),
            paths[2].clone(),
            paths[3].clone(),
        ));
    }

    // Otherwise, use Api to download
    let api = ApiBuilder::from_cache(cache)
        .build()
        .context("Failed to build HF Api")?;
    let api_repo = api.model(repo_id);

    println!("Downloading model files from Hugging Face...");
    let encoder = api_repo.get(&files[0])?;
    let decoder = api_repo.get(&files[1])?;
    let joiner = api_repo.get(&files[2])?;
    let tokens = api_repo.get(&files[3])?;

    Ok((encoder, decoder, joiner, tokens))
}
