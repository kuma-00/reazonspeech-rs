use anyhow::{Context, Result};
use hf_hub::{api::sync::{ApiBuilder}, Cache, Repo, RepoType};
use sherpa_rs::zipformer::{ZipFormer, ZipFormerConfig};
use std::path::PathBuf;

pub struct TranscriptionResult {
    pub text: String,
    pub audio_duration: f32,
}

pub struct ReazonSpeech {
    model: ZipFormer,
    provider: String,
}

impl ReazonSpeech {
    /// Initialize ReazonSpeech with models. 
    /// If `model_dir` is None, it will download models from Hugging Face Hub.
    pub fn new(model_dir: Option<PathBuf>) -> Result<Self> {
        let (encoder, decoder, joiner, tokens) = Self::download_models(model_dir)?;

        // Validate files exist
        for f in &[&encoder, &decoder, &joiner, &tokens] {
            if !f.exists() {
                anyhow::bail!("Model file not found: {:?}", f);
            }
        }

        let provider = if cfg!(target_os = "macos") {
            "coreml".to_string()
        } else {
            "cpu".to_string()
        };

        let provider_name = provider.clone();
        let config = ZipFormerConfig {
            encoder: encoder.to_string_lossy().to_string(),
            decoder: decoder.to_string_lossy().to_string(),
            joiner: joiner.to_string_lossy().to_string(),
            tokens: tokens.to_string_lossy().to_string(),
            num_threads: Some(4),
            provider: Some(provider),
            debug: false,
        };

        let model = ZipFormer::new(config).map_err(|e| anyhow::anyhow!("{}", e))?;
        Ok(Self { model, provider: provider_name })
    }

    pub fn provider(&self) -> &str {
        &self.provider
    }

    fn download_models(cache_dir: Option<PathBuf>) -> Result<(PathBuf, PathBuf, PathBuf, PathBuf)> {
        let repo_id = "reazon-research/reazonspeech-k2-v2".to_string();
        let cache = if let Some(dir) = cache_dir {
            Cache::new(dir)
        } else {
            Cache::default()
        };
        let repo = Repo::new(repo_id.clone(), RepoType::Model);
        let repo_cache = cache.repo(repo);

        let files = [
            "encoder-epoch-99-avg-1.onnx",
            "decoder-epoch-99-avg-1.onnx",
            "joiner-epoch-99-avg-1.onnx",
            "tokens.txt",
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
        let api = ApiBuilder::from_cache(cache).build().context("Failed to build HF Api")?;
        let api_repo = api.model(repo_id);

        println!("Downloading model files from Hugging Face...");
        let encoder = api_repo.get(files[0])?;
        let decoder = api_repo.get(files[1])?;
        let joiner = api_repo.get(files[2])?;
        let tokens = api_repo.get(files[3])?;

        Ok((encoder, decoder, joiner, tokens))
    }

    pub fn transcribe(&mut self, wav_path: PathBuf) -> Result<TranscriptionResult> {
        let (mut samples, sample_rate) = sherpa_rs::read_audio_file(&wav_path.to_string_lossy())
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        // 1. Duration check
        let duration = samples.len() as f32 / sample_rate as f32;
        const TOO_LONG_SECONDS: f32 = 30.0;
        if duration > TOO_LONG_SECONDS {
            eprintln!(
                "Warning: Passing a long audio input ({:.1}s) is not recommended, \
                 because K2 will require a large amount of memory. \
                 Read the upstream discussion for more details: \
                 https://github.com/k2-fsa/icefall/issues/1680",
                duration
            );
        }

        // 2. Padding (pad_audio equivalent)
        // Python: pad_audio(audio, 0.9) -> np.pad(..., pad_width=int(0.9 * sr), mode='constant')
        const PAD_SECONDS: f32 = 0.9;
        let pad_samples = (PAD_SECONDS * sample_rate as f32) as usize;
        let padding = vec![0.0f32; pad_samples];

        // Pad both sides
        let mut padded_samples = Vec::with_capacity(padding.len() + samples.len() + padding.len());
        padded_samples.extend_from_slice(&padding);
        padded_samples.append(&mut samples);
        padded_samples.extend_from_slice(&padding);

        let text = self.model.decode(sample_rate, padded_samples);
        Ok(TranscriptionResult {
            text,
            audio_duration: duration,
        })
    }
}
