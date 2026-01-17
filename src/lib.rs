use anyhow::{Result, Context};
use hf_hub::api::sync::Api;
use sherpa_rs::zipformer::{ZipFormer, ZipFormerConfig};
use std::path::PathBuf;

pub struct ReazonSpeech {
    model: ZipFormer,
}

impl ReazonSpeech {
    /// Initialize ReazonSpeech with models. 
    /// If `model_dir` is None, it will download models from Hugging Face Hub.
    pub fn new(model_dir: Option<PathBuf>) -> Result<Self> {
        let (encoder, decoder, joiner, tokens) = if let Some(dir) = model_dir {
            (
                dir.join("encoder-epoch-99-avg-1.onnx"),
                dir.join("decoder-epoch-99-avg-1.onnx"),
                dir.join("joiner-epoch-99-avg-1.onnx"),
                dir.join("tokens.txt"),
            )
        } else {
            Self::download_models()?
        };

        // Validate files exist
        for f in &[&encoder, &decoder, &joiner, &tokens] {
            if !f.exists() {
                anyhow::bail!("Model file not found: {:?}", f);
            }
        }

        let config = ZipFormerConfig {
            encoder: encoder.to_string_lossy().to_string(),
            decoder: decoder.to_string_lossy().to_string(),
            joiner: joiner.to_string_lossy().to_string(),
            tokens: tokens.to_string_lossy().to_string(),
            num_threads: Some(4),
            provider: Some("cpu".to_string()),
            debug: false,
        };

        let model = ZipFormer::new(config).map_err(|e| anyhow::anyhow!("{}", e))?;
        Ok(Self { model })
    }

    fn download_models() -> Result<(PathBuf, PathBuf, PathBuf, PathBuf)> {
        let api = Api::new().context("Failed to create HF Api")?;
        let repo = api.model("reazon-research/reazonspeech-k2-v2".to_string());

        println!("Downloading model files from Hugging Face...");
        let encoder = repo.get("encoder-epoch-99-avg-1.onnx")?;
        let decoder = repo.get("decoder-epoch-99-avg-1.onnx")?;
        let joiner = repo.get("joiner-epoch-99-avg-1.onnx")?;
        let tokens = repo.get("tokens.txt")?;

        Ok((encoder, decoder, joiner, tokens))
    }

    pub fn transcribe(&mut self, wav_path: PathBuf) -> Result<String> {
        let (samples, sample_rate) = sherpa_rs::read_audio_file(&wav_path.to_string_lossy())
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        let text = self.model.decode(sample_rate, samples);
        Ok(text)
    }
}
