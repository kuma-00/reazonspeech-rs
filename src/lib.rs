mod downloader;

pub use downloader::{Language, Precision};
use downloader::download_models;
use anyhow::Result;
use sherpa_rs::zipformer::{ZipFormer, ZipFormerConfig};
use std::path::PathBuf;

pub struct AudioData {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

pub struct TranscriptionResult {
    pub text: String,
    pub audio_duration: f32,
}

pub struct ReazonSpeech {
    model: ZipFormer,
    provider: String,
}

impl ReazonSpeech {
    pub fn new(
        model_dir: Option<PathBuf>,
        device: Option<String>,
        precision: Option<Precision>,
        language: Option<Language>,
    ) -> Result<Self> {
        let (encoder, decoder, joiner, tokens) = download_models(model_dir, precision, language)?;

        // Validate files exist
        for f in &[&encoder, &decoder, &joiner, &tokens] {
            if !f.exists() {
                anyhow::bail!("Model file not found: {:?}", f);
            }
        }

        let provider = if let Some(d) = device {
            d
        } else if cfg!(target_os = "macos") {
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

    pub fn transcribe(&mut self, audio: AudioData) -> Result<TranscriptionResult> {
        let (mut samples, sample_rate) = (audio.samples, audio.sample_rate);

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
