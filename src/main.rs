use anyhow::Result;
use clap::Parser;
use reazonspeech_rs::ReazonSpeech;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the input WAV file (16kHz, mono recommended)
    #[arg(short, long)]
    input: PathBuf,

    /// Optional path to the model directory. If not provided, models will be downloaded from HF Hub.
    #[arg(short, long)]
    model_dir: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!("Initializing ReazonSpeech (k2-asr)...");
    let start_init = Instant::now();
    let mut model = ReazonSpeech::new(args.model_dir)?;
    let duration_init = start_init.elapsed();
    println!("Initialization took: {:.2}s (Provider: {})", duration_init.as_secs_f32(), model.provider());

    println!("Transcribing: {:?}", args.input);
    let start_transcribe = Instant::now();
    let result = model.transcribe(args.input)?;
    let duration_transcribe = start_transcribe.elapsed();
    let rtf = duration_transcribe.as_secs_f32() / result.audio_duration;

    println!("\nRecognition Result:");
    println!("-------------------");
    println!("{}", result.text);
    println!("-------------------");
    println!("Audio duration:    {:.2}s", result.audio_duration);
    println!("Transcription took: {:.2}s", duration_transcribe.as_secs_f32());
    println!("RTF:                {:.4}", rtf);

    Ok(())
}
