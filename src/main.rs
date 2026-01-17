use anyhow::Result;
use clap::Parser;
use reazonspeech_rs::ReazonSpeech;
use std::path::PathBuf;

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
    let mut model = ReazonSpeech::new(args.model_dir)?;

    println!("Transcribing: {:?}", args.input);
    let text = model.transcribe(args.input)?;

    println!("\nRecognition Result:");
    println!("-------------------");
    println!("{}", text);
    println!("-------------------");

    Ok(())
}
