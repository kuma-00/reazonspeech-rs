# ReazonSpeech Rust (reazonspeech-rs)

ReazonSpeechのk2-asr (Zipformer) モデルをRustに移植した実装です。
Hugging Face Hubからのモデル自動ダウンロード機能を備えており、Mac M4 (Apple Silicon) を含むマルチプラットフォームで動作します。

## 特徴

- **高速かつ軽量**: `sherpa-onnx` (Rustバインディング `sherpa-rs`) を使用した効率的な推論。
- **モデル自動ダウンロード**: 初回実行時に `reazon-research/reazonspeech-k2-v2` モデルをHugging Faceから自動取得。
- **ポータブル**: Mac M4 (ARM64) および Linux での動作を確認済み。
- **シンプル**: 音声ファイルを指定するだけで日本語のテキスト書き起こしが可能。

## セットアップ

### 必須環境
ビルドには `clang` (`libclang`) が必要です。

#### macOS (Homebrew)
```bash
brew install llvm
export LIBCLANG_PATH="$(brew --prefix llvm)/lib"
```

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y libclang-dev clang
```

### インストール
リポジトリをクローンしてビルドします。
```bash
cargo build --release
```

## 使い方

書き起こしたい音声ファイル (16kHz WAV) を指定して実行します。

```bash
cargo run --release -- --input path/to/your_audio.wav
```

### コマンドライン引数
- `-i, --input <PATH>`: 入力WAVファイル (必須)。
- `-m, --model-dir <PATH>`: ローカルのモデルディレクトリ (オプション。指定しない場合はHFから自動ダウンロード)。

## ライセンス・引用

本プロジェクトは [ReazonSpeech](https://github.com/reazon-research/ReazonSpeech) のモデルを使用しています。
モデルおよびオリジナルの実装に関する著作権は株式会社レアゾン・ホールディングスに帰属します。

- [ReazonSpeech プロジェクト](https://reazon-research.github.io/ReazonSpeech/)
- [Hugging Face: reazonspeech-k2-v2](https://huggingface.co/reazon-research/reazonspeech-k2-v2)
