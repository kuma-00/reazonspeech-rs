# ReazonSpeech Rust (reazonspeech-rs)

ReazonSpeechのk2-asr (Zipformer) モデルをRustに移植した実装です。
Hugging Face Hubからのモデル自動ダウンロード機能を備えており、マルチプラットフォームで動作します。

## 特徴

- **高速かつ軽量**: `sherpa-onnx` を使用した効率的な推論。macOSでは **CoreML** を自動的に使用し、Apple Neural Engineによる高速化が行われます。その他OSでは **GPU** を使用して高速に推論できます。
- **モデル自動ダウンロード**: 初回実行時に `reazon-research/reazonspeech-k2-v2` モデルをHugging Faceから自動取得します。
- **ポータブル**: macOS、Linuxでの動作を確認済み。
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
- `-m, --model-dir <PATH>`: ローカルのモデルディレクトリ (オプション)。
- `-d, --device <STR>`: 推論デバイス (`cpu`, `cuda`, `coreml`)。macOSではデフォルトで `coreml`、その他は `cpu`。
- `-p, --precision <STR>`: 推論精度 (`fp32`, `int8`, `int8-fp32`)。デフォルトは `fp32`。
- `-l, --language <STR>`: 言語モデル (`ja`, `ja-en`, `ja-en-mls-5k`)。デフォルトは `ja`。

## ライセンス・引用

本プロジェクトは [ReazonSpeech](https://github.com/reazon-research/ReazonSpeech) のモデルを使用しています。
モデルおよびオリジナルの実装に関する著作権は株式会社レアゾン・ホールディングスに帰属します。

- [ReazonSpeech プロジェクト](https://research.reazon.jp/)
- [Hugging Face: reazonspeech-k2-v2](https://huggingface.co/reazon-research/reazonspeech-k2-v2)

## ライセンス

このプロジェクト自体は **Apache License 2.0** の下で公開されています。
詳細は [LICENSE](LICENSE) ファイルを参照してください。

また、本プロジェクトが配布・使用する **ReazonSpeech モデル** も **Apache License 2.0** の下で提供されています。
その他の依存ライブラリ：
- `sherpa-onnx`: Apache-2.0
- `sherpa-rs`: MIT
