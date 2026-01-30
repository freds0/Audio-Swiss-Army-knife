# Audio Swiss Army Knife (`audio_sak`)

A comprehensive toolkit for audio processing tasks, including conversion, normalization, segmentation, transcription, and embedding extraction.

## Installation

### From Source

Clone the repository and install the package using pip:

```bash
git clone https://github.com/freds0/Audio-Swiss-Army-knife
cd Audio-Swiss-Army-knife
pip install .
```

For development installation (editable mode):

```bash
pip install -e .
```

### Dependencies

The package relies on several external libraries. Key dependencies include:
- `librosa`, `soundfile`, `pydub` for audio manipulation
- `torch`, `torchaudio` for deep learning models
- `transformers`, `openai-whisper` for transcription and embeddings
- `ffmpeg` must be installed on your system (e.g., `sudo apt install ffmpeg` on Ubuntu).

## Usage Guide

The tools are organized into submodules within the `audio_sak` package. You can run them as modules or scripts.

### 1. Conversion

Tools for format conversion and sample rate changes.

**Convert MP3 directory to WAV:**

```bash
python -m audio_sak.conversion.convert_mp3_to_wav -i /path/to/mp3_folder -o /path/to/wav_output
```

**Convert Sample Rate (using SoX):**

```bash
python -m audio_sak.conversion.convert_sample_rate_with_sox -i /input/dir -o /output/dir -s 16000
```
*(Note: Requires `sox` installed on the system)*

**Other conversion tools avaialble:**
- `convert_flac_to_wav`
- `convert_m4a_to_wav`
- `convert_ogg_to_wav`
- `convert_stereo_to_mono`

### 2. Normalization

Normalize audio volume levels.

**Normalize by Mean dBFS:**

Calculates the mean dBFS of a folder and normalizes all files to that level.

```bash
python -m audio_sak.normalization.normalize_audios_by_dbfs -i /input/dir -o /output/dir
```

**Remove Silence (using VAD):**

```bash
python -m audio_sak.normalization.remove_silence_using_vad -i /input/dir -o /output/dir
```

### 3. Segmentation

Split long audio files into smaller segments based on silence or duration.

**Segment Audio:**

Splits audio files in the input directory based on silence detection and merges small segments to fit duration constraints.

```bash
python -m audio_sak.segmentation.segment_audio \
    --input /path/to/wavs \
    --output /path/to/segments \
    --min_duration 2.0 \
    --max_duration 10.0 \
    --threshold_db 40
```

### 4. Transcription

Transcribe audio using state-of-the-art models.

**Transcribe with Whisper:**

Uses OpenAI's Whisper model (via Hugging Face Transformers).

```bash
python -m audio_sak.transcription.transcribe_with_whisper \
    --input_dir /path/to/wavs \
    --output_file transcription.csv
```

**Transcribe with OpenAI API:**

```bash
python -m audio_sak.transcription.transcribe_with_openai --api_key YOUR_KEY ...
```

### 5. Embeddings Extraction

Extract speaker embeddings using various models.

**Extract Wav2Vec Embeddings:**

```bash
python -m audio_sak.embeddings.extract_wav2vec_embeddings \
    --input_dir /path/to/wavs \
    --output_dir /path/to/embeddings
```

**Available Embedding Extractors:**
- `extract_clova_embeddings`
- `extract_ge2e_embeddings`
- `extract_hubert_embeddings`
- `extract_resemblyzer_embeddings`
- `extract_wavlm_embeddings`
- `extract_whisper_embeddings` (Encoder output)

## Project Structure

```
src/audio_sak/
├── conversion/       # Format and sampling rate conversion
├── embeddings/       # Speaker embedding extraction
├── normalization/    # Volume normalization and silence removal
├── segmentation/     # Audio splitting/segmentation
├── transcription/    # ASR (Whisper, Wav2Vec, etc.)
└── utils/            # Helper utilities
```

## License

MIT License
