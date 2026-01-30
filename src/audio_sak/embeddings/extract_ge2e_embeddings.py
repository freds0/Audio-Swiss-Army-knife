#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import sys
import torch
import logging
from pathlib import Path
from os.path import join, exists, basename
from os import makedirs
from tqdm import tqdm
from glob import glob
import numpy as np

# Handle local imports safely or assume submodule availability
# But wait, `sys.path.append('Real-Time-Voice-Cloning')` assumes folder exists.
# We should probably trust the existing logic but wrap imports.
try:
    sys.path.append('Real-Time-Voice-Cloning')
    from encoder.model import SpeakerEncoder
    from encoder.inference import compute_partial_slices
    from encoder.audio import normalize_volume, preprocess_wav, wav_to_mel_spectrogram
except ImportError:
    SpeakerEncoder = None
    compute_partial_slices = None
    normalize_volume = None
    preprocess_wav = None
    wav_to_mel_spectrogram = None

logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Mel-filterbank
mel_window_length = 25
mel_window_step = 10
mel_n_channels = 40

## Audio
sampling_rate = 16000
partials_n_frames = 160
inference_n_frames = 80
audio_norm_target_dBFS = -30

def load_model(model_path):
    if SpeakerEncoder is None:
        logger.error("Real-Time-Voice-Cloning submodule missing.")
        return None
        
    try:
        weights_fpath = Path(model_path)
        model = SpeakerEncoder(device, torch.device("cpu"))
        checkpoint = torch.load(weights_fpath, device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Failed to load model {model_path}: {e}")
        return None

def embed_frames_batch(model, frames_batch):
    if model is None:
        raise Exception("Model was not loaded. Call load_model() before inference.")

    frames = torch.from_numpy(frames_batch).to(device)
    embed = model.forward(frames).detach().cpu().numpy()
    return embed

def extract_embed_utterance(model, wav, using_partials=True, return_partials=False, **kwargs):
    if not using_partials:
        frames = wav_to_mel_spectrogram(wav)
        embed = embed_frames_batch(model, frames[None, ...])[0]
        if return_partials:
            return embed, None, None
        return embed

    wave_slices, mel_slices = compute_partial_slices(len(wav), **kwargs)
    max_wave_length = wave_slices[-1].stop
    if max_wave_length >= len(wav):
        wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")

    frames = wav_to_mel_spectrogram(wav)
    frames_batch = np.array([frames[s] for s in mel_slices])
    partial_embeds = embed_frames_batch(model, frames_batch)

    raw_embed = np.mean(partial_embeds, axis=0)
    embed = raw_embed / np.linalg.norm(raw_embed, 2)

    if return_partials:
        return embed, partial_embeds, wave_slices
    return embed


def extract_ge2e_embeddings(filelist, model_path, output_dir):
    model = load_model(model_path)
    if model is None:
        return

    os.makedirs(output_dir, exist_ok=True)
    
    for filepath in tqdm(filelist, desc="Extracting"):
        if not exists(filepath):
            logger.warning("file {} doesnt exist!".format(filepath))
            continue
        try:
            filename = basename(filepath)
            # Extract Embedding
            preprocessed_wav = preprocess_wav(filepath)
            file_embedding = extract_embed_utterance(model, preprocessed_wav)
            embedding = torch.tensor(file_embedding.reshape(-1).tolist())
            # Saving embedding
            output_filename = filename.split(".")[0] + ".pt"
            output_filepath = join(output_dir, output_filename)
            torch.save(embedding, output_filepath)
        except Exception as e:
            logger.error(f"Failed to process {filepath}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Extract GE2E embeddings.")
    parser.add_argument('-b', '--base_dir', default='./')
    parser.add_argument('-i', '--input_dir', help='Wavs folder')
    parser.add_argument('-c', '--input_csv', help='Metadata filepath')
    parser.add_argument('--model_path', default='./checkpoints/ge2e/pretrained.pt', help='Model .pth filepath')
    parser.add_argument('-o', '--output_dir', default='output_embeddings', help='Output folder')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    output_dir = join(args.base_dir, args.output_dir)

    filelist = []
    if (args.input_dir is not None):
        input_dir = join(args.base_dir, args.input_dir)
        filelist = glob(input_dir + '/*.wav')

    elif (args.input_csv is not None):
        with open(join(args.base_dir, args.input_csv), encoding="utf-8") as f:
            content_file = f.readlines()
            filelist = [line.split(",")[0].strip() for line in content_file if line.strip()]
    else:
        logger.error("Error: args input_dir or input_csv are necessary!")
        return

    extract_ge2e_embeddings(filelist, args.model_path, output_dir)


if __name__ == "__main__":
    main()
