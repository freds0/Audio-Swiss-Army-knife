#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
import torchaudio
import librosa
from glob import glob
from os import makedirs
from os.path import join, basename, exists, splitext
from tqdm import tqdm

logger = logging.getLogger(__name__)

def remove_silence(input_filepath, output_filepath, top_db=40, dry_run=False):
    if dry_run:
        logger.info(f"Dry run: trim silence {input_filepath} (top_db={top_db}) -> {output_filepath}")
        return

    try:
        waveform, sr = torchaudio.load(input_filepath)
        # librosa expects numpy, torchaudio returns tensor
        waveform_np = waveform.numpy().flatten()
        
        # librosa.effects.trim returns (trimmed_y, index)
        # trim expects mono or stereo numpy. shape (..., n_samples)
        # torchaudio loads as (channels, samples). Flattening might merge channels if stereo?
        # If stereo, trim works on audio signal.
        # Original script: trimmed_waveform, index = librosa.effects.trim(waveform, top_db=top_db)
        # Warning: librosa might expect numpy array, passing tensor might fail or work if coerced.
        # But `torchaudio` loads tensor. `librosa` functions often convert to numpy.
        
        # Better:
        effects = [['silence', '1', '0.1', '{}%'.format(top_db), '1', '0.1', '{}%'.format(top_db)]]
        # Wait, original script used `librosa.effects.trim(waveform, top_db=top_db)`.
        # `waveform` from torchaudio is Tensor. Librosa usually works on numpy.
        # I should convert to numpy.
        
        waveform_np = waveform.numpy()
        trimmed, index = librosa.effects.trim(waveform_np, top_db=top_db)
        # trimmed is numpy.
        
        # Save using torchaudio? torchaudio.save expects Tensor.
        # Convert back.
        import torch
        trimmed_tensor = torch.from_numpy(trimmed)
        
        torchaudio.save(output_filepath, trimmed_tensor, sr, encoding="PCM_S", bits_per_sample=16)

    except Exception as e:
        logger.error(f"Failed to trim {input_filepath}: {e}")
        raise

def process_directory(input_dir, output_dir, top_db=40, dry_run=False):
    makedirs(output_dir, exist_ok=True)
    files = sorted(glob(join(input_dir, "*.wav")))
    
    for input_filepath in tqdm(files, desc="Trimming Silence"):
        output_filepath = join(output_dir, basename(input_filepath))
        remove_silence(input_filepath, output_filepath, top_db, dry_run)

def main():
    parser = argparse.ArgumentParser(description="Remove silence using librosa.")
    parser.add_argument('-i', '--input', default='input', help='Input folder')
    parser.add_argument('-o', '--output', default='output', help='Output folder')
    parser.add_argument('-t', '--top_db', default=40, type=float, help='Threshold db')
    parser.add_argument('--dry-run', action='store_true', help='Simulate execution')
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    
    process_directory(args.input, args.output, args.top_db, args.dry_run)

if __name__ == "__main__":
    main()
