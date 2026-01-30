#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
import torch
import torchaudio
from glob import glob
from os import makedirs
from os.path import join, basename, splitext
from tqdm import tqdm

logger = logging.getLogger(__name__)

def convert_stereo_to_mono(input_filepath, output_filepath, dry_run=False):
    """
    Convert stereo audio to mono by averaging channels.
    """
    if dry_run:
        logger.info(f"Dry run: convert {input_filepath} to mono -> {output_filepath}")
        return

    try:
        waveform, sr = torchaudio.load(input_filepath)
        # Verify shape
        if waveform.dim() == 2 and waveform.shape[0] > 1:
            # Average across channels (dim 0)
            mono_waveform = torch.mean(waveform, dim=0, keepdim=True)
            torchaudio.save(output_filepath, mono_waveform, sr, encoding="PCM_S", bits_per_sample=16)
        else:
            # Already mono or unexpected shape, just copy/save
            logger.info(f"{input_filepath} is already mono or has strange shape {waveform.shape}. Copying to {output_filepath}")
            torchaudio.save(output_filepath, waveform, sr, encoding="PCM_S", bits_per_sample=16)
            
    except Exception as e:
        logger.error(f"Failed to convert {input_filepath}: {e}")
        raise

def process_directory(input_dir, output_dir, dry_run=False):
    makedirs(output_dir, exist_ok=True)
    files = glob(join(input_dir, '*.wav'))
    
    for input_filepath in tqdm(files, desc="Converting Stereo to Mono"):
        output_filepath = join(output_dir, basename(input_filepath))
        convert_stereo_to_mono(input_filepath, output_filepath, dry_run)

def main():
    parser = argparse.ArgumentParser(description="Convert WAV files from stereo to mono.")
    parser.add_argument('-i', '--input', default='input', help='Input folder')
    parser.add_argument('-o', '--output', default='output', help='Output folder')
    parser.add_argument('--dry-run', action='store_true', help='Simulate execution')
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    
    process_directory(args.input, args.output, args.dry_run)

if __name__ == "__main__":
    main()
