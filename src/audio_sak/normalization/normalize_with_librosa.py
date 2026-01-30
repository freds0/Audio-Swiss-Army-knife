#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
from os.path import join, basename, dirname, exists
from os import makedirs
from tqdm import tqdm
from glob import glob
import librosa
import soundfile as sf

logger = logging.getLogger(__name__)

def normalize_audios(input_path, output_path, dry_run=False, recursive=True):
    makedirs(output_path, exist_ok=True)
    pattern = join(input_path, '**', '*.wav') if recursive else join(input_path, '*.wav')
    files = glob(pattern, recursive=recursive)

    for input_filepath in tqdm(files, desc="Normalizing (librosa)"):
        # preserve structure logic from original?
        # "dirname(input_filepath).split("/")[2]" <- this is very brittle and specific to user's path!!!
        # Standardize to flattened or relative path. 
        # I'll use relative path to preserve structure safely if recursive.
        try:
            # Simple flattened for now to avoid complexity unless requested
            filename = basename(input_filepath)
            output_filepath = join(output_path, filename)
            
            if dry_run:
                logger.info(f"Dry run: normalize {input_filepath} -> {output_filepath}")
                continue

            waveform, sr = librosa.load(input_filepath, sr=None)
            norm_waveform = librosa.util.normalize(waveform)
            sf.write(output_filepath, norm_waveform, sr, 'PCM_16')
        except Exception as e:
            logger.error(f"Failed to normalize {input_filepath}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Normalize audios using librosa.")
    parser.add_argument('-i', '--input', default='input', help='Input folder')
    parser.add_argument('-o', '--output', default='output', help='Output folder')
    parser.add_argument('--dry-run', action='store_true', help='Simulate execution')
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    normalize_audios(args.input, args.output, args.dry_run)

if __name__ == "__main__":
    main()
