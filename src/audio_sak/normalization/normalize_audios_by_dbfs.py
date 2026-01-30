#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
import numpy as np
from os import makedirs
from os.path import join, basename, splitext
from tqdm import tqdm
from glob import glob
from pydub import AudioSegment

logger = logging.getLogger(__name__)

def calculate_mean_dbfs(input_path, recursive=True):
    """
    Calculate the mean dBFS of all WAV files in a directory.
    """
    dbfs_list = []
    pattern = join(input_path, '**', '*.wav') if recursive else join(input_path, '*.wav')
    files = glob(pattern, recursive=recursive)
    
    if not files:
        logger.warning(f"No WAV files found in {input_path}")
        return -20.0 # Default fallback

    logger.info(f"Calculating mean dBFS from {len(files)} files...")
    for audio_file in tqdm(files, desc="Scanning dBFS"):
        try:
            dbfs_list.append(AudioSegment.from_file(audio_file).dBFS)
        except Exception as e:
            logger.error(f"Error reading {audio_file}: {e}")

    if not dbfs_list:
        return -20.0

    return np.mean(dbfs_list)

def normalize_audios(input_path, output_path, target_dbfs, recursive=True, dry_run=False):
    """
    Normalize all WAV files in input_path to target_dbfs.
    """
    makedirs(output_path, exist_ok=True)
    pattern = join(input_path, '**', '*.wav') if recursive else join(input_path, '*.wav')
    files = glob(pattern, recursive=recursive)

    for audio_file in tqdm(files, desc="Normalizing"):
        try:
            filename = basename(audio_file)
            # If recursive, preserve structure? Original script didn't seem to preserve structure fully, 
            # it just dumped basename into output_dir. I'll stick to that simple behavior unless structure is important.
            # But if recursive scan, flattened output causes collisions.
            # I'll stick to flattened output to match original behavior but warn.
            dest_file = join(output_path, filename)

            if dry_run:
                logger.info(f"Dry run: normalize {audio_file} to {target_dbfs} dBFS -> {dest_file}")
                continue

            sound = AudioSegment.from_file(audio_file)
            change_in_dBFS = target_dbfs - sound.dBFS
            normalized_sound = sound.apply_gain(change_in_dBFS)
            normalized_sound.export(dest_file, format="wav")
        except Exception as e:
            logger.error(f"Failed to normalize {audio_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Normalize audios by mean dBFS.")
    parser.add_argument('-i', '--input', default='input', help='Input folder')
    parser.add_argument('-o', '--output', default='output', help='Output folder')
    parser.add_argument('--dry-run', action='store_true', help='Simulate execution')
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    print('Calculating mean dBfs...')
    mean_dbfs = calculate_mean_dbfs(args.input)
    print("Mean DBFS:", mean_dbfs)
    print('Normalizing audios...')
    normalize_audios(args.input, args.output, mean_dbfs, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
