#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
from glob import glob
from os import makedirs
from os.path import join, basename, splitext
from tqdm import tqdm
import librosa
import soundfile as sf

logger = logging.getLogger(__name__)

def convert_file(input_filepath, output_filepath, target_sr=44100, dry_run=False, allow_upsample=False):
    """
    Resample a WAV file using librosa.

    Args:
        input_filepath (str): Path to input file.
        output_filepath (str): Path to output file.
        target_sr (int): Target sampling rate.
        dry_run (bool): If True, simulate execution.
        allow_upsample (bool): If True, allow resampling to a higher rate.
    """
    try:
        data, orig_sr = librosa.load(input_filepath, sr=None)
        orig_sr = int(orig_sr)
    except Exception as e:
        logger.error(f"Failed to load {input_filepath}: {e}")
        return

    if orig_sr == target_sr:
        logger.info(f"Skipping {input_filepath}: Already at {orig_sr}Hz")
        return

    if orig_sr < target_sr and not allow_upsample:
        logger.warning(f"Skipping {input_filepath}: Original SR {orig_sr} < Target {target_sr} (Upsampling disabled)")
        return

    if dry_run:
        logger.info(f"Dry run: resample {input_filepath} ({orig_sr}Hz) -> {output_filepath} ({target_sr}Hz)")
        return

    try:
        converted_data = librosa.resample(y=data, orig_sr=orig_sr, target_sr=target_sr)
        sf.write(output_filepath, converted_data, target_sr)
    except Exception as e:
        logger.error(f"Failed to write {output_filepath}: {e}")
        raise

def convert_directory(input_dir, output_dir, target_sr=44100, dry_run=False, allow_upsample=False):
    """
    Batch resample all WAV files in a directory.
    """
    makedirs(output_dir, exist_ok=True)
    files = glob(join(input_dir, '*.wav'))
    
    for input_filepath in tqdm(files, desc="Resampling with librosa"):
        output_filepath = join(output_dir, basename(input_filepath))
        convert_file(input_filepath, output_filepath, target_sr, dry_run, allow_upsample)

def main():
    parser = argparse.ArgumentParser(description="Resample WAV files using librosa.")
    parser.add_argument('-i', '--input', default='input', help='Input folder')
    parser.add_argument('-o', '--output', default='output', help='Output folder')
    parser.add_argument('-s', '--sr', default=44100, type=int, help='Target sampling rate')
    parser.add_argument('--dry-run', action='store_true', help='Simulate conversion')
    parser.add_argument('--upsample', action='store_true', help='Allow upsampling')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    convert_directory(args.input, args.output, args.sr, args.dry_run, args.upsample)

if __name__ == "__main__":
    main()
