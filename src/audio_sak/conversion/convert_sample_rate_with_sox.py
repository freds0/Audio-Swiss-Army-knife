#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import subprocess
import logging
from glob import glob
from os import makedirs
from os.path import join, basename, splitext
from tqdm import tqdm

logger = logging.getLogger(__name__)

NUMBER_BITS = 16
ENCODING = "signed-integer"
NUMBER_CHANNELS = 1

def convert_file(input_filepath, output_filepath, sr=44100, dry_run=False):
    """
    Resample audio using SoX.
    """
    command = [
        "sox", input_filepath,
        "-V0", # Quiet mode
        "-c", str(NUMBER_CHANNELS),
        "-r", str(sr),
        "-b", str(NUMBER_BITS),
        "-e", ENCODING,
        output_filepath
    ]

    if dry_run:
        logger.info(f"Dry run: {' '.join(command)}")
        return

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error converting {input_filepath} with sox: {e}")
        raise

def process_directory(input_dir, output_dir, sr=44100, dry_run=False):
    makedirs(output_dir, exist_ok=True)
    files = glob(join(input_dir, '*.wav'))
    
    for input_filepath in tqdm(files, desc="Resampling with SoX"):
        output_filepath = join(output_dir, basename(input_filepath))
        convert_file(input_filepath, output_filepath, sr, dry_run)

def main():
    parser = argparse.ArgumentParser(description="Resample audio using SoX.")
    parser.add_argument('-i', '--input', default='input', help='Input folder')
    parser.add_argument('-o', '--output', default='output', help='Output folder')
    parser.add_argument('-s', '--sr', default=44100, type=int, help='Target sampling rate')
    parser.add_argument('--dry-run', action='store_true', help='Simulate execution')
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    
    process_directory(args.input, args.output, args.sr, args.dry_run)

if __name__ == "__main__":
    main()
