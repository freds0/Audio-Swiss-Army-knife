#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import subprocess
import logging
from os import makedirs
from os.path import join, basename, splitext
from tqdm import tqdm
from glob import glob

logger = logging.getLogger(__name__)

def convert_wav_to_flac(input_filepath, output_filepath, dry_run=False):
    command = ["flac", input_filepath, "-o", output_filepath, "-f", "--silent"] # -f force overwrite, --silent

    if dry_run:
        logger.info(f"Dry run: {' '.join(command)}")
        return

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error converting {input_filepath}: {e}")
        raise

def process_directory(input_dir, output_dir, dry_run=False):
    makedirs(output_dir, exist_ok=True)
    files = glob(join(input_dir, '*.wav'))
    
    for input_filepath in tqdm(files, desc="Converting WAV to FLAC"):
        output_filepath = join(output_dir, splitext(basename(input_filepath))[0] + '.flac')
        convert_wav_to_flac(input_filepath, output_filepath, dry_run)

def main():
    parser = argparse.ArgumentParser(description="Convert WAV to FLAC.")
    parser.add_argument('-i', '--input', default='input', help='Input folder')
    parser.add_argument('-o', '--output', default='output', help='Output folder')
    parser.add_argument('--dry-run', action='store_true', help='Simulate execution')
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    
    process_directory(args.input, args.output, args.dry_run)

if __name__ == "__main__":
    main()
