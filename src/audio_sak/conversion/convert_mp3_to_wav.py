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

def convert_file(input_filepath, output_filepath, target_sr=44100, dry_run=False):
    """
    Convert an MP3 file to WAV format using ffmpeg.

    Args:
        input_filepath (str): Path to the input MP3 file.
        output_filepath (str): Path to the output WAV file.
        target_sr (int): Target sampling rate. Defaults to 44100.
        dry_run (bool): If True, only logs the command without executing.
    """
    command = [
        "ffmpeg",
        "-i", input_filepath,
        "-ar", str(target_sr),
        "-y",  # Overwrite output files
        output_filepath
    ]
    
    if dry_run:
        logger.info(f"Dry run: {' '.join(command)}")
        return

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error converting {input_filepath}: {e}")
        raise

def convert_directory(input_dir, output_dir, target_sr=44100, dry_run=False):
    """
    Batch convert all MP3 files in a directory to WAV.
    """
    makedirs(output_dir, exist_ok=True)
    files = glob(join(input_dir, '*.mp3'))
    
    for input_filepath in tqdm(files, desc="Converting MP3 to WAV"):
        output_filepath = join(output_dir, splitext(basename(input_filepath))[0] + '.wav')
        convert_file(input_filepath, output_filepath, target_sr, dry_run)

def main():
    parser = argparse.ArgumentParser(description="Convert MP3 files to WAV.")
    parser.add_argument('-i', '--input', default='input', help='Input folder')
    parser.add_argument('-o', '--output', default='output', help='Output folder')
    parser.add_argument('-s', '--sr', default=44100, type=int, help='Target sampling rate')
    parser.add_argument('--dry-run', action='store_true', help='Simulate conversion without executing')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    convert_directory(args.input, args.output, args.sr, args.dry_run)

if __name__ == "__main__":
    main()
