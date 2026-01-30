#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pydub
import argparse
import subprocess
import logging
from os import makedirs
from os.path import join, basename, splitext
from tqdm import tqdm
from glob import glob

logger = logging.getLogger(__name__)

def convert_file(input_filepath, output_filepath, tool='pydub', dry_run=False):
    """
    Convert a WAV file to MP3 format using pydub or lame.

    Args:
        input_filepath (str): Path to input WAV file.
        output_filepath (str): Path to output MP3 file.
        tool (str): 'pydub' or 'lame'. Defaults to 'pydub'.
        dry_run (bool): If True, simulate execution.
    """
    if dry_run:
        logger.info(f"Dry run: convert {input_filepath} to {output_filepath} using {tool}")
        return

    if tool == 'lame':
        command = ["lame", input_filepath, output_filepath]
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error converting {input_filepath} with lame: {e}")
            raise

    elif tool == 'pydub':
        try:
            sound = pydub.AudioSegment.from_wav(input_filepath)
            sound.export(output_filepath, format="mp3")
        except Exception as e:
            logger.error(f"Error converting {input_filepath} with pydub: {e}")
            raise
    else:
        raise ValueError(f"Unknown tool: {tool}")

def convert_directory(input_dir, output_dir, tool='pydub', dry_run=False):
    """
    Batch convert all WAV files in a directory to MP3.
    """
    makedirs(output_dir, exist_ok=True)
    files = glob(join(input_dir, '*.wav'))
    
    for input_filepath in tqdm(files, desc="Converting WAV to MP3"):
        output_filepath = join(output_dir, splitext(basename(input_filepath))[0] + '.mp3')
        convert_file(input_filepath, output_filepath, tool, dry_run)

def main():
    parser = argparse.ArgumentParser(description="Convert WAV files to MP3.")
    parser.add_argument('-i', '--input', default='input', help='Input folder')
    parser.add_argument('-o', '--output', default='output', help='Output folder')
    parser.add_argument('-t', '--tool', default='pydub', choices=['pydub', 'lame'], help='Conversion tool to use')
    parser.add_argument('--dry-run', action='store_true', help='Simulate conversion without executing')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    convert_directory(args.input, args.output, args.tool, args.dry_run)

if __name__ == "__main__":
    main()
