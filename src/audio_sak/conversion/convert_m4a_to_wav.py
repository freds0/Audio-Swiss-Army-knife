#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
from glob import glob
from os import makedirs
from os.path import join, basename, splitext
from tqdm import tqdm
from pydub import AudioSegment

logger = logging.getLogger(__name__)

def convert_m4a_to_wav(input_filepath, output_filepath, dry_run=False):
    if dry_run:
        logger.info(f"Dry run: {input_filepath} -> {output_filepath}")
        return

    try:
        audio = AudioSegment.from_file(input_filepath, format="m4a") 
        # 16 bits PCM
        audio = audio.set_sample_width(2)  # 2 bytes (16 bits)
        audio.export(output_filepath, format="wav", parameters=["-acodec", "pcm_s16le"])
    except Exception as e:
        logger.error(f"Failed to convert {input_filepath}: {e}")
        raise

def process_directory(input_dir, output_dir, dry_run=False):
    makedirs(output_dir, exist_ok=True)
    files = glob(join(input_dir, '*.m4a'))
    
    for input_filepath in tqdm(files, desc="Converting M4A to WAV"):
        output_filepath = join(output_dir, splitext(basename(input_filepath))[0] + '.wav')
        convert_m4a_to_wav(input_filepath, output_filepath, dry_run)

def main():
    parser = argparse.ArgumentParser(description="Convert M4A to WAV.")
    parser.add_argument('-i', '--input', default='input', help='Input folder')
    parser.add_argument('-o', '--output', default='output', help='Output folder')
    parser.add_argument('--dry-run', action='store_true', help='Simulate execution')
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    
    process_directory(args.input, args.output, args.dry_run)

if __name__ == "__main__":
    main()
