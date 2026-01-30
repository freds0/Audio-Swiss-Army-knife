#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
import glob
from os import makedirs
from os.path import join, basename, exists
from pydub.effects import normalize
from pydub import AudioSegment
import tqdm

logger = logging.getLogger(__name__)

def normalize_audio(input_filepath, output_filepath, dry_run=False):
    if dry_run:
        logger.info(f"Dry run: normalize {input_filepath} -> {output_filepath}")
        return

    try:
        waveform = AudioSegment.from_file(input_filepath) # improved from_wav to from_file
        waveform = normalize(waveform)
        waveform.export(output_filepath, format="wav")
    except Exception as e:
        logger.error(f"Failed to normalize {input_filepath}: {e}")
        raise

def process_directory(input_dir, output_dir, dry_run=False):
    makedirs(output_dir, exist_ok=True)
    files = sorted(glob.glob(join(input_dir, "*.wav")))
    
    for audio_filepath in tqdm.tqdm(files, desc="Normalizing (pydub)"):
        filename = basename(audio_filepath)
        output_filepath = join(output_dir, filename)
        normalize_audio(audio_filepath, output_filepath, dry_run)

def main():
    parser = argparse.ArgumentParser(description="Normalize audios using pydub.")
    parser.add_argument('-i', '--input', default='input', help='Input folder')
    parser.add_argument('-o', '--output', default='output', help='Output folder')
    parser.add_argument('--dry-run', action='store_true', help='Simulate execution')
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    
    process_directory(args.input, args.output, args.dry_run)

if __name__ == "__main__":
    main()
