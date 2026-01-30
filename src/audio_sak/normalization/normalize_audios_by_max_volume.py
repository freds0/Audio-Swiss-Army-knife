#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
import glob
from os import makedirs
from os.path import join, basename, exists, splitext
from pydub import effects  
from pydub import AudioSegment
import tqdm

logger = logging.getLogger(__name__)

def normalize_files(input_dir, output_dir, dry_run=False):
    makedirs(output_dir, exist_ok=True)
    files = sorted(glob.glob(join(input_dir, "*.wav")))
    
    for path_file in tqdm.tqdm(files, desc="Peak Normalization"):
        try:
            filename = basename(path_file)
            dest_file = join(output_dir, filename)

            if dry_run:
                logger.info(f"Dry run: normalize {path_file} -> {dest_file}")
                continue

            _sound = AudioSegment.from_file(path_file, "wav")  
            sound = effects.normalize(_sound)  
            sound.export(dest_file, format="wav")            
        except Exception as e:
            logger.error(f"Failed to normalize {path_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Peak normalize audio files.")
    parser.add_argument('-i', '--input', default='input', help='Input folder')
    parser.add_argument('-o', '--output', default='output', help='Output folder')
    parser.add_argument('--dry-run', action='store_true', help='Simulate execution')
  
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    
    normalize_files(args.input, args.output, args.dry_run)

if __name__ == "__main__":
    main()
