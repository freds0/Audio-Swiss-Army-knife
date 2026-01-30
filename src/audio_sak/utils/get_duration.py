#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import soundfile as sf
import glob
from tqdm import tqdm
import logging
import os

logger = logging.getLogger(__name__)

def get_seconds(x):
    try:
        f = sf.SoundFile(x)
        t = len(f) / f.samplerate
        return t
    except Exception as e:
        logger.error(f"Error reading {x}: {e}")
        return 0

def calculate(input_dir):
    total = 0
    files = glob.glob(os.path.join(input_dir, '**', '*.wav'), recursive=True)
    if not files: # Fallback
        files = glob.glob(os.path.join(input_dir, '*.wav'))

    for filepath in tqdm(files, desc="Calculating Duration"):
        total += get_seconds(filepath)

    hours = total / 3600
    minutes = (total % 3600) / 60
    seconds = (total % 3600) % 60
    
    logger.info(f'Total (sec): {total:.2f}')
    logger.info(f'Hours: {hours:.2f}')
    logger.info(f'Minutes: {minutes:.2f}')
    logger.info(f'Seconds: {seconds:.2f}')
    
    print(f'Total (sec): {total:.2f}')
    print(f'Hours: {hours:.2f}')
    print(f'Minutes: {minutes:.2f}')
    print(f'Seconds: {seconds:.2f}')

def main():
    parser = argparse.ArgumentParser(description="Calculate total duration of wav files.")
    parser.add_argument('--input_dir', required=True, help='Input folder')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    calculate(args.input_dir)

if __name__ == "__main__":
    main()

