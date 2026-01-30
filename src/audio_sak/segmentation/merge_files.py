#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
import os
from glob import glob
from pydub import AudioSegment
from tqdm import tqdm

logger = logging.getLogger(__name__)

def merge_audio_files(wavs_dir, output_file, gap_ms=50):
    files = sorted(glob(os.path.join(wavs_dir, '*.wav')))
    if not files:
        logger.warning(f"No WAV files found in {wavs_dir}")
        return

    merged_audio_data = AudioSegment.empty()
    silence = AudioSegment.silent(duration=gap_ms)
    
    logger.info(f"Merging {len(files)} files...")
    
    for i, filepath in enumerate(tqdm(files, desc="Merging")):
        try:
            audio_data = AudioSegment.from_file(filepath)
            
            if i > 0:
                merged_audio_data += silence
                
            merged_audio_data += audio_data
        except Exception as e:
            logger.error(f"Failed to merge {filepath}: {e}")

    # Increase volume logic from original (merged_audio_data += 5) -> +5dB
    merged_audio_data += 5

    try:
        merged_audio_data.export(output_file, format="wav")
        logger.info(f"Saved merged file to {output_file}")
    except Exception as e:
        logger.error(f"Failed to export {output_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Merge WAV files with silence gap.")
    parser.add_argument('-i', '--input', default='input', help='Input folder')
    parser.add_argument('-o', '--output', default='output.wav', help='Output file')
    parser.add_argument('-g', '--gap', type=int, default=50, help='Gap in ms')
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    
    merge_audio_files(args.input, args.output, args.gap)

if __name__ == "__main__":
    main()
