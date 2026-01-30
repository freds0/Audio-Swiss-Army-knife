#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
import os
from glob import glob
import torchaudio
from tqdm import tqdm

logger = logging.getLogger(__name__)

def voice_detector(input_filepath):
    try:
        waveform, sr = torchaudio.load(input_filepath)
        # Check for vad availability or use alternative logic if missing
        try:
            trimmed_waveform = torchaudio.functional.vad(waveform, sr)
            score = sum(trimmed_waveform.squeeze()).item()
            logger.info(f"{input_filepath}: {score}")
            return score
        except AttributeError:
             logger.warning("torchaudio.functional.vad is deprecrated or missing. Skipping VAD.")
             return 0
    except Exception as e:
        logger.error(f"Failed to process {input_filepath}: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(description="Run VAD detection.")
    parser.add_argument('-i', '--input', type=str, default='input', help='Dataset root dir')
    parser.add_argument('-o', '--output', type=str, default='output', help='Output Dataset dir') 
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    args.output = os.path.join(args.input, args.output) if not os.path.isabs(args.output) else args.output 
    os.makedirs(args.output, exist_ok=True)

    wavs = glob(os.path.join(args.input, "*.wav"))
    for input_filepath in tqdm(wavs, desc="Detecting Voice"):
        voice_detector(input_filepath)

if __name__ == "__main__":
    main()
