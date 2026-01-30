#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
import os
import librosa
import soundfile as sf
from tqdm import tqdm

logger = logging.getLogger(__name__)

def segment_wav(input_file, output_dir, duration):
    try:
        if not os.path.exists(input_file):
            logger.error(f"Input file {input_file} not found.")
            return

        filename = os.path.splitext(os.path.basename(input_file))[0]
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Loading {input_file}...")
        y, sr = librosa.load(input_file, sr=None)

        segment_size = int(duration * sr)
        # Avoid division by zero if segment_size is 0 (duration 0)
        if segment_size <= 0:
             logger.error("Duration must be positive.")
             return

        num_segments = int(len(y) // segment_size)
        remainder = len(y) % segment_size
        if remainder > 0:
             # Original code ignored remainder? `range(num_segments)`
             # "i * segment_size" to "(i+1) * segment_size".
             # It dropped the last partial segment.
             # I will keep this behavior but maybe warn?
             # Or should I include remainder?
             # "segment_audio_same_duration" implies fixed size.
             # Keep original logic.
             pass

        logger.info(f"Splitting into {num_segments} segments of {duration}s...")
        
        for i in tqdm(range(num_segments), desc="Segmenting"):
            start_sample = i * segment_size
            end_sample = (i + 1) * segment_size
            segment = y[start_sample:end_sample]
       
            output_file = os.path.join(output_dir, f'{filename}-{i:04d}.wav')
            sf.write(output_file, segment, sr, subtype='PCM_16')
            
    except Exception as e:
        logger.error(f"Failed to segment {input_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Segment audio into fixed duration chunks.")
    parser.add_argument('-i', '--input', required=True, help="Input filepath")
    parser.add_argument('-o', '--output', default="output", help="Output folder")
    parser.add_argument('-d', '--duration', type=float, default=15.0, help='Duration in seconds')
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    segment_wav(args.input, args.output, args.duration)

if __name__ == "__main__":
    main()