#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
from pydub import AudioSegment
from os import listdir
from os.path import isfile, join, basename
from tqdm import tqdm

logger = logging.getLogger(__name__)

def get_silence(audio_path, threshold, interval):
    "get length of silence in seconds from a wav file"
    try:
        wav = AudioSegment.from_wav(audio_path)
        
        # Optimized silence detection logic instead of manual chunks?
        # Manual chunks is slow for python. 
        # Pydub has detect_silence? Or split_on_silence?
        # But this function measures *total silence*? 
        # Original code: `silent_blocks += 1` in loop.
        # This counts total silent blocks.
        
        # Let's keep logic but improve safety.
        chunks = [wav[i:i+interval] for i in range(0, len(wav), interval)]
        silent_blocks = 0
        for c in chunks:
            if c.dBFS == float('-inf') or c.dBFS < threshold:
                silent_blocks += 1
        
        return round(silent_blocks * (interval / 1000), 3)
    except Exception as e:
        logger.error(f"Error processing {audio_path}: {e}")
        return 0

def get_duration(audio_path):
    try:
        wav = AudioSegment.from_wav(audio_path)
        return round(len(wav) / 1000, 3)
    except Exception:
        return 0 # Error handled above likely

def main():
    parser = argparse.ArgumentParser(description="Calculate silence ratio in audio files.")
    parser.add_argument('--input_dir', required=True, help='Input folder')
    parser.add_argument('--threshold', type=float, default=-80, help='Threshold for silence (dBFS).')
    parser.add_argument('--interval', type=int, default=1, help='Interval in ms.')
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    audio_files = [join(args.input_dir, f) for f in listdir(args.input_dir) if isfile(join(args.input_dir, f)) and f.endswith('.wav')]
    
    print("filename|leading_silence|duration|percentage")
    for file_path in tqdm(audio_files, desc="Processing"):
        silence = get_silence(file_path, args.threshold, args.interval)
        duration = get_duration(file_path)
        
        if duration > 0:
            percentage = round(silence / duration * 100, 3)
        else:
            percentage = 0
            
        print(f"{basename(file_path)}|{silence}|{duration}|{percentage}")

if __name__ == "__main__":
    main()


