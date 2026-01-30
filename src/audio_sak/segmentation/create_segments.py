#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Adapted from https://gist.github.com/keithito/771cfc1a1ab69d1957914e377e65b6bd from Keith Ito: kito@kito.us
import argparse
import logging
import csv
import os
import pandas as pd
import librosa
import numpy as np
import soundfile as sf
from os import makedirs
from os.path import join, basename
from tqdm import tqdm

logger = logging.getLogger(__name__)

class Segment:
    '''
    Linked segments lists
    '''
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.next = None
        self.gap = 0 # gap between segments (current and next)
        self.filename = None
        self.id = None

    def set_next(self, next_segment):
        self.next = next_segment
        self.gap = next_segment.start - self.end

    def set_filename_and_id(self, filename, output_id):
        self.filename = filename # Full path or relative path?
        self.id = output_id

    def duration(self, sample_rate):
        return (self.end - self.start) / sample_rate

def load_segments(filename):
    '''
    Read segments from csv file and recreate a segments list
    '''
    # Header logic: 
    # original script uses header=None, iloc.
    # Col 0: id
    # Col 1: folder/filename (input)
    # Col 2: start
    # Col 3: end
    
    try:
        df = pd.read_csv(filename, sep='|', header=None, quoting=csv.QUOTE_NONE)
    except Exception as e:
        logger.error(f"Failed to read CSV {filename}: {e}")
        return None, 0

    head = None
    prev = None
    total = len(df)
    
    for index, row in tqdm(df.iterrows(), total=total, desc="Loading Segments"):
        try:
            start = float(row.iloc[2]) # iloc[2] in original
            end = float(row.iloc[3])
            
            segment = Segment(start, end)
            
            # folder/filename parse
            # Original: folder, filename = df.iloc[index, 1].split("/")
            # filepath = os.path.join(folder, filename)
            # This assumes strict "folder/filename" format.
            # If input is just "filename", it fails.
            raw_path = str(row.iloc[1])
            parts = raw_path.split("/")
            if len(parts) >= 2:
                # Reconstruct path? Or just use as is?
                # Original logic: `os.path.join(folder, filename)`
                # If parts > 2, e.g. "a/b/c.wav", `split("/")` returns 3.
                # Original: `folder, filename = ...` would FAIL if > 2 parts!
                # I'll fix this to be more robust.
                filepath = raw_path 
            else:
                filepath = raw_path

            id_file = row.iloc[0]
            segment.set_filename_and_id(filepath, id_file)
            
            if head is None:
                head = segment
            else:
                prev.set_next(segment)
            prev = segment
        except Exception as e:
            logger.warning(f"Skipping row {index}: {e}")

    return head, total

def build_segments(segments, total_segments, input_dir, output_dir):
    '''
    Build best segments of wav files
    '''
    makedirs(output_dir, exist_ok=True)

    s = segments
    current_filename = ''
    wav, sample_rate = None, 0

    processed_count = 0
    with tqdm(total=total_segments, desc="Building Segments") as pbar:
        while s is not None:
            # Check if we need to load a new file
            # Original code assumes segments are sorted by filename?
            # If not, it reloads frequently.
            # "if filename != s.filename"
            # s.filename comes from CSV.
            # Note: s.filename from CSV might be relative path.
            # `filepath = os.path.join(input_dir, filename)`
            
            if current_filename != s.filename:
                current_filename = s.filename
                filepath = join(input_dir, current_filename)
                
                # Handle potential path issues if input_dir already contains part of path?
                # or if csv contains absolute path?
                # Assuming relative to input_dir.
                
                try:
                    # logger.info(f"Loading {filepath}")
                    wav, sample_rate = librosa.load(filepath, sr=None)
                except Exception as e:
                    logger.error(f"Failed to load {filepath}: {e}")
                    # Skip segments for this file?
                    # Move to next S until filename changes?
                    failed_filename = current_filename
                    while s is not None and s.filename == failed_filename:
                         s = s.next
                         pbar.update(1)
                    continue

            # Extract segment
            try:
                # Original: wav[s.start:s.end] * 32767 -> int16
                # Using soundfile with PCM_16 subtype handles standard float audio [-1, 1].
                # Librosa loads as float32 [-1, 1].
                
                segment_audio = wav[int(s.start):int(s.end)]
                out_path = join(output_dir, f'{s.id}.wav')
                
                sf.write(out_path, segment_audio, sample_rate, subtype='PCM_16')
            except Exception as e:
                logger.error(f"Failed to write segment {s.id}: {e}")

            s = s.next
            pbar.update(1)

def main():
    parser = argparse.ArgumentParser(description="Create segments from CSV.")
    parser.add_argument('--csvfile', default='segments.csv', help='Name of the csv file')
    parser.add_argument('--input', default='input', help='Name of the origin wav folder')
    parser.add_argument('--output', default='output', help='Name of wav folder')
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    segments, total_segments = load_segments(args.csvfile)
    if segments:
        build_segments(segments, total_segments, args.input, args.output)
    else:
        logger.error("No segments loaded.")

if __name__ == "__main__":
    main()
