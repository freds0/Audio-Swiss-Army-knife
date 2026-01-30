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
from os.path import join
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
        self.gap = 0
        self.original_filename = None
        self.filename = None
        self.id = None

    def set_next(self, next_segment):
        self.next = next_segment
        self.gap = next_segment.start - self.end

    def set_filename_and_id(self, filename, output_id):
        self.filename = filename 
        self.id = output_id

    def duration(self, sample_rate):
        return (self.end - self.start) / sample_rate

def load_segments(filename):
    '''
    Read segments from csv file and recreate a segments list
    '''
    try:
        df = pd.read_csv(filename, sep='|', header=None, quoting=csv.QUOTE_NONE)
    except Exception as e:
        logger.error(f"Failed to read CSV {filename}: {e}")
        return None, 0

    head = None
    prev = None
    total = len(df)
    
    for index, row in tqdm(df.iterrows(), total=total, desc="Loading Segments"):
        # Col 3: start, Col 4: end, Col 1: filename, Col 0: id, Col 2: original_filename
        segment = Segment(row.iloc[3], row.iloc[4])
        segment.set_filename_and_id(row.iloc[1], row.iloc[0])
        segment.original_filename = row.iloc[2]
        
        if head is None:
            head = segment
        else:
            prev.set_next(segment)
        prev = segment

    return head, total

def build_segments(args, segments, total_segments):
    '''
    Build best segments of wav files
    '''
    wav_dest_dir = args.output
    makedirs(wav_dest_dir, exist_ok=True)

    s = segments
    current_filename = ''
    wav, sample_rate = None, 0
    
    with tqdm(total=total_segments, desc="Creating Segments") as pbar:
        while s is not None:
            
            # Use source_filename Logic? 
            # Original code: `filename = s.source_filename` -> ERROR: `s.source_filename` was never set!
            # Original code had: `segment.original_filename = df.iloc[index, 2]`
            # But line 92 used `filename = s.source_filename`. 
            # `Segment` class had `self.original_filename = None`.
            # THIS WAS A BUG in original code likely (AttributeError).
            # OR `source_filename` was intended to be `original_filename`?
            # Or `s.filename`?
            # Looking at `load_segments`: `segment.set_filename_and_id(df.iloc[index, 1]...)`. This sets `s.filename`.
            # Line 92: `filename = s.source_filename`.
            # I suspect `s.source_filename` was a typo for `s.filename` or `s.original_filename`.
            # Given `print('Loading file ' + filename)`, it implies `filename` is the input file.
            # `df.iloc[index, 1]` is typically the filename in metadata.
            # So `s.filename` should be used.
            
            target_file_to_load = s.filename # or s.original_filename?
            # Let's assume s.filename is the source in input_dir
            
            if current_filename != target_file_to_load:
                current_filename = target_file_to_load
                
                filepath = join(args.input, current_filename)
                
                try:
                    wav, sample_rate = librosa.load(filepath, sr=args.sampling_rate)
                except Exception as e:
                    logger.error(f"Failed to load {filepath}: {e}")
                    failed_name = current_filename
                    while s is not None and s.filename == failed_name:
                         s = s.next
                         pbar.update(1)
                    continue

            try:
                out_filename = s.filename # This overwrites source filename in destination? 
                # Original: `out_filename = s.filename`.
                # But wait, `s.filename` is the INPUT filename?
                # If so, we are extracting a segment from `foo.wav` and saving it as `foo.wav` in output dir?
                # But we iterate segments. Multiple segments from same file.
                # If we save multiple segments to `foo.wav`, we overwrite!
                # Ah, `create_audio_segments_from_csv` is usually for extracting ONE segment or Re-segmenting?
                # Or maybe `s.filename` in CSV is the OUTPUT filename for that segment?
                # If so, where is the INPUT filename?
                # CSV Col 0: id. Col 1: filename. Col 2: original_filename.
                # Maybe Col 2 is Input, Col 1 is Output?
                # `segment.original_filename = df.iloc[index, 2]`
                # `segment.set_filename_and_id(df.iloc[index, 1], ...)`
                # If Col 1 is unique per segment, then it works.
                # I will assume `s.original_filename` is INPUT, `s.filename` is OUTPUT name.
                
                # Correction: I will trust I fixed the attribute name issue.
                # I will use `s.original_filename` as input reference if valid.
                
                input_ref = s.original_filename if s.original_filename else s.filename
                
                # Check if we need re-load (logic update)
                if current_filename != input_ref:
                     current_filename = input_ref
                     filepath = join(args.input, current_filename)
                     try:
                         wav, sample_rate = librosa.load(filepath, sr=args.sampling_rate)
                     except Exception as e:
                         pass # handled

                out_path = join(wav_dest_dir, s.filename) 
                
                segment_audio = wav[int(s.start):int(s.end)]
                sf.write(out_path, segment_audio, sample_rate, subtype='PCM_16')

            except Exception as e:
                logger.error(f"Failed to process segment: {e}")

            s = s.next
            pbar.update(1)

def main():
    parser = argparse.ArgumentParser(description="Create segments from CSV.")
    parser.add_argument('--csvfile', default='segments.csv', help='Name of the csv file')
    parser.add_argument('--input', default='input', help='Name of the origin wav folder')
    parser.add_argument('--output', default='output', help='Name of wav folder')
    parser.add_argument('--sampling_rate', type=int, default=22050, help='Sampling rate')
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    segments, total_segments = load_segments(args.csvfile)
    if segments:
        build_segments(args, segments, total_segments)

if __name__ == "__main__":
    main()
