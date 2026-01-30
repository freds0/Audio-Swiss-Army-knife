#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Adapted from https://gist.github.com/keithito/771cfc1a1ab69d1957914e377e65b6bd from Keith Ito: kito@kito.us
import argparse
import logging
import os
from glob import glob
from os import listdir, makedirs
from os.path import isfile, join, basename, splitext
from collections import OrderedDict
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm
import pydub

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
        self.filename = filename
        self.id = output_id

    def merge_from(self, next_segment):
        # merge two segments (current and next)
        self.next = next_segment.next
        self.gap = next_segment.gap
        self.end = next_segment.end

    def duration(self, sample_rate):
        return (self.end - self.start) / sample_rate

    def __repr__(self):
        return f"Segment(start={self.start}, end={self.end}, gap={self.gap})"

def segment_wav(audio_data, threshold_db, sr=None):
    '''
    Segment audio file and return a segment linked list
    
    Args:
        audio_data: np.ndarray, audio samples
        threshold_db: float, silence threshold in dB
        sr: int, sample rate (unused for splitting but good for context)
    '''
    # Find gaps at a fine resolution:
    # frame_length=1024, hop_length=256 are librosa defaults or good for speech
    parts = librosa.effects.split(audio_data, top_db=threshold_db, frame_length=1024, hop_length=256)

    # Build up a linked list of segments:
    head = None
    prev = None
    for start, end in parts:
        segment = Segment(start, end)
        if head is None:
            head = segment
        else:
            prev.set_next(segment)
        prev = segment
    return head

def find_best_merge(segments, sample_rate, max_duration, max_gap_duration):
    '''
    Find small segments that can be merged by analyzing max_duration and max_gap_duration
    '''
    best = None
    best_score = 0
    s = segments
    while s is not None and s.next is not None:
        gap_duration = s.gap / sample_rate
        merged_duration = (s.next.end - s.start) / sample_rate
        if gap_duration <= max_gap_duration and merged_duration <= max_duration:
            score = max_gap_duration - gap_duration
            if score > best_score:
                best = s
                best_score = score
        s = s.next
    return best

def find_segments(filename, wav, sample_rate, min_duration, max_duration, max_gap_duration, threshold_db, output_dir):
    '''
    Given an audio file, creates the best possible segment list
    '''
    # Segment audio file
    segments = segment_wav(wav, threshold_db, sr=sample_rate)
    
    # Merge until we can't merge any more
    while True:
        best = find_best_merge(segments, sample_rate, max_duration, max_gap_duration)
        if best is None:
            break
        best.merge_from(best.next)

    # Convert to list and process
    result = []
    s = segments
    while s is not None:
        # Check duration errors
        dur = s.duration(sample_rate)
        if dur < min_duration or dur > max_duration:
             # Only log valid segments or handle errors?
             # Original code wrote to errors.txt but kept the segment in the list?
             # "result.append(s)" happens before check? Yes.
             # Wait, original code:
             # result.append(s)
             # if (s.duration...): write error
             # s.end += int(0.2 * sample_rate)
             pass
             
        result.append(s)
        
        if (dur < min_duration or dur > max_duration):
            error_file = join(output_dir, "errors.txt")
            with open(error_file, "a") as f:
                f.write(filename + "\n")

        # Extend the end by 0.2 sec as we sometimes lose the ends of words ending in unvoiced sounds.
        # Ensure we don't go out of bounds?
        # Original code didn't check bounds, relying on slice clipping or luck?
        # numpy slicing [start:end] clips automatically if end > len.
        s.end += int(0.2 * sample_rate)
        
        s = s.next

    return result

def load_filenames(input_dir):
    '''
    Given an folder, creates a wav file alphabetical order dict
    '''
    mappings = OrderedDict()
    files = sorted(glob(join(input_dir, "*.wav")))
    for filepath in files:
        filename = basename(filepath).split('.')[0]
        mappings[filename] = filepath
    return mappings

def build_segments(input_dir, output_dir, min_duration, max_duration, max_gap_duration, threshold_db, args_filename, args_filename_id):
    '''
    Build best segments of wav files
    '''
    total_duration = 0
    mean_duration_seg = 0
    max_duration_seg = 0
    min_duration_seg = 999
    all_segments = []
    
    init_filename_id = args_filename_id if args_filename_id else 1
    filenames = load_filenames(input_dir)
    
    if not filenames:
        logger.warning(f"No WAV files found in {input_dir}")
        return

    for i, (file_id, filename) in enumerate(tqdm(filenames.items(), desc="Processing Files")):
        logger.info(f'Loading {file_id}: {filename} ({i+1} of {len(filenames)})')
        try:
            wav, sample_rate = librosa.load(filename, sr=None)
        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}")
            continue
            
        audio_duration = len(wav) / sample_rate / 60
        logger.info(f' -> Loaded {audio_duration:.2f} min of audio. Splitting...')

        # Find best segments
        segments = find_segments(filename, wav, sample_rate, min_duration, max_duration, max_gap_duration, threshold_db, output_dir)
        
        # Calculate duration of KEPT segments
        current_segments_duration = sum((s.duration(sample_rate) for s in segments))
        total_duration += current_segments_duration

        # Create records for the segments
        output_filename_base = args_filename if args_filename else file_id
        output_filename_id = init_filename_id
        
        for s in segments:
            all_segments.append(s)
            s.set_filename_and_id(filename, '%s-%04d' % (output_filename_base, output_filename_id))
            output_filename_id += 1

        if segments:
            avg_seg_len = current_segments_duration / len(segments)
            logger.info(' -> Segmented into %d parts (%.1f min, %.2f sec avg)' % (len(segments), current_segments_duration / 60, avg_seg_len))
        else:
            logger.info(' -> No segments found.')

        # Write segments to disk:
        for s in segments:
            # Scale float32 to int16 range, and cast.
            # soundfile can write float32 directly as PCM_16 if subtype is specified?
            # Or we can just write float32 and let it handle?
            # Original code did explicit scalable: (wav * 32767).astype(int16)
            # This is standard for librosa [-1, 1] load.
            # I will use soundfile, which handles float inputs nicely if subtype='PCM_16' is used?
            # Actually sf.write(file, data, samplerate, subtype='PCM_16') will clip and quantize.
            # But if the data is float [-1, 1], sf.write default behavior is to scale? 
            # documentation says: If data is float, it is saved as float by default for WAV (float32).
            # If we want 16-bit PCM, we must specify subtype='PCM_16'.
            # And SF will NOT automatically scale float 1.0 to 32767. It expects float range [-1, 1] and maps it to min/max of int16.
            # So I don't need to manually multiply by 32767.
            
            segment_audio = wav[int(s.start):int(s.end)]
            out_path = join(output_dir, f'{s.id}.wav')
            
            sf.write(out_path, segment_audio, sample_rate, subtype='PCM_16')

            seg_len_sec = len(segment_audio) / sample_rate
            
            if seg_len_sec > max_duration_seg:
                max_duration_seg = seg_len_sec
            if seg_len_sec < min_duration_seg:
                min_duration_seg = seg_len_sec

            mean_duration_seg += seg_len_sec

    if all_segments:
        mean_duration_seg /= len(all_segments)
        logger.info(' -> Wrote %d segment wav files' % len(all_segments))
        logger.info(' -> Progress: %d segments, %.2f hours, %.2f sec avg' % (
            len(all_segments), total_duration / 3600, mean_duration_seg))

        logger.info('Writing metadata for %d segments (%.2f hours)' % (len(all_segments), total_duration / 3600))
        with open(join(output_dir, 'segments.csv'), 'w') as f:
            for s in all_segments:
                f.write('%s|%s|%d|%d\n' % (s.id, s.filename, s.start, s.end))
     
        logger.info(f'Min: {min_duration_seg:.2f}')
        logger.info(f'Mean: {mean_duration_seg:.2f}')
        logger.info(f'Max: {max_duration_seg:.2f}')
    else:
        logger.warning("No segments created.")

def convert_mp3_to_wav(input_mp3, output_wav):
    '''
    Convert mp3 folder files to wav
    '''
    if not input_mp3:
        return
    mp3files = [f for f in listdir(input_mp3) if isfile(join(input_mp3, f)) and f.lower().endswith('.mp3')]
    makedirs(output_wav, exist_ok=True)
    for mp3 in tqdm(mp3files, desc="Converting MP3"):
        sound = pydub.AudioSegment.from_mp3(join(input_mp3, mp3))
        filename = splitext(mp3)[0]
        sound.export(join(output_wav, f'{filename}.wav'), format="wav")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input', help='Input wav folder')
    parser.add_argument('--output', default='output', help='Output wav folder')
    parser.add_argument('--input_mp3', default=None, help='Input mp3 folder')
    parser.add_argument('--min_duration', type=float, default=5.0, help='Minimum duration of a segment in seconds')
    parser.add_argument('--max_duration', type=float, default=15.0, help='Maximum duration of a segment in seconds')
    parser.add_argument('--max_gap_duration', type=float, default=3.0, help='Maximum duration of a gap between segments in seconds')
    parser.add_argument('--output_filename', type=str, default='', help='Default output filename')
    parser.add_argument('--output_filename_id', type=int, default=1, help='Sequencial number used for id filename.')
    parser.add_argument('--threshold_db', type=float, default=28.0, help='The threshold (in decibels) below reference to consider as silence: ')
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.input_mp3:
        convert_mp3_to_wav(args.input_mp3, args.output) # original usage said output to output?
        # Original: convert_mp3_to_wav(args.input_mp3, args.output)
        # But `build_segments` reads from `args.input`.
        # So maybe meaningful usage is: convert mp3 to input, then run?
        # Or convert mp3 to temporary dict?
        # Original script: convert_mp3 to output?
        # But build_segments runs on input_folder.
        # This seems like a separate utility bundled in.
        
    folders = [f for f in listdir(args.input) if not isfile(join(args.input, f))]
    if not folders:
        # Maybe input is the folder itself?
        # Original script iterates `listdir(args.input)`.
        # And constructs `join(args.input, folder)`.
        # This implies structure: `input/subdir1/*.wav`, `input/subdir2/*.wav`...
        # If flat structure, loop might fail or treat file as folder.
        # I'll enable flat structure handling.
        files_in_root = glob(join(args.input, "*.wav"))
        if files_in_root:
            logger.info("Detected flat directory structure.")
            build_segments(args.input, args.output, args.min_duration, args.max_duration, args.max_gap_duration, args.threshold_db, args.output_filename, args.output_filename_id)
            return

    for folder in folders:
        input_folder = join(args.input, folder)
        output_folder = join(args.output, folder)
        makedirs(output_folder, exist_ok=True)
        build_segments(input_folder, output_folder, args.min_duration, args.max_duration, args.max_gap_duration, args.threshold_db, args.output_filename, args.output_filename_id)

if __name__ == "__main__":
    main()

