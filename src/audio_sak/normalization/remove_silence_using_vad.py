#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Source: https://github.com/WeberJulian/TTS-1/blob/multilingual/TTS/bin/remove_silence_using_vad.py
import os
import contextlib
import wave
import collections
import logging
import argparse
import multiprocessing
from itertools import chain
from glob import glob
from functools import partial

import webrtcvad
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

logger = logging.getLogger(__name__)

def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate

def write_wave(path, audio, sample_rate):
    """Writes a .wav file.
    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)

class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data."""
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames."""
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])

def process_file(file_info, aggressiveness=2, padding_duration_ms=300, frame_duration_ms=30, force=False):
    """
    Process a single file to remove silence.
    file_info: tuple (input_path, output_dir)
    """
    filepath, output_dir = file_info
    output_path = os.path.join(output_dir, os.path.basename(filepath))
    
    if os.path.exists(output_path) and not force:
        return False

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        audio, sample_rate = read_wave(filepath)
        vad = webrtcvad.Vad(aggressiveness)
        frames = frame_generator(frame_duration_ms, audio, sample_rate)
        frames = list(frames)
        segments = vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames)
        
        # Concatenate segments
        concatenated_audio = b''.join(segments)
        
        if concatenated_audio:
            write_wave(output_path, concatenated_audio, sample_rate)
            return True
        else:
            logger.warning(f"No speech detected in {filepath}, copying original.")
            write_wave(output_path, audio, sample_rate)
            return True

    except Exception as e:
        logger.error(f"Error processing {filepath}: {e}")
        return False

def execute_silence_removal(input_dir, output_dir, glob_pattern='*.wav', force=False, aggressiveness=2):
    files = sorted(glob(os.path.join(input_dir, glob_pattern))) # simplified glob usage
    # Note: original code did recursive search? 
    # Original: join(args.input, folder, args.glob) inside loop over os.listdir(args.input)
    # This implies a specific directory structure.
    # I will adapt to a simpler recursive glob if user wants, or just flattened list.
    # Let's support recursive finding if glob has **.
    
    if not files:
        # Fallback to recursive search if input_dir seems to be a root
        files = sorted(glob(os.path.join(input_dir, '**', glob_pattern), recursive=True))

    if not files:
        logger.warning(f"No files found in {input_dir}")
        return

    logger.info(f"Processing {len(files)} files...")
    
    file_tuples = [(f, output_dir) for f in files]
    
    # Partial function to pass constant args
    worker = partial(process_file, aggressiveness=aggressiveness, force=force)
    
    num_threads = multiprocessing.cpu_count()
    process_map(worker, file_tuples, max_workers=num_threads, chunksize=1, desc="Removing Silence")

def main():
    parser = argparse.ArgumentParser(description="Remove silence using VAD.")
    parser.add_argument('-i', '--input', type=str, default='input', help='Input directory')
    parser.add_argument('-o', '--output', type=str, default='output', help='Output directory')
    parser.add_argument('--force', action='store_true', help='Force overwrite')
    parser.add_argument('-g', '--glob', type=str, default='*.wav', help='Glob pattern')
    parser.add_argument('-a', '--aggressiveness', type=int, default=2, help='VAD aggressiveness (0-3)')
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    
    execute_silence_removal(args.input, args.output, args.glob, args.force, args.aggressiveness)

if __name__ == "__main__":
    main()
