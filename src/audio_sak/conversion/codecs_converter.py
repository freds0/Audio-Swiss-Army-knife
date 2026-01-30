#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import subprocess
import logging
import shutil
import tempfile
from glob import glob
from os import makedirs
from os.path import join, basename, splitext, exists
from tqdm import tqdm

logger = logging.getLogger(__name__)

TEMP_SAMPLING_RATE = 8000
OUTPUT_SAMPLING_RATE = 16000

def run_ffmpeg(command, description):
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during {description}: {e}")
        raise

def simulate_codec(input_filepath, output_filepath, codec, dry_run=False):
    """
    Simulate codec compression artifacts by encoding to a low-bitrate format and decoding back to WAV.
    
    Args:
        input_filepath (str): Input WAV file.
        output_filepath (str): Output WAV file.
        codec (str): Codec to simulate ('mp3', 'gsm', 'g726', 'adpcm', 'g723', 'ogg').
        dry_run (bool): If True, simulate execution.
    """
    if dry_run:
        logger.info(f"Dry run: simulate {codec} on {input_filepath} -> {output_filepath}")
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        filename = splitext(basename(input_filepath))[0]
        
        # Define codec-specific parameters
        if codec == 'mp3':
            intermediate_ext = '.mp3'
            enc_opts = ["-acodec", "mp3", "-ar", str(TEMP_SAMPLING_RATE), "-ac", "1", "-b:a", "16k"]
            dec_opts = ["-acodec", "mp3", "-f", "wav", "-ar", str(OUTPUT_SAMPLING_RATE)]
        elif codec == 'gsm':
            intermediate_ext = '.wav'
            enc_opts = ["-c:a", "libgsm", "-ar", str(TEMP_SAMPLING_RATE), "-ab", "13000", "-ac", "1", "-f", "gsm"]
            dec_opts = ["-c:a", "libgsm", "-f", "wav", "-ar", str(OUTPUT_SAMPLING_RATE)]
        elif codec == 'g726':
            intermediate_ext = '.wav'
            enc_opts = ["-acodec", "g726", "-ar", str(TEMP_SAMPLING_RATE), "-f", "g726"]
            dec_opts = ["-f", "g726", "-ac", "1", "-ar", str(TEMP_SAMPLING_RATE), "-f", "wav", "-ar", str(OUTPUT_SAMPLING_RATE)]
        elif codec == 'adpcm':
            intermediate_ext = '.wav'
            enc_opts = ["-acodec", "adpcm_ms", "-ar", str(TEMP_SAMPLING_RATE), "-b:a", "16k", "-f", "wav"]
            dec_opts = ["-acodec", "adpcm_ms", "-f", "wav", "-ar", str(OUTPUT_SAMPLING_RATE)]
        elif codec == 'g723':
            intermediate_ext = '.wav'
            enc_opts = ["-acodec", "g723_1", "-ar", str(TEMP_SAMPLING_RATE), "-ac", "1", "-b:a", "6.3k", "-f", "wav"]
            dec_opts = ["-acodec", "g723_1", "-f", "wav", "-ar", str(OUTPUT_SAMPLING_RATE)]
        elif codec == 'ogg':
            intermediate_ext = '.oga'
            enc_opts = ["-c:a", "libopus", "-ar", str(TEMP_SAMPLING_RATE), "-b:a", "4.5k"]
            dec_opts = ["-c:a", "libopus", "-f", "wav", "-ar", str(OUTPUT_SAMPLING_RATE)]
        else:
            raise ValueError(f"Unknown codec: {codec}")

        tmp_filepath = join(tmp_dir, filename + intermediate_ext)
        
        # Encode
        cmd_enc = ["ffmpeg", "-i", input_filepath] + enc_opts + ["-y", tmp_filepath]
        run_ffmpeg(cmd_enc, f"encoding to {codec}")
        
        # Decode
        cmd_dec = ["ffmpeg"] 
        if codec == 'g726': # Special handling for raw g726 input if needed, but ffmpeg usually needs -f before -i if raw
             # The original code had: ffmpeg -f g726 -ac 1 -ar 8000 -i ...
             # My enc_opts above for decode were put after -i? No, I need to restructure for decode options before/after input.
             # Let's align with original logic.
             pass
        
        # Re-constructing commands based on original logic more carefully
        if codec == 'g726':
             cmd_dec = ["ffmpeg", "-f", "g726", "-ac", "1", "-ar", str(TEMP_SAMPLING_RATE), "-i", tmp_filepath, "-f", "wav", "-ar", str(OUTPUT_SAMPLING_RATE), "-y", output_filepath]
        elif codec == 'gsm':
             # Original: ffmpeg -c:a libgsm -ar 8000 -ac 1 -i ...
             cmd_dec = ["ffmpeg", "-c:a", "libgsm", "-ar", str(TEMP_SAMPLING_RATE), "-ac", "1", "-i", tmp_filepath, "-f", "wav", "-ar", str(OUTPUT_SAMPLING_RATE), "-y", output_filepath]
        else:
             cmd_dec = ["ffmpeg", "-i", tmp_filepath] + dec_opts + ["-y", output_filepath]

        run_ffmpeg(cmd_dec, f"decoding from {codec}")

def process_directory(input_dir, output_dir, codec, dry_run=False):
    makedirs(output_dir, exist_ok=True)
    files = glob(join(input_dir, '*.wav'))
    
    for input_filepath in tqdm(files, desc=f"Simulating {codec}"):
        filename = basename(input_filepath)
        output_filepath = join(output_dir, filename)
        simulate_codec(input_filepath, output_filepath, codec, dry_run)

def main():
    parser = argparse.ArgumentParser(description="Simulate codec artifacts.")
    parser.add_argument('-i', '--input', default='input', help='Input folder (WAV)')
    parser.add_argument('-o', '--output', default='output', help='Output folder')
    parser.add_argument('-c', '--codec', default='mp3', choices=['mp3', 'gsm', 'g726', 'adpcm', 'g723', 'ogg'], help='Codec to simulate')
    parser.add_argument('--dry-run', action='store_true', help='Simulate execution')
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    
    process_directory(args.input, args.output, args.codec, args.dry_run)

if __name__ == "__main__":
    main()
