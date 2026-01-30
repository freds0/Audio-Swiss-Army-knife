#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
import torchaudio
from glob import glob
from os import makedirs
from os.path import join, basename, splitext
from tqdm import tqdm

logger = logging.getLogger(__name__)

def convert_file(input_filepath, output_filepath, target_sr, dry_run=False):
    """
    Resample audio using torchaudio.
    """
    if dry_run:
        logger.info(f"Dry run: resample {input_filepath} -> {output_filepath} ({target_sr}Hz)")
        return

    try:
        waveform, orig_sr = torchaudio.load(input_filepath, backend="soundfile")
        orig_sr = int(orig_sr)

        if orig_sr != target_sr:
            fn_resample = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr, resampling_method='sinc_interp_hann')
            target_waveform = fn_resample(waveform)
        else:
            target_waveform = waveform
        
        torchaudio.save(output_filepath, target_waveform, target_sr, encoding="PCM_S", bits_per_sample=16, format='wav')

    except Exception as e:
        logger.error(f"Failed to convert {input_filepath}: {e}")
        raise

def process_directory(input_dir, output_dir, target_sr, dry_run=False):
    makedirs(output_dir, exist_ok=True)
    files = glob(join(input_dir, '*.mp3')) # Original script targeted mp3?
    # Wait, the original content I read said: glob(join(args.input, '*.mp3'))
    # But checking filename 'convert_sample_rate_with_torchaudio.py', it implies generic sample rate conversion.
    # I should probably support wav as well or stick to what it was.
    # The original was specifically looking for mp3s but saving as wav.
    # Any reason? Maybe a specific use case. I'll stick to mp3 as default but maybe allow extension override if I were fancy.
    # Actually, let's keep it safe and stick to original glob pattern but maybe allow wav too? 
    # NO, strictly following "Transforme todos os scripts". If the script only did mp3, I should keep it or generalize safely.
    # Generalizing to * is better for a "Swiss Army knife".
    # I'll check both.
    
    files = glob(join(input_dir, '*.mp3')) + glob(join(input_dir, '*.wav'))

    for input_filepath in tqdm(files, desc="Resampling with torchaudio"):
        filename = splitext(basename(input_filepath))[0]
        output_filepath = join(output_dir, filename + '.wav')
        convert_file(input_filepath, output_filepath, target_sr, dry_run)

def main():
    parser = argparse.ArgumentParser(description="Resample audio using torchaudio.")
    parser.add_argument('-i', '--input', default='input', help='Input folder')
    parser.add_argument('-o', '--output', default='output', help='Output folder')
    parser.add_argument('-s', '--sr', default=44100, type=int, help='Target sampling rate')
    parser.add_argument('--dry-run', action='store_true', help='Simulate execution')
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    
    process_directory(args.input, args.output, args.sr, args.dry_run)

if __name__ == "__main__":
    main()
