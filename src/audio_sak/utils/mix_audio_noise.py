#!/usr/bin/env python
# -*- coding: utf-8 -*-
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
import torch
import torchaudio
import torchaudio.functional as F
from random import choice, randrange
from os import makedirs
from os.path import join, basename
from tqdm import tqdm
from glob import glob

logger = logging.getLogger(__name__)

def mix_file(input_clean_filepath, input_noise_filepath, output_filepath, force):
    try:
        # Read data
        clean_audio, sr = torchaudio.load(input_clean_filepath)
        noise_audio, _ = torchaudio.load(input_noise_filepath)

        if not force:
            logger.info(f"Dry run: mix {input_clean_filepath} + {input_noise_filepath} -> {output_filepath}")
        else:
            # Handle lengths
            if noise_audio.shape[1] < clean_audio.shape[1]:
                # Repeat noise? Or fail?
                # For now, let's repeat.
                repeats = (clean_audio.shape[1] // noise_audio.shape[1]) + 1
                noise_audio = noise_audio.repeat(1, repeats)
                
            end = randrange(noise_audio.shape[1] - clean_audio.shape[1] + 1)
            begin = randrange(end) if end > 0 else 0
            # Wait, `randrange(end)`? If end is 0?
            # Correct logic:
            valid_start_range = noise_audio.shape[1] - clean_audio.shape[1]
            if valid_start_range > 0:
                start_idx = randrange(valid_start_range)
                noise_clip = noise_audio[:, start_idx : start_idx + clean_audio.shape[1]]
            else:
                 noise_clip = noise_audio[:, :clean_audio.shape[1]]

            snr_dbs = torch.randint(low=-10, high=20, size=(1,))
            mixed_audio = F.add_noise(clean_audio, noise_clip, snr_dbs)

            torchaudio.save(output_filepath, mixed_audio, sr, encoding="PCM_S", bits_per_sample=16, format='wav')
            logger.debug(f"Saved {output_filepath}")
            
    except Exception as e:
        logger.error(f"Failed to mix {input_clean_filepath}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Mix clean audio with noise.")
    parser.add_argument('-i', '--input_clean', default='clean', help='Input clean wavs folder')
    parser.add_argument('-n', '--input_noise', default='noise', help='Input noise wavs folder')
    parser.add_argument('-o', '--output', default='output', help='Output folder')
    parser.add_argument('-f', '--force', action='store_true', default=False, help="Actually perform mixing (default dry run)")
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    noise_list = glob(join(args.input_noise, '*.wav'))
    if not noise_list:
        logger.error(f"No noise files found in {args.input_noise}")
        return

    makedirs(args.output, exist_ok = True)    
    clean_files = glob(join(args.input_clean, '*.wav'))
    
    for input_clean_filepath in tqdm(clean_files, desc="Mixing"):
        output_filepath = join(args.output, basename(input_clean_filepath))
        input_noise_filepath = choice(noise_list)
        mix_file(input_clean_filepath, input_noise_filepath, output_filepath, args.force)

if __name__ == "__main__":
    main()
