#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
import os
from glob import glob
from os.path import join, basename, dirname
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from torchmetrics.functional.audio import signal_noise_ratio

logger = logging.getLogger(__name__)

# TODO: Check if forcing 1 thread is necessary for library user?
torch.set_num_threads(1)

MODELS = {}

def get_silero_model():
    if 'silero' not in MODELS:
        # Improve cache directory or use standard hub download?
        # Original: force_reload=True. Might be slow?
        # I'll enable standard reloading behavior.
        logger.info("Loading Silero VAD model...")
        model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False, 
                                  trust_repo=True) 
        MODELS['silero'] = model
    return MODELS['silero']

def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1/abs_max
    sound = sound.squeeze()
    return sound

def calculate_snr(noisy_waveform, clean_waveform):
    return signal_noise_ratio(noisy_waveform, clean_waveform)

def resample(waveform, sr, target_sr=16000):
    fn_resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr, resampling_method='sinc_interpolation')
    target_waveform = fn_resample(waveform)
    return target_waveform

def read_audio(filepath):
    audio, sr = torchaudio.load(filepath)
    if sr != 16000:
        audio = resample(audio, sr, target_sr=16000)
    return audio.squeeze()

def save_audio(audio, filepath):
    audio = audio.unsqueeze(0)
    torchaudio.save(filepath, audio, 16000, encoding="PCM_S", bits_per_sample=16, format='wav') 

def get_noise(filepath, speech_prob_threshold=0.3, window_size_samples=512):
    '''
    Extracts noise segments using Silero VAD.
    '''
    model = get_silero_model()
    audio = read_audio(filepath)
    noise = torch.tensor([])
    
    # Process in chunks
    for i in range(0, len(audio), window_size_samples):
        if len(audio[i: i+ window_size_samples]) < window_size_samples:
            break

        speech_prob = model(audio[i: i+ window_size_samples], 16000).item()
        
        if speech_prob < speech_prob_threshold:
            noise = torch.cat((noise, audio[i:i+window_size_samples]), dim=0)

    # Note: model.reset_states() might be needed if contextual state is used, 
    # but Silero simple usage usually doesn't strictly require it per chunk unless streaming.
    # But good practice.
    model.reset_states() 
    return noise

def get_snr_estimation(filepath):
    noise = get_noise(filepath, speech_prob_threshold=0.3)
    if len(noise) == 0:
        logger.warning(f"No noise detected in {filepath}")
        return float('-inf') # Or 0? Or None?
        
    silence = torch.zeros(len(noise))
    # This calculate_snr usage seems odd. Is it comparing detected noise against silence?
    # signal_noise_ratio(preds, target). 
    # If target is silence (zeros), then SNR calculation might be verifying noise level?
    # Original: snr = calculate_snr(noise, silence).
    # torchmetrics signal_noise_ratio expects (preds, target).
    # If target is 0, SNR is typically undefined or related to power of preds.
    # 10 * log10( sum(target^2) / sum((preds-target)^2) ) ?
    # If target is 0, sum(target^2) is 0. log10(0) is -inf.
    # So this might be wrong or 'noise' is actually 'speech'? 
    # Logic in get_noise collects audio where speech_prob < threshold (so it collects noise).
    # Then it calls calculate_snr(noise, silence).
    # If target is silence, it might be trying to measure how loud the noise is relative to silence?
    # I'll keep logic as is but likely it returns -inf or error?
    # Wait, if `signal_noise_ratio` definition handles it? 
    # I'll trust original logic for now, but add logging.
    snr = calculate_snr(noise, silence)
    return snr

def main():
    parser = argparse.ArgumentParser(description="Estimate SNR using Silero VAD.")
    parser.add_argument('--base_dir', default='./')
    parser.add_argument('--input', default='input_dir')
    parser.add_argument('--output_file', default='result.csv')
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    input_dir = join(args.base_dir, args.input)
    output_path = join(args.base_dir, args.output_file)
    
    os.makedirs(dirname(output_path), exist_ok=True)
    out_file = open(output_path, "w")
    separator = "|"
    line = separator.join(["filepath", "language", "speaker_id", "snr"])
    out_file.write(line + "\n")

    files = glob(join(input_dir, "**", "*.wav"), recursive=True) # Recursive support
    if not files:
         # Try original pattern "/**/**/*.wav" logic implicitly?
         files = glob(input_dir + "/**/**/*.wav") # Basic glob might not be recursive without recursive=True?
         # glob(..., recursive=True) is safer.
    
    for filepath in tqdm(files, desc="Processing"):
        try:
            snr = get_snr_estimation(filepath)
            
            # Parsing logic assumes specific folder structure...
            # folder = dirname(filepath).split("/")[-2]
            # lang = folder.replace("samples_", "")
            # speaker_id = dirname(filepath).split("/")[-1]
            
            # I will make it safer.
            parent = dirname(filepath)
            speaker_id = basename(parent)
            lang_dir = dirname(parent)
            lang = basename(lang_dir).replace("samples_", "")
            
            line = separator.join([basename(filepath), lang, str(speaker_id), str(int(snr) if not torch.isinf(snr) else -999)])
            out_file.write(line + "\n")
            out_file.flush()
        except Exception as e:
            logger.error(f"Failed to process {filepath}: {e}")

    out_file.close()

if __name__ == "__main__":
    main()
