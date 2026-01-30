#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
source: https://gist.github.com/johnmeade/0b4c1b6ed3c732f7cb017d9c1d16ab65

Estimate background noise power level of a speech waveform.
Requires some non-speech regions in the wave.
Requirements:
    pip install numpy librosa soundfile webrtcvad
MIT License John Meade 2021
"""

import librosa
import numpy as np
import soundfile as sf
import webrtcvad
import struct
import logging
import argparse

logger = logging.getLogger(__name__)

int16_max = (2 ** 15) - 1

def float_to_pcm16(wav):
    "pack a wav, in [-1, 1] numpy float array format, into 16 bit PCM bytes"
    # Ensure clipping to avoid overflow
    wav = np.clip(wav, -1, 1)
    return struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))

def nonspeech(wav, sr, vad_aggressiveness=3, webrtc_sr=16000, webrtc_window_ms=30, dilations=3):
    """
    Args:
        wav: the waveform, a float numpy array with range [-1, 1].
        sr: sample rate of input wav.
        vad_aggressiveness: must be 1, 2 or 3. Higher is more strict about filtering nonspeech.
        webrtc_sr: must be 8000, 16000, 32000, or 48000. Use a value close to your input.
        webrtc_window_ms: must be 10, 20, or 30ms.
        dilations: number of windows to dilate VAD results. Increase if phonemes are leaking into output.
    Returns:
        wave of all regions with no speech content, concatenated together
    """

    # resample
    if sr != webrtc_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=webrtc_sr)

    # init VAD
    vad = webrtcvad.Vad(vad_aggressiveness)

    # trim wav to integer number of windows
    W = webrtc_window_ms * webrtc_sr // 1000
    T = len(wav)
    rem = T % W
    wav = wav if rem == 0 else wav[:-rem]

    # run VAD
    windows = [wav[i * W:(i+1) * W] for i in range(len(wav) // W)]
    va = [vad.is_speech(float_to_pcm16(win), webrtc_sr) for win in windows]

    # dilate VAD to adjacent frames
    va = np.array(va, dtype=bool)
    for _ in range(dilations):
        va[:-1] |= va[1:]
        va[1:] |= va[:-1]

    # collect all frames without VA, concat into background-audio wave
    bg = [win for win, is_speech in zip(windows, va) if not is_speech]
    if len(bg) == 0:
        return np.array([])
    bg = np.concatenate(bg)
    return bg

def mean_noise_power(wav_fpath):
    # read wav
    wav, sr = sf.read(wav_fpath)

    # normalize
    wav /= abs(wav).max().clip(1e-6)

    # obtain nonspeech samples
    bg = nonspeech(wav, sr)
    
    if len(bg) == 0:
        logger.warning(f"No non-speech detected in {wav_fpath}")
        return float('nan')

    # compute mean power
    return (bg ** 2).mean()

def main():
    parser = argparse.ArgumentParser(description="Estimate mean noise power using WebRTCVAD.")
    parser.add_argument('--input', required=True, help='Path to input wav file')
    parser.add_argument('--nonspeech_output', default=None, help='Path to save non-speech audio (optional)')
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    try:
        power = mean_noise_power(args.input)
        logger.info(f"Mean Noise Power: {power}")
        
        if args.nonspeech_output:
            wav, sr = sf.read(args.input)
            wav /= abs(wav).max().clip(1e-6)
            bg = nonspeech(wav, sr, webrtc_sr=16000)
            if len(bg) > 0:
                sf.write(args.nonspeech_output, bg, 16000)
                logger.info(f"Saved non-speech to {args.nonspeech_output}")
            else:
                logger.warning("No non-speech found, nothing saved.")
                
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == '__main__':
    main()