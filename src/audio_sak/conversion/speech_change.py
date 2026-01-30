#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import io
import logging
import torch
import torchaudio
import pyrubberband as pyrb
from pydub import AudioSegment
from pydub.effects import speedup

logger = logging.getLogger(__name__)

class SpeedChange:
    """
    Change the speed of an audio signal.
    Uses pyrubberband for slowing down (time stretching) and pydub for speeding up.
    """
    def __init__(self, orig_freq=16000):
        self.orig_freq = orig_freq

    def __call__(self, wav, speed_factor):
        wav_np = wav.squeeze().numpy()
        
        if speed_factor < 1.0:
            # Slow down using pyrubberband (time stretch)
            try:
                # pyrubberband expects numpy array
                speeded_wav = pyrb.time_stretch(y=wav_np, sr=self.orig_freq, rate=speed_factor)
            except Exception as e:
                logger.error(f"pyrubberband failed: {e}")
                raise
        else:
            # Speed up using pydub
            try:
                audio_segment = self._convert_nparray_to_audio_segment(wav_np)
                speeded_audio_segment = speedup(audio_segment, speed_factor)
                speeded_wav = self._convert_audio_segment_to_nparray(speeded_audio_segment)
            except Exception as e:
                logger.error(f"pydub failed: {e}")
                raise

        # Convert back to torch tensor
        return torch.tensor(speeded_wav, dtype=torch.float32).unsqueeze(0)

    def _convert_nparray_to_audio_segment(self, data):
        buffer_ = io.BytesIO()
        torchaudio.save(uri=buffer_, src=torch.from_numpy(data).unsqueeze(0), sample_rate=self.orig_freq, format="wav")
        buffer_.seek(0)
        return AudioSegment.from_file(buffer_, format="wav")

    def _convert_audio_segment_to_nparray(self, audio_segment):
        buffer_ = io.BytesIO()
        audio_segment.export(buffer_, format="wav")
        # Rewind buffer might be handled by torchaudio.load but better be safe if needed. 
        # Actually torchaudio.load accepts file-like object.
        buffer_.seek(0) 
        waveform, _ = torchaudio.load(buffer_)
        return waveform.squeeze().numpy()

def main():
    parser = argparse.ArgumentParser(description='Change audio speed.')
    parser.add_argument('-i', '--input', required=True, help='Input WAV filepath.')
    parser.add_argument('-o', '--output', required=True, help='Output WAV filepath.')
    parser.add_argument('-s', '--speed', type=float, required=True, help='Speed factor (0.5 = half speed, 2.0 = double speed).')
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    try:
        waveform, sr = torchaudio.load(args.input)
        speed_function = SpeedChange(orig_freq=sr)
        transformed_waveform = speed_function(waveform, args.speed)
        torchaudio.save(uri=args.output, src=transformed_waveform, sample_rate=sr)
        logger.info(f"Saved to {args.output}")
    except Exception as e:
        logger.error(f"Failed to process: {e}")

if __name__ == '__main__':
    main()
