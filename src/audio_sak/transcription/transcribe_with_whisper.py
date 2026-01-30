#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
from glob import glob
from os.path import join, basename
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

logger = logging.getLogger(__name__)

class WhisperTranscriptor:
    '''
    Source: https://huggingface.co/openai/whisper-large-v3
    '''
    def __init__(self, model_id="openai/whisper-large-v3", device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.processor, self.model = self._load_model(model_id)

    def _load_model(self, model_id):
        logger.info(f"Loading Whisper model: {model_id}")
        try:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
            )
            processor = AutoProcessor.from_pretrained(model_id)
            return processor, model.to(self.device)
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise

    def _speech_file_to_array_fn(self, filepath, target_sample_rate=16000):
        waveform, sample_rate = torchaudio.load(filepath)
        if sample_rate != target_sample_rate:
            resampler = T.Resample(sample_rate, target_sample_rate)
            waveform = resampler(waveform)
        return waveform.squeeze()

    def transcribe(self, input_filepath):
        try:
            waveform = self._speech_file_to_array_fn(input_filepath)
            input_features = self.processor(waveform, sampling_rate=16000, return_tensors="pt").input_features
            input_features = input_features.to(self.device).to(self.torch_dtype)

            with torch.no_grad():
                predicted_ids = self.model.generate(input_features)

            text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
            return text[0]
        except Exception as e:
            logger.error(f"Error transcribing {input_filepath}: {e}")
            return ""

    def transcribe_folder(self, input_dir, output_file):
        files = sorted(glob(join(input_dir, '*.wav')))
        logger.info(f"Transcribing {len(files)} files to {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as ofile:
            for input_filepath in tqdm(files, desc="Transcribing"):
                transcription = self.transcribe(input_filepath)
                line = "{}|{}".format(input_filepath, transcription.strip())
                ofile.write(line + "\n")

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio using Whisper.")
    parser.add_argument('--input_dir', default='samples/noisy', help='Wavs folder')
    parser.add_argument('--output_file', default='transcription_whisper.csv', help='CSV output file')
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    asr_model = WhisperTranscriptor()
    asr_model.transcribe_folder(args.input_dir, args.output_file)

if __name__ == "__main__":
    main()

