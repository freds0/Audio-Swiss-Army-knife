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
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

logger = logging.getLogger(__name__)

class Wav2Vec2Transcriptor:
    '''
    Source: https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53
    '''
    def __init__(self, lang="en", device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor, self.model = self._load_model(lang)

    def _load_model(self, language_id):
        models = {
            'sp': "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
            'it': "jonatasgrosman/wav2vec2-large-xlsr-53-italian",
            'ge': "jonatasgrosman/wav2vec2-large-xlsr-53-german",
            'pl': "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
            'pt': "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
            'en': "jonatasgrosman/wav2vec2-large-xlsr-53-english",
            'du': "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
            'fr': "jonatasgrosman/wav2vec2-large-xlsr-53-french",
        }
        
        model_id = models.get(language_id, "jonatasgrosman/wav2vec2-large-xlsr-53-english")
        logger.info(f"Loading Wav2Vec2 model: {model_id}")

        try:
            processor = Wav2Vec2Processor.from_pretrained(model_id)
            model = Wav2Vec2ForCTC.from_pretrained(model_id)
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
            waveform = self._speech_file_to_array_fn(input_filepath, 16000)
            input_values = torch.tensor(waveform, device=self.device).unsqueeze(0)

            with torch.no_grad():
                logits = self.model(input_values).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            text = self.processor.batch_decode(predicted_ids)
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
    parser = argparse.ArgumentParser(description="Transcribe audio using Wav2Vec2.")
    parser.add_argument('--input_dir', default='samples/noisy', help='Wavs folder')
    parser.add_argument('--output_file', default='transcription_wav2vec.csv', help='CSV output file')
    parser.add_argument('--lang', default='en', help='Language code: du, en, fr, ge, it, pl, pt')
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    asr_model = Wav2Vec2Transcriptor(lang=args.lang)
    asr_model.transcribe_folder(args.input_dir, args.output_file)

if __name__ == "__main__":
    main()