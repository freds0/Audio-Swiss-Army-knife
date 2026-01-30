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
from transformers import AutoProcessor, Wav2Vec2ForCTC

logger = logging.getLogger(__name__)

class MmsTranscriptor:
    '''
    Source: https://huggingface.co/facebook/mms-1b-fl102
    '''
    def __init__(self, model_name='fl102', lang="eng", device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Original logic maps 'en' to 'eng' but what about others?
        # MMS uses ISO 3 codes.
        if lang == "en":
            lang = "eng"
        self.processor, self.model = self._load_model(model_name, lang)

    def _load_model(self, model_name='fl102', lang="eng"):
        models = {
            "fl102": "facebook/mms-1b-fl102",
            "l1107": "facebook/mms-1b-l1107",
            "all": "facebook/mms-1b-all"
        }
        model_id = models.get(model_name, "facebook/mms-1b-fl102")
        logger.info(f"Loading MMS model: {model_id} (lang={lang})")

        try:
            processor = AutoProcessor.from_pretrained(model_id)
            model = Wav2Vec2ForCTC.from_pretrained(model_id)

            if lang:
                processor.tokenizer.set_target_lang(lang)
                model.load_adapter(lang)

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
            inputs = self.processor(waveform, sampling_rate=16000, return_tensors="pt")
            inputs = inputs.to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs).logits

            ids = torch.argmax(outputs, dim=-1)[0]
            text = self.processor.decode(ids)
            return text
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
    parser = argparse.ArgumentParser(description="Transcribe audio using MMS.")
    parser.add_argument('-i', '--input_dir', default='samples/noisy', help='Wavs folder')
    parser.add_argument('-o', '--output_file', default='transcription_mms.csv', help='CSV output file')
    parser.add_argument('-m', '--model', default='fl102', help='Model version: fl102 | l1107 | all')      
    parser.add_argument('-l', '--lang', default='eng', help='Language code (ISO 639-3)')
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    asr_model = MmsTranscriptor(args.model, args.lang)
    asr_model.transcribe_folder(args.input_dir, args.output_file)

if __name__ == "__main__":
    main()
