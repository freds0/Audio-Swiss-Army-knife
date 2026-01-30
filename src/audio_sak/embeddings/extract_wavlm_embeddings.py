#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import torch
import torchaudio
import logging
import os
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
from os.path import join, exists, basename
from os import makedirs
from tqdm import tqdm
from glob import glob

logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_name="wavlm-large"):
    model_path = None
    if (model_name == "wavlm-large"): 
        model_path = "microsoft/wavlm-large"
    elif (model_name == "wavlm-base-plus"): 
        model_path = "microsoft/wavlm-base-plus"
    elif (model_name == "wavlm-base-plus-sv"): 
        model_path = "microsoft/wavlm-base-plus-sv"
    
    logger.info(f"Loading {model_path}...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    model = WavLMModel.from_pretrained(model_path)
    model.eval()
    model = model.to(device)
    return model, feature_extractor


def extract_wavlm_embeddings(filelist, output_dir, model_name):
    try:
        model, feature_extractor = load_model(model_name)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    for filepath in tqdm(filelist, desc="Extracting"):
        if not exists(filepath):
            logger.warning("file {} doesnt exist!".format(filepath))
            continue
        try:
            filename = basename(filepath)
            audio_data, sr = torchaudio.load(filepath)
            
            if sr != 16000:
                audio_data = torchaudio.functional.resample(audio_data, sr, 16000)

            audio_data = audio_data.squeeze().to(device)
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(dim=0)

            # Extract Embedding
            inputs = feature_extractor(
                audio_data, sampling_rate=16000, return_tensors="pt"
            )
            input_features = inputs.input_values.to(device)
            
            with torch.no_grad():
                file_embedding = model(input_features).last_hidden_state

            # Saving embedding
            output_filename = filename.split(".")[0] + ".pt"
            output_filepath = join(output_dir, output_filename)
            torch.save(file_embedding.cpu(), output_filepath)
        except Exception as e:
            logger.error(f"Failed to process {filepath}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Extract WavLM embeddings.")
    parser.add_argument('-b', '--base_dir', default='./')
    parser.add_argument('-i', '--input_dir', help='Wavs folder')
    parser.add_argument('-c', '--input_csv', help='Metadata filepath')
    parser.add_argument('-o', '--output_dir', default='output_embeddings', help='Output folder')
    parser.add_argument('-m', '--model_name', default="wavlm-large", help='Available Models: wavlm-large | wavlm-base-plus | wavlm-base-plus-sv')
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    output_dir = join(args.base_dir, args.output_dir)

    filelist = []
    if (args.input_dir is not None):
        input_dir = join(args.base_dir, args.input_dir)
        filelist = glob(input_dir + '/*.wav')

    elif (args.input_csv is not None):
        with open(join(args.base_dir, args.input_csv), encoding="utf-8") as f:
            content_file = f.readlines()
            filelist = [line.split(",")[0].strip() for line in content_file if line.strip()]
    else:
        logger.error("Error: args input_dir or input_csv are necessary!")
        return

    extract_wavlm_embeddings(filelist, output_dir, args.model_name)


if __name__ == "__main__":
    main()
