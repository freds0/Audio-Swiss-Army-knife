#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import torch
import torchaudio
import logging
import os
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from os.path import join, exists, basename
from os import makedirs
from tqdm import tqdm
from glob import glob

logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_name="wav2vec2-xls-r-300m"):
    model_path = None
    if (model_name == "wav2vec2-xls-r-300m"):
        model_path = "facebook/wav2vec2-xls-r-300m"
    elif (model_name == "wav2vec2-xls-r-1b"):
        model_path = "facebook/wav2vec2-xls-r-1b" 
    elif (model_name == "wav2vec2-xls-r-2b"):
        model_path = "facebook/wav2vec2-xls-r-2b"  
    elif (model_name == "wav2vec2-base-100h"):
        model_path = "facebook/wav2vec2-base-100h" 
    elif (model_name == "wav2vec2-base-960h"):
        model_path = "facebook/wav2vec2-base-960h"
    elif (model_name == "wav2vec2-large-xlsr-53"):
        model_path = "facebook/wav2vec2-large-xlsr-53"
    elif (model_name == "wav2vec2-large"):
        model_path = "facebook/wav2vec2-large" 
    elif (model_name == "wav2vec2-large-robust"):
        model_path = "facebook/wav2vec2-large-robust" 
    
    logger.info(f"Loading {model_path}...")
    model = Wav2Vec2Model.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
    return model, feature_extractor


def extract_wav2vec_embeddings(filelist, output_dir, model_name):
    try:
        model, feature_extractor = load_model(model_name)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    for filepath in tqdm(filelist, desc="Extracting"):
        # Load audio file
        if not exists(filepath):
            logger.warning("file {} doesnt exist!".format(filepath))
            continue
            
        try:
            filename = basename(filepath)
            audio_data, sr = torchaudio.load(filepath)
            
            if sr != 16000:
                audio_data = torchaudio.functional.resample(audio_data, sr, 16000)
            
            audio_data = audio_data.to(device)
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(dim=0, keepdim=True)

            # Extract Embedding
            inputs = feature_extractor(
                audio_data.squeeze(), sampling_rate=16000, return_tensors="pt"
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
    parser = argparse.ArgumentParser(description="Extract Wav2Vec2 embeddings.")
    parser.add_argument('-b', '--base_dir', default='./')
    parser.add_argument('-i', '--input_dir', help='Wavs folder')
    parser.add_argument('-c', '--input_csv', help='Metadata filepath')
    parser.add_argument('-o', '--output_dir', default='output_embeddings', help='Output folder')
    parser.add_argument('-m', '--model_name', default="wav2vec2-base-960h",
                        help="Available models")
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

    extract_wav2vec_embeddings(filelist, output_dir, args.model_name)


if __name__ == "__main__":
    main()
