#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import torch
import torchaudio
import logging
import os
from os.path import join, exists, basename
from os import makedirs
from tqdm import tqdm
from glob import glob

try:
    from TTS.tts.utils.speakers import SpeakerManager
except ImportError:
    SpeakerManager = None

logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_cuda = torch.cuda.is_available()

def load_model(model_path, config_path):
    if SpeakerManager is None:
        logger.error("TTS not installed (coqui-tts).")
        return None
        
    try:
        encoder_manager = SpeakerManager(
            encoder_model_path=model_path,
            encoder_config_path=config_path,
            d_vectors_file_path=None,
            use_cuda=use_cuda,
        )
        return encoder_manager
    except Exception as e:
        logger.error(f"Failed to load Clova model: {e}")
        return None


def extract_clova_embeddings(filelist, model_path, config_path, output_dir):
    encoder_manager = load_model(model_path, config_path)
    if encoder_manager is None:
        return

    os.makedirs(output_dir, exist_ok=True)
    
    for filepath in tqdm(filelist, desc="Extracting"):
        # Load audio file
        if not exists(filepath):
            logger.warning("file {} doesnt exist!".format(filepath))
            continue
        try:
            filename = basename(filepath)
            # Extract Embedding
            file_embedding = encoder_manager.compute_embedding_from_clip(filepath)
            file_embedding = torch.as_tensor(file_embedding)
            
            # Saving embedding
            output_filename = filename.split(".")[0] + ".pt"
            output_filepath = join(output_dir, output_filename)
            torch.save(file_embedding, output_filepath)
        except Exception as e:
            logger.error(f"Failed to process {filepath}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Extract Clova embeddings.")
    parser.add_argument('-b', '--base_dir', default='./')
    parser.add_argument('-i', '--input_dir', help='Wavs folder')
    parser.add_argument('-c', '--input_csv', help='Metadata filepath')
    parser.add_argument('--model_path', default='./checkpoints/clova/model_se.pth.tar', help='Model .pth filepath')
    parser.add_argument('--model_config', default='./checkpoints/clova/config_se.json', help='Model config filepath')
    parser.add_argument('-o', '--output_dir', default='output_embeddings', help='Output folder')
    
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

    extract_clova_embeddings(filelist, args.model_path, args.model_config, output_dir)


if __name__ == "__main__":
    main()