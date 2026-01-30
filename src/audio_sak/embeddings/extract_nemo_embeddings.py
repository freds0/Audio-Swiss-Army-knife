#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import torch
import logging
import os
from os.path import join, exists, basename
from os import makedirs
from tqdm import tqdm
from glob import glob

try:
    import nemo.collections.asr as nemo_asr
except ImportError:
    nemo_asr = None

logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_name="speakernet"):
    if nemo_asr is None:
        logger.error("nemo_toolkit not installed.")
        return None
        
    speaker_model = None
    try:
        if (model_name == "speakernet"):
            speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
                model_name="speakerverification_speakernet"
            )
        elif (model_name == "titanet"):
            speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
                model_name="titanet_large"
            )
        
        if speaker_model:
            speaker_model.to(device) # Ensure device usage? NeMo usually handles it if configured.
            
    except Exception as e:
        logger.error(f"Failed to load VAD model {model_name}: {e}")
        
    return speaker_model


def extract_nemo_embeddings(filelist, output_dir, model_name):
    model = load_model(model_name)
    if model is None:
        return

    os.makedirs(output_dir, exist_ok=True)
    
    for filepath in tqdm(filelist, desc="Extracting Embeddings"):
        if not exists(filepath):
            logger.warning("file {} doesnt exist!".format(filepath))
            continue
        
        try:
            filename = basename(filepath)
            # NeMo get_embedding usually expects path or signal.
            file_embedding = model.get_embedding(filepath)
            # Ensure it's tensor or numpy
            if isinstance(file_embedding, torch.Tensor):
                embedding = file_embedding.cpu().detach()
            else:
                 embedding = torch.tensor(file_embedding)
            
            # Saving embedding
            output_filename = filename.split(".")[0] + ".pt"
            output_filepath = join(output_dir, output_filename)
            torch.save(embedding, output_filepath)
        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Extract NeMo embeddings.")
    parser.add_argument('-b', '--base_dir', default='./')
    parser.add_argument('-i', '--input_dir', help='Wavs folder')
    parser.add_argument('-c', '--input_csv', help='Metadata filepath')
    parser.add_argument('-o', '--output_dir', default='output_embeddings', help='Output folder')
    parser.add_argument('-m', '--model_name', default="speakernet", help='Available Models: speakernet and titanet.')
    
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
            filelist = [line.split(",")[0] for line in content_file]
    else:
        logger.error("Error: args input_dir or input_csv are necessary!")
        return

    extract_nemo_embeddings(filelist, output_dir, args.model_name)


if __name__ == "__main__":
    main()