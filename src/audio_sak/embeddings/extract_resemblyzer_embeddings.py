#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
import torch
import torchaudio
import os
from os.path import join, exists, basename
from os import makedirs
from tqdm import tqdm
from glob import glob

try:
    from resemblyzer import VoiceEncoder, preprocess_wav
except ImportError:
    VoiceEncoder = None
    preprocess_wav = None

logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def extract_embeddings(filelist, output_dir):
    '''
    Embeddings from: Generalized End-To-End Loss for Speaker Verification 
    '''
    if VoiceEncoder is None:
        logger.error("resemblyzer not installed.")
        return

    encoder = VoiceEncoder()

    for filepath in tqdm(filelist, desc="Extracting"):
        # Load audio file
        if not exists(filepath):
            logger.warning("file {} doesnt exist!".format(filepath))
            continue
        try:
            filename = basename(filepath)
            wav = preprocess_wav(filepath)
            file_embedding = encoder.embed_utterance(wav)
            embedding = torch.tensor(file_embedding)

            # Saving embedding
            output_filename = filename.split(".")[0] + ".pt"
            output_filepath = join(output_dir, output_filename)
            torch.save(embedding, output_filepath)
        except Exception as e:
            logger.error(f"Failed to process {filepath}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Extract Resemblyzer embeddings.")
    parser.add_argument('-i', '--input', default="input", help='Input folder')
    parser.add_argument('-o', '--output', default='output', help='Output folder')
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    filelist = glob(join(args.input, '*.wav'))

    makedirs(args.output, exist_ok=True)
    extract_embeddings(filelist, args.output)


if __name__ == "__main__":
    main()
