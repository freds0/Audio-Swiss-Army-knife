#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
import os
import time
import random
from glob import glob
from os.path import join, exists
from tqdm import tqdm
from pydub import AudioSegment
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

logger = logging.getLogger(__name__)

def get_audio_duration(audio_file_path):
    audio = AudioSegment.from_file(audio_file_path)
    return len(audio) / 1000.0  # seconds

class OpenAITranscriptor:
    def __init__(self, api_key=None):
        if OpenAI is None:
            logger.error("openai package not installed. Please install it.")
            raise ImportError("openai package not installed")
            
        self.api_key = api_key if api_key else os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found in environment or arguments.")
            
        self.client = OpenAI(api_key=self.api_key)

    def transcribe(self, audio_file_path, retries=3):
        for attempt in range(retries):
            try:
                with open(audio_file_path, 'rb') as audio_file:
                    transcription = self.client.audio.transcriptions.create(
                        model="whisper-1", 
                        file=audio_file,
                        response_format="text",
                        # language="pt" # Should be arg?
                    )
                return transcription
            except Exception as e:
                logger.warning(f"Error transcribing {audio_file_path} ({attempt + 1}/{retries}): {e}")
                time.sleep(random.randint(1, 3))
        return None

    def transcribe_folder(self, input_dir, output_file, resume=True):
        files = sorted(glob(join(input_dir, '*.flac'))) + sorted(glob(join(input_dir, '*.wav')))
        if not files:
            # Recursive check?
            files = sorted(glob(join(input_dir, '**', '*.flac'), recursive=True)) + sorted(glob(join(input_dir, '**', '*.wav'), recursive=True))
        
        logger.info(f"Found {len(files)} files.")

        processed_files = set()
        if resume and exists(output_file):
            with open(output_file, "r") as f:
                for line in f:
                    parts = line.strip().split("|")
                    if parts:
                        processed_files.add(parts[0])
            logger.info(f"Resuming. {len(processed_files)} already processed.")

        with open(output_file, 'a' if resume else 'w') as ofile:
            for filepath in tqdm(files, desc="Transcribing"):
                if filepath in processed_files:
                    continue

                try:
                    duration = get_audio_duration(filepath)
                    if duration <= 0.1:
                        logger.debug(f"Skipping short file {filepath}")
                        continue

                    text = self.transcribe(filepath)
                    if not text:
                        continue

                    line = "{}|{}".format(filepath, str(text).strip())
                    ofile.write(line + "\n")
                    ofile.flush() # Ensure write
                except Exception as e:
                    logger.error(f"Failed to process {filepath}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio using OpenAI API.")
    parser.add_argument('--input_dir', default='train', help='Input folder')
    parser.add_argument('--output_file', default='openai_transcription.csv', help='CSV output file')
    parser.add_argument('--api_key', default=None, help='OpenAI API Key (optional, defaults to env var)')
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    try:
        transcriptor = OpenAITranscriptor(api_key=args.api_key)
        transcriptor.transcribe_folder(args.input_dir, args.output_file)
    except ImportError:
        logger.error("OpenAI module missing.")
    except Exception as e:
        logger.error(f"Execution failed: {e}")

if __name__ == "__main__":
    main()

