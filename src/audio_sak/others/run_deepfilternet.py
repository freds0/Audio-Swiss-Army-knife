#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
import subprocess
import os
from tqdm import tqdm

logger = logging.getLogger(__name__)

def execute(input_dir, output_dir):
    if not os.path.exists(input_dir):
        logger.error(f"Input {input_dir} not found.")
        return

    # Check if input is a directory of folders or directory of files?
    # Original code: folders = listdir(input_dir), loops folders.
    # Checks if input_dir contains folders.
    try:
        folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
        
        # If no folders, maybe it's just files?
        # Original code assumed folders structure: input_dir/folder/file.wav.
        # "folders = listdir(input_dir)" -> "in_dir = join(input_dir, folder)".
        # "command = ... --noisy-dir {in_dir}"
        # If input_dir contains wav files directly, "deepFilter" usage might differ.
        # I'll stick to original logic but add safety.
        
        if not folders:
            logger.info("No subdirectories found. Processing input directory itself.")
            # Process input_dir directly?
            # command = deepFilter ...
            os.makedirs(output_dir, exist_ok=True)
            command = f"deepFilter --noisy-dir {input_dir} --output-dir={output_dir}"
            logger.info(f"Running: {command}")
            subprocess.run(command, shell=True)
            return

        os.makedirs(output_dir, exist_ok=True)
        for folder in tqdm(folders, desc="Processing Folders"):
            in_dir = os.path.join(input_dir, folder)
            out_dir = os.path.join(output_dir, folder)
            os.makedirs(out_dir, exist_ok=True)
            
            command = f"deepFilter --noisy-dir {in_dir} --output-dir={out_dir}"
            logger.debug(f"Running: {command}")
            subprocess.run(command, shell=True)
            
    except Exception as e:
        logger.error(f"Execution failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run DeepFilterNet on audio folders.")
    parser.add_argument('--input', default='input', help='Input directory')
    parser.add_argument('--output', default='output', help='Output directory')
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    
    execute(args.input, args.output)

if __name__ == "__main__":
    main()
 
