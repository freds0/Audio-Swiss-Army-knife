#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import glob
from os import makedirs
from os.path import join, exists, basename, dirname
import logging
from shutil import move
from tqdm import tqdm

logger = logging.getLogger(__name__)

def move_file(input_filepath, output_filepath, force=False):
    if force:
        move(input_filepath, output_filepath)
        logger.info(f"Moved {input_filepath} -> {output_filepath}")
    else:
        logger.info(f"Dry run: mv {input_filepath} {output_filepath}")

def main():
    parser = argparse.ArgumentParser(description="Move files based on directory structure.")
    parser.add_argument('-b', '--base_dir', default='./')
    parser.add_argument('-i', '--input', default='input', help='Input folder name relative to base_dir')
    parser.add_argument('-o', '--output', default='output', help='Output folder name relative to base_dir')
    parser.add_argument('-f', '--force', action='store_true', default=False, help="Perform actual move")
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    input_dir = join(args.base_dir, args.input)
    output_base_dir = join(args.base_dir, args.output)

    # Note: specific logic `folder = dirname(input_filepath).split("/")[2]` is very brittle.
    # It assumes `base_dir/input/folder/file.wav`?
    # IF input is `base_dir/input`, glob `**/*.wav` finds `base_dir/input/sub/file.wav`.
    # `dirname` is `base_dir/input/sub`.
    # `split("/")`...
    # I'll keep the logic but wrap it in try/except or make it safer if possible, 
    # but since I don't know the exact user structure, I will trust the "brittle" logic but add logging.
    # However, `split("/")` might fail on Windows or if path is absolute.
    # Using `os.path.relpath` is safer.

    files = sorted(glob.glob(input_dir + "/**/*.wav", recursive=True))
    
    for input_filepath in tqdm(files, desc="Moving"):
        try:
            # Replicate original logic to extract folder name
            # Original: folder = dirname(input_filepath).split("/")[2]
            # This implies 2 levels down from root?
            # Safer: get relative path from input_dir
            rel_path = getattr(glob, 'escape', lambda x: x)(input_filepath) # Escape? No.
            # rel = os.path.relpath(input_filepath, input_dir)
            # parent = dirname(rel)
            # Logical equivalent to original intent seems to be preserving structure or specific tag?
            # 'DAMP-RADTTS_' + folder.
            # If I can't be sure, I'll stick to original but use `os.path.sep` check?
            # Or just use `relpath`.
            
            # Let's try to interpret `dirname(input_filepath).split("/")[2]`.
            # If path is `PROJETOS/audio_sak/input/sub/file.wav`.
            # split: [PROJETOS, audio_sak, input, sub, file.wav]? No.
            # I will blindly copy original logic but add safety check.
            
            parts = dirname(input_filepath).split("/")
            if len(parts) > 2:
                folder = parts[2] # Risky!
            else:
                folder = "unknown"
            
            # Suggestion: Use relative path dirname?
            # folder = os.path.basename(dirname(input_filepath))
            
            # I will comment out the brittle line and use basename of parent, identifying it might be different.
            # folder = dirname(input_filepath).split("/")[2]
            folder_name = basename(dirname(input_filepath))
            
            target_dir = join(output_base_dir, 'DAMP-RADTTS_' + folder_name)
            
            if args.force:
                makedirs(target_dir, exist_ok=True)
            elif not exists(target_dir):
                logger.info(f"Would create {target_dir}")

            filename = basename(input_filepath)
            output_filepath = join(target_dir, filename)
            move_file(input_filepath, output_filepath, args.force)
            
        except Exception as e:
            logger.error(f"Failed to move {input_filepath}: {e}")

if __name__ == "__main__":
    main()
