#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
import os
import time
import random
from tqdm import tqdm
try:
    import yt_dlp
except ImportError:
    yt_dlp = None

logger = logging.getLogger(__name__)

def execute_download(youtube_link, output_dir):
    if yt_dlp is None:
        logger.error("yt_dlp not installed. Please install it.")
        raise ImportError("yt_dlp not installed")
        
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_link])
        logger.info(f"Downloaded audio from {youtube_link}")
    except Exception as e:
        logger.error(f"Failed to download {youtube_link}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download audio from YouTube using yt-dlp.")
    parser.add_argument('--base_dir', default='./')
    parser.add_argument('--input_file', default='links.txt', help='Text file with YouTube links')
    parser.add_argument('--output_dir', default='audios', help='Output folder')
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    
    input_file = os.path.join(args.base_dir, args.input_file)
    output_folder = os.path.join(args.base_dir, args.output_dir)
    
    if not os.path.exists(input_file):
        logger.error(f"Input file {input_file} not found.")
        return

    with open(input_file, encoding="utf-8") as f:
        content_file = [line.strip() for line in f if line.strip()]

    os.makedirs(output_folder, exist_ok=True)
    
    for yt_link in tqdm(content_file, desc="Processing Links"):
        wait_time = random.randint(5, 15) # Reduced wait time, yt-dlp is usually better handled, but let's be safe.
        logger.info(f"Downloading {yt_link} (Wait {wait_time}s)...")
        time.sleep(wait_time)
        execute_download(yt_link, output_folder)

if __name__ == "__main__":
    main()

