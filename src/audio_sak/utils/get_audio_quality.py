#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
from os.path import join, dirname
import torchaudio
import torchaudio.functional as F
try:
    from torchaudio.pipelines import SQUIM_OBJECTIVE
    objective_model = SQUIM_OBJECTIVE.get_model()
except ImportError:
    SQUIM_OBJECTIVE = None
    objective_model = None

from tqdm import tqdm
import concurrent.futures

logger = logging.getLogger(__name__)

# Global input_dir needed for process_audio_file if using map with string arg
# I'll encapsulate in class or use partial.
GLOBAL_INPUT_DIR = ""

def get_audio_quality(input_filepath):
    if objective_model is None:
        logger.error("SQUIM_OBJECTIVE not found.")
        return 0, 0, 0
        
    try:
        waveform, sr = torchaudio.load(input_filepath)
    except Exception as e:
        logger.error(f"Error loading file {input_filepath}: {e}")
        return 0, 0, 0

    if sr != 16000:
        waveform = F.resample(waveform, sr, 16000)

    stoi_hyp, pesq_hyp, si_sdr_hyp = objective_model(waveform)
    return stoi_hyp[0].item(), pesq_hyp[0].item(), si_sdr_hyp[0].item()

def write_metrics(output_file, wav_filename, stoi, pesq, si_sdr):
    line = f"{wav_filename}|{stoi}|{pesq}|{si_sdr}\n"
    output_file.write(line)
    output_file.flush()

def process_audio_file(input_data):
    # Expected format: wav_filename|...
    parts = input_data.strip().split("|")
    wav_filename = parts[0]
    
    input_filepath = join(GLOBAL_INPUT_DIR, wav_filename)
    stoi, pesq, si_sdr = get_audio_quality(input_filepath)
    return wav_filename, stoi, pesq, si_sdr

def main():
    parser = argparse.ArgumentParser(description="Calculate audio quality metrics from metadata CSV.")
    parser.add_argument('--input', '-i', type=str, default='metadata.csv', help='Input metadata CSV')
    parser.add_argument('--output', '-o', type=str, default='metrics.csv', help='Output metrics CSV')
    parser.add_argument('--num_threads', '-n', type=int, default=4)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    global GLOBAL_INPUT_DIR
    GLOBAL_INPUT_DIR = dirname(args.input)

    try:
        with open(args.input, 'r') as infile:
            lines = infile.readlines()
            
        if len(lines) < 2:
            logger.error("Input CSV is empty or missing header.")
            return
            
        input_data = lines[1:]
        
        with open(args.output, "w") as ofile:
            ofile.write("wav_filename|stoi|pesq|si_sdr\n")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
                results = list(tqdm(executor.map(process_audio_file, input_data), total=len(input_data)))
            
            for result in results:
                wav_filename, stoi, pesq, si_sdr = result
                write_metrics(ofile, wav_filename, stoi, pesq, si_sdr)
                
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    main()

