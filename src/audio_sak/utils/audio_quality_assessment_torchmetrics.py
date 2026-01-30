#!/usr/bin/env python
# -*- coding: utf-8 -*-
#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
pip install torchmetrics[audio]
pip install pesq pystoi
'''
import argparse
import logging
import torch
import torchaudio
from torchmetrics.audio import ShortTimeObjectiveIntelligibility as stoi
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio as si_sdr
from torchmetrics.audio import PerceptualEvaluationSpeechQuality as pesq
from torchaudio.transforms import Resample
from glob import glob
from os import path

logger = logging.getLogger(__name__)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
target_sr = 16000

# Initialize metrics (lazy init might be better but script usage implies instant run)
try:
    stoi_metric = stoi(target_sr).to(device)
    pesq_metric = pesq(target_sr, 'wb').to(device)
    si_sdr_metric = si_sdr().to(device)
except Exception as e:
    logger.warning(f"Could not initialize metrics: {e}")

def audio_quality(clean_path, noisy_path):   
    clean_waveform, sr = torchaudio.load(clean_path)
    noisy_waveform, _ = torchaudio.load(noisy_path)

    if clean_waveform.shape[0] == 2:
        clean_waveform = torch.mean(clean_waveform, dim=0, keepdim=True)
    if noisy_waveform.shape[0] == 2:
        noisy_waveform = torch.mean(noisy_waveform, dim=0, keepdim=True)

    if sr != target_sr:
        fn_resample = Resample(orig_freq=sr, new_freq=target_sr)
        clean_waveform = fn_resample(clean_waveform)
        noisy_waveform = fn_resample(noisy_waveform)

    stoi_hyp = stoi_metric(noisy_waveform.to(device), clean_waveform.to(device)).item()
    pesq_hyp = pesq_metric(noisy_waveform.to(device), clean_waveform.to(device)).item()
    si_sdr_hyp = si_sdr_metric(noisy_waveform.to(device), clean_waveform.to(device)).item()

    return stoi_hyp, pesq_hyp, si_sdr_hyp

def main():
    parser = argparse.ArgumentParser(description="Calculate audio quality metrics (STOI, PESQ, SI-SDR).")
    parser.add_argument('-c', '--clean', default='clean', help='Clean audio folder')
    parser.add_argument('-n', '--noisy', default='noisy', help='Noisy audio folder')
    parser.add_argument('-o', '--output', default='output_quality.csv', help='Output filepath')
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # Note: original script code: `stoi_hyp, ... = audio_quality(args.clean, args.noisy)` 
    # IMPLIED that clean/noisy are PATHS to FILES? But arguments help says "folder"?
    # The original main code only called it ONCE with folder paths?
    # But `torchaudio.load` takes a file path, not folder.
    # So original code was likely broken for folders or intended for single file test?
    # BUT logic `filename = args.clean` then write to csv suggests it was one-shot.
    # AND commented out logic `clean_files = glob...`.
    # I will support single file or folder?
    # Argument names suggest "folder" but code suggests file.
    # I'll check if inputs are files or dirs and act accordingly.
    
    if path.isdir(args.clean) and path.isdir(args.noisy):
        # Folder mode
        clean_files = sorted(glob(path.join(args.clean, '*.wav')))
        # Assumes matching filenames
        with open(args.output, 'w') as f:
            f.write("filepath,stoi,pesq,si-sdr\n")
            
        for c_file in clean_files:
            fname = path.basename(c_file)
            n_file = path.join(args.noisy, fname)
            if path.exists(n_file):
                try:
                    s, p, si = audio_quality(c_file, n_file)
                    f.write(f"{fname},{s},{p},{si}\n")
                    f.flush()
                except Exception as e:
                    logger.error(f"Error processing {fname}: {e}")
            else:
                logger.warning(f"No match for {fname} in noisy folder")
                
    elif path.isfile(args.clean) and path.isfile(args.noisy):
         # File mode
        with open(args.output, 'w') as f:
             f.write("filepath,stoi,pesq,si-sdr\n")
             s, p, si = audio_quality(args.clean, args.noisy)
             f.write(f"{args.clean},{s},{p},{si}\n")

if __name__ == "__main__":
    main()

