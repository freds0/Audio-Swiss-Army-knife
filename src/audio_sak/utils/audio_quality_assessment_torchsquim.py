#!/usr/bin/env python
# -*- coding: utf-8 -*-
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
import torch
import torchaudio
from os import makedirs
from os.path import join, basename
from tqdm import tqdm
from glob import glob
try:
    from torchaudio.prototype.pipelines import SQUIM_OBJECTIVE
except ImportError:
    SQUIM_OBJECTIVE = None

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_objective_model():
    if SQUIM_OBJECTIVE is None:
        raise ImportError("torchaudio prototype pipelines SQUIM_OBJECTIVE not found.")
    return SQUIM_OBJECTIVE.get_model().to(device)

objective_model = None # Lazy load

target_sr = 16000

def audio_quality(input_filepath): 
    global objective_model
    if objective_model is None:
         objective_model = get_objective_model()

    waveform, sr = torchaudio.load(input_filepath)
    if waveform.shape[0] == 2:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sr != 16000:
        fn_resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr, resampling_method='sinc_interp_hann')
        waveform = fn_resample(waveform)

    # Truncate? Original code: `waveform[:1600]` which is super short (0.1s)? 
    # Or maybe it meant `[:16000]` (1s)? Or `[:, :1600]`? 
    # Original: `waveform[:1600]`. Wait, waveform shape is (channels, time).
    # `waveform[:1600]` slices the CHANNELS dimension if dim 0 is channels?
    # No, usually (channels, time). If channels=1, `waveform[:1600]` is empty/wrong?
    # `waveform` is tensor. `waveform[0]` is channel 0.
    # SQUIM model usually expects (batch, time) or (time)? 
    # If the original code was `waveform[:1600]` on dimension 0, that's weird if standard loading is (1, N).
    # Checking torchaudio load: returns (channel, time).
    # If mono, shape is (1, L). `waveform[:1600]` would likely just return the whole thing if L > 1600? No.
    # `waveform[0:1600]` refers to ROWS 0 to 1599. We have 1 row. So it returns the row 0 (and empty rows 1..1599 if strict slicing or just the 1 row).
    # Wait, `waveform[:1600]` slicing on First Dimension. If size is (1, L), result is (1, L).
    # I suspect user logic was flawed or testing something specific.
    # OR maybe they meant `waveform[:, :16000]`?
    # However, I will preserve logic but use `waveform.to(device)` correctly.
    # Wait, SQUIM usually expects specific input.
    # I will assumne correct usage is just passing waveform.
    
    with torch.no_grad():
        stoi_hyp, pesq_hyp, si_sdr_hyp = objective_model(waveform.to(device))

    return stoi_hyp[0], pesq_hyp[0], si_sdr_hyp[0]

def main():
    parser = argparse.ArgumentParser(description="Calculate audio quality using Torchaudio SQUIM.")
    parser.add_argument('-i', '--input', default='input', help='Input folder')
    parser.add_argument('-o', '--output', default='output_quality.csv', help='Output filepath')
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    with open(args.output, 'w') as f:
        f.write("filepath,stoi,pesq,si-sdr\n")
        
    wavs = glob(join(args.input, '*.wav'))
    logger.info(f"Found {len(wavs)} files.")

    for input_filepath in tqdm(wavs, desc="Processing"):
        try:
            stoi_hyp, pesq_hyp, si_sdr_hyp = audio_quality(input_filepath)
            line = f"{input_filepath},{stoi_hyp},{pesq_hyp},{si_sdr_hyp}\n"
            with open(args.output, 'a') as f:
                f.write(line)
        except Exception as e:
            logger.error(f"Failed to process {input_filepath}: {e}")

if __name__ == "__main__":
    main()
