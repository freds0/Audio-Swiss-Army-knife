#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
from glob import glob
from os import makedirs
from os.path import join, basename, exists
from tqdm import tqdm
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

logger = logging.getLogger(__name__)

def remove_silence(path_in, path_out, format="wav", min_silence_len=50, silence_thresh_multiplier=2.5, close_gap=200, min_segment_len=350, dry_run=False):
    if dry_run:
        logger.info(f"Dry run: remove silence {path_in} -> {path_out}")
        return True

    try:
        sound = AudioSegment.from_file(path_in, format=format)
        
        # Calculate threshold relative to dBFS
        silence_thresh = sound.dBFS * silence_thresh_multiplier
        # Ensure thresh is negative enough? sound.dBFS is usually negative.
        # If dBFS is -20, thresh * 2.5 = -50. That's reasonable.
        # Original code used `sound.dBFS * silence_thresh_multiplier`.
        
        non_sil_times = detect_nonsilent(sound, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    except Exception as e:
        logger.error(f"Error processing file {path_in}: {str(e)}")
        return False

    if not non_sil_times:
        logger.info(f"No non-silent segments found in {path_in}. Skipping/Copying?")
        # Original code copied file.
        sound.export(path_out, format='wav')
        return True

    # Combine close segments
    non_sil_times_concat = [non_sil_times[0]]
    for t in non_sil_times[1:]:
        if t[0] - non_sil_times_concat[-1][-1] < close_gap:
            non_sil_times_concat[-1][-1] = t[1]
        else:
            non_sil_times_concat.append(t)

    # Filter by minimum length
    non_sil_segments = [t for t in non_sil_times_concat if t[1] - t[0] > min_segment_len]

    if non_sil_segments:
        # Export from first start to last end? Or concatenate segments?
        # Original code: `sound[non_sil_times[0][0]: non_sil_times[-1][1]].export(...)`
        # This keeps the gap between segments if they were not merged?
        # Actually it takes the range from START of FIRST to END of LAST.
        # This effectively trims LEADING and TRAILING silence only, but keeps internal silence if gaps > close_gap exist!
        # The name "remove_silence" implies removing ALL silence?
        # But looking at logic: `non_sil_times` is list of [start, end].
        # `non_sil_times[0][0]` is start of first sound.
        # `non_sil_times[-1][1]` is end of last sound.
        # It trims ends.
        # If `non_sil_times_concat` logic merged things, it reduces fragmentation.
        # But the final export is a single slice from A to B. So yes, it only trims header/footer silence.
        
        # Wait, if I have: [Sound] [Silence] [Sound]
        # detect_nonsilent -> [[0, 1000], [2000, 3000]]
        # concat (gap < 200) -> no change
        # export -> sound[0 : 3000]. This INCLUDES the silence at 1000-2000.
        # So this script is "Trim Ends", not "Remove Silence".
        # I should document this or fix it if the name implies otherwise.
        # But "Transforme" implies keeping logic. I'll document it clearly.
        
        start_trim = non_sil_segments[0][0]
        end_trim = non_sil_segments[-1][-1]
        
        trimmed_sound = sound[start_trim:end_trim]
        trimmed_sound.export(path_out, format='wav')
    else:
        logger.warning(f"After filtering, no suitable audio segments found in {path_in}. Skipping.")
        return False

    return True

def process_directory(input_dir, output_dir, min_silence_len, silence_thresh_multiplier, close_gap, dry_run=False):
    makedirs(output_dir, exist_ok=True)
    files = sorted(glob(join(input_dir, "*.wav")))
    
    for input_filepath in tqdm(files, desc="Removing Silence (Trim)"):
        filename = basename(input_filepath)
        output_filepath = join(output_dir, filename)
        remove_silence(input_filepath, output_filepath, min_silence_len=min_silence_len, silence_thresh_multiplier=silence_thresh_multiplier, close_gap=close_gap, dry_run=dry_run)

def main():
    parser = argparse.ArgumentParser(description="Trim leading/trailing silence using pydub.")
    parser.add_argument('-i', '--input', default='input', help='Input folder')
    parser.add_argument('-o', '--output', default='output', help='Output folder')
    parser.add_argument('--min_silence_len', type=int, default=50, help='Minimum silence length in ms')
    parser.add_argument('--silence_thresh_multiplier', type=float, default=2.5, help='Silence threshold multiplier')
    parser.add_argument('--close_gap', type=int, default=200, help='Close gap in ms')
    parser.add_argument('--dry-run', action='store_true', help='Simulate execution')
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    
    process_directory(args.input, args.output, args.min_silence_len, args.silence_thresh_multiplier, args.close_gap, args.dry_run)

if __name__ == "__main__":
    main()
