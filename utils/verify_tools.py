import os
import shutil
import logging
from audio_sak.normalization import normalize_audios_by_dbfs
from audio_sak.segmentation import segment_audio
# from audio_sak.transcription import transcribe_with_whisper # might need heavy deps
# from audio_sak.embeddings import extract_wav2vec_embeddings 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
OUTPUT_DIR = os.path.join(BASE_DIR, "test_output")

def setup_dirs():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    
def test_normalization():
    logger.info("Testing Normalization...")
    norm_out = os.path.join(OUTPUT_DIR, "normalization")
    os.makedirs(norm_out, exist_ok=True)
    
    # We need to call the function logic. The module usually has a function or main.
    # Checking imports, likely: normalize_audios_by_dbfs.normalize_audios(input_dir, output_dir)
    # But I need to check the exact signature.
    # Assuming standard CLI structure, I might need to run it via subprocess or import main logic.
    # For now, let's try running as subprocess to simulate CLI usage which is safer for "black box" testing.
    
    cmd = f"python -m audio_sak.normalization.normalize_audios_by_dbfs -i {AUDIO_DIR} -o {norm_out}"
    ret = os.system(cmd)
    if ret == 0:
        logger.info("Normalization Test Passed")
    else:
        logger.error("Normalization Test Failed")

def test_segmentation():
    logger.info("Testing Segmentation...")
    seg_out = os.path.join(OUTPUT_DIR, "segmentation")
    os.makedirs(seg_out, exist_ok=True)
    
    cmd = f"python -m audio_sak.segmentation.segment_audio --input {AUDIO_DIR} --output {seg_out} --min_duration 1.0 --max_duration 5.0 --threshold_db 40"
    ret = os.system(cmd)
    if ret == 0:
        logger.info("Segmentation Test Passed")
    else:
        logger.error("Segmentation Test Failed")

def test_conversion():
    logger.info("Testing Conversion (WAV to FLAC)...")
    # There isn't a direct wav to flac in the README examples, but let's check file listing previously:
    # useful_audio/conversion/convert_wav_to_flac.py exists.
    conv_out = os.path.join(OUTPUT_DIR, "conversion")
    os.makedirs(conv_out, exist_ok=True)
    
    cmd = f"python -m audio_sak.conversion.convert_wav_to_flac -i {AUDIO_DIR} -o {conv_out}"
    ret = os.system(cmd)
    if ret == 0:
        logger.info("Conversion Test Passed")
    else:
        logger.error("Conversion Test Failed")

if __name__ == "__main__":
    setup_dirs()
    test_normalization()
    test_segmentation()
    test_conversion()
