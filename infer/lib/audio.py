import platform
import os
import subprocess
import numpy as np
import av
import random
from io import BytesIO
import traceback
import logging
import librosa
import string

def wav2(i, o, format):
    inp = av.open(i, "rb")
    if format == "m4a":
        format = "mp4"
    out = av.open(o, "wb", format=format)
    if format == "ogg":
        format = "libvorbis"
    if format == "mp4":
        format = "aac"

    ostream = out.add_stream(format)

    for frame in inp.decode(audio=0):
        for p in ostream.encode(frame):
            out.mux(p)

    for p in ostream.encode(None):
        out.mux(p)

    out.close()
    inp.close()


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def load_audio(file, sr):
    """Loads audio using librosa, converting with sox if necessary, with improved error handling."""

    file = clean_path(file)
    if not os.path.exists(file):
        raise FileNotFoundError(f"Audio file not found: {file}")

    try:
        logging.debug(f"Attempting to load audio file: {file}")

        audio, sr_loaded = librosa.load(file, sr=sr)

        if sr is not None and sr != sr_loaded:
            logging.warning(f"Resampling audio from {sr_loaded} Hz to {sr} Hz.")
            audio = librosa.resample(audio, sr_loaded, sr)

        return audio.astype(np.float32)

    except Exception as e:  # Catch any other exceptions
        catch(e)

def clean_path(path_str):
    if platform.system() == "Windows":
        path_str = path_str.replace("/", "\\")
    return path_str.strip(" ").strip('"').strip("\n").strip('"').strip(" ")





