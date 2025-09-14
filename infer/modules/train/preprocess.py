
import multiprocessing
import os
import sys
import traceback
import argparse
import logging
import time
import random

import librosa
import numpy as np
from scipy import signal
from scipy.io import wavfile
import soundfile as sf

from ...lib.audio import load_audio
from ...lib.slicer2 import Slicer

parser = argparse.ArgumentParser(description="Preprocess audio files.")

parser.add_argument("inp_root", help="Input directory")
parser.add_argument("sr", type=int, help="Sample rate")
parser.add_argument("n_p", type=int, help="Number of processes")
parser.add_argument("exp_dir", help="Experiment directory")
parser.add_argument("--per", type=float, default=3.7, help="Duration per chunk (seconds)") # Configurable
parser.add_argument("--target_amplitude", type=float, default=0.9, help="Target amplitude") # Configurable

args = parser.parse_args()

def norm_write(tmp_audio, idx0, idx1, sr, target_sr=16000, target_amplitude=0.9):
    tmp_max = np.abs(tmp_audio).max()
    if tmp_max > 1.0:  # Check for clipping (adjust threshold as needed)
        logging.warning(f"{idx0}-{idx1}-clipped (max: {tmp_max})")
        tmp_audio = tmp_audio / tmp_max * target_amplitude # Scale to target amplitude
    else:
        tmp_audio = tmp_audio / tmp_max * target_amplitude

    wavfile.write(
        os.path.join(args.exp_dir, "0_gt_wavs", f"{idx0}_{idx1}.wav"),
        sr,
        tmp_audio.astype(np.float32),
    )
    if sr != target_sr:
        tmp_audio = librosa.resample(tmp_audio, orig_sr=sr, target_sr=target_sr)
    wavfile.write(
        os.path.join(args.exp_dir, "1_16k_wavs", f"{idx0}_{idx1}.wav"),
        target_sr,
        tmp_audio.astype(np.float32),
    )

def pipeline(path, idx0, sr, slicer, bh, ah, per, target_amplitude):
    try:
        audio = load_audio(path, sr)
        audio = signal.lfilter(bh, ah, audio)  # Filter audio

        idx1 = 0
        for chunk in slicer.slice(audio):  # Simplified slicing logic
            for i in range(int(len(chunk) / (sr * per)) +1): # simplified chunking logic
                start = int(sr*per*i)
                tmp_audio = chunk[start:min(start+int(sr*per), len(chunk))]
                norm_write(tmp_audio, idx0, idx1, sr, target_sr=16000, target_amplitude=target_amplitude)
                idx1 += 1

        logging.info(f"{path}\t-> Success")
    except Exception as e:
        logging.exception(f"{path}\t-> Error:")

def pipeline_mp(infos, sr, slicer, bh, ah, per, target_amplitude):
    for path, idx0 in infos:
        pipeline(path, idx0, sr, slicer, bh, ah, per, target_amplitude)

def preprocess_trainset(inp_root, sr, n_p, exp_dir, per, target_amplitude):
    slicer = Slicer(sr=sr, threshold=-42, min_length=1500, min_interval=400, hop_size=15, max_sil_kept=500)
    bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=sr)

    logging.info("start preprocess")

    try:
        infos = [(os.path.join(inp_root, name), idx) for idx, name in enumerate(sorted(os.listdir(inp_root)))]

        if n_p <= 1: # Run sequentially
            pipeline_mp(infos, sr, slicer, bh, ah, per, target_amplitude)
        else:
            with multiprocessing.Pool(processes=n_p) as pool: # Use a context manager
                pool.starmap(pipeline_mp, [(infos[i::n_p], sr, slicer, bh, ah, per, target_amplitude) for i in range(n_p)])

    except Exception as e:
        logging.exception("Fail:")

    logging.info("end preprocess")

logging.basicConfig(filename=os.path.join(args.exp_dir, "preprocess.log"), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
preprocess_trainset(args.inp_root, args.sr, args.n_p, args.exp_dir, args.per, args.target_amplitude)





