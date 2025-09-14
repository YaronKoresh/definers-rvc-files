import os
import sys
import traceback
import argparse
import logging
import time
import fairseq
import torch
import numpy as np
import soundfile as sf

import torch
from fairseq.data.dictionary import Dictionary
torch.serialization.add_safe_globals([Dictionary])

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

def readwave(wav_path, normalize=False, mono=True):
    try:
        wav, sr = sf.read(wav_path)
        if sr != 16000:
            raise ValueError(f"Sample rate is not 16kHz: {sr}")
        feats = torch.from_numpy(wav).float()
        if feats.dim() == 2:  # double channels
            if mono:
                feats = feats.mean(-1)
                logging.info(f"Averaged stereo channels for {wav_path}")
            else:
                logging.warning(f"Stereo audio, but mono=False. Processing only the first channel for {wav_path}")
                feats = feats[:, 0]
        if feats.dim() != 1:
            raise ValueError(f"Unexpected number of channels: {feats.dim()}")

        if normalize:
            feats = F.layer_norm(feats, feats.shape)
        feats = feats.view(1, -1)
        return feats
    except Exception as e:
        logging.exception(f"Error reading wave file {wav_path}:")
        raise  # Re-raise the exception after logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract HuBERT features.")
    parser.add_argument("device", help="Device to use (cpu, cuda, mps, directml)")
    parser.add_argument("n_part", type=int, help="Total number of parts")
    parser.add_argument("i_part", type=int, help="Part number")
    parser.add_argument("exp_dir", help="Experiment directory")
    parser.add_argument("version", choices=["v1", "v2"], help="HuBERT version")
    parser.add_argument("--half", action="store_true", help="Use half-precision")
    parser.add_argument("--model_path", default="assets/hubert/hubert_base.pt", help="Path to HuBERT model")
    parser.add_argument("--mono", action="store_true", help="Convert stereo to mono")
    args = parser.parse_args()

    if "privateuseone" not in args.device:
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
    else:
        import torch_directml
        device = torch_directml.device(torch_directml.default_device())

        def forward_dml(ctx, x, scale):
            ctx.scale = scale
            res = x.clone().detach()
            return res

        fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml

    logging.basicConfig(filename=os.path.join(args.exp_dir, f"extract_f0_feature_{args.i_part}.log"), level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info(" ".join(sys.argv))  # Log command-line arguments

    logging.info(f"exp_dir: {args.exp_dir}")
    wavPath = os.path.join(args.exp_dir, "1_16k_wavs")
    outPath = os.path.join(args.exp_dir, "3_feature256" if args.version == "v1" else "3_feature768")
    os.makedirs(outPath, exist_ok=True)

    logging.info(f"load model(s) from {args.model_path}")
    if not os.path.exists(args.model_path):
        logging.error(f"Extracting is shut down because {args.model_path} does not exist.")
        sys.exit(1)  # Exit with error code

    models, saved_cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [args.model_path],
        suffix="",
    )
    model = models[0]
    model = model.to(device)
    logging.info(f"move model to {device}")
    if args.half and device not in ["mps", "cpu"]:
        model = model.half()
    model.eval()

    todo = sorted(list(os.listdir(wavPath)))[args.i_part::args.n_part]
    n = max(1, len(todo) // 10)  # 最多打印十条
    if len(todo) == 0:
        logging.info("no-feature-todo")
    else:
        logging.info(f"all-feature-{len(todo)}")
        start_time = time.time()
        for idx, file in enumerate(todo):
            try:
                if file.endswith(".wav"):
                    wav_path = os.path.join(wavPath, file)
                    out_path = os.path.join(outPath, file.replace("wav", "npy"))

                    if os.path.exists(out_path):
                        continue

                    feats = readwave(wav_path, normalize=saved_cfg.task.normalize, mono=args.mono)
                    padding_mask = torch.BoolTensor(feats.shape).fill_(False)
                    inputs = {
                        "source": (
                            feats.half().to(device)
                            if args.half and device not in ["mps", "cpu"]
                            else feats.to(device)
                        ),
                        "padding_mask": padding_mask.to(device),
                        "output_layer": 9 if args.version == "v1" else 12,  # layer 9
                    }
                    with torch.no_grad():
                        logits = model.extract_features(**inputs)
                        feats = (
                            model.final_proj(logits[0]) if args.version == "v1" else logits[0]
                        )

                    feats = feats.squeeze(0).float().cpu().numpy()
                    if np.isnan(feats).sum() == 0:
                        np.save(out_path, feats, allow_pickle=False)
                    else:
                        logging.warning(f"{file}-contains nan")
                    if idx % n == 0:
                        logging.info(f"now-{len(todo)},all-{idx},{file},{feats.shape}")
            except Exception as e:
                logging.exception(f"Error processing file {file}:")

        end_time = time.time()
        logging.info(f"Finished processing {len(todo)} files in {end_time - start_time:.2f} seconds.")

    logging.info("all-feature-done")
    sys.exit(0)  # Indicate success




