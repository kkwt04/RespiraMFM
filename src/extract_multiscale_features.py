"""Offline, resumable multi-scale OPERA-CT feature extraction from an HDF5 split."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from RespFeatureExtractor.multiscale import MultiScaleOperaExtractor

def _text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)

def extract(input_path, output_path, checkpoint_id, windows_ms, hops_ms, device):
    with h5py.File(input_path, "r") as source:
        filenames = [_text(value) for value in source["filenames"][:]]
        labels = source["labels"][:]
        metadata = source["metadata"][:]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extractor = MultiScaleOperaExtractor(windows_ms=windows_ms, hops_ms=hops_ms)
    extractor.encoder.to(device)
    string_type = h5py.string_dtype(encoding="utf-8")
    mode = "a" if output_path.exists() else "w"
    with h5py.File(output_path, mode) as output:
        if "filenames" not in output:
            output.create_dataset("filenames", data=filenames, dtype=string_type)
            output.create_dataset("labels", data=labels)
            output.create_dataset("metadata", data=metadata, dtype=string_type)
            output.create_dataset("text_embeddings", data=np.empty((len(filenames), 0), dtype=np.float32))
            for scale in windows_ms:
                output.create_dataset(f"audio_embeddings_{scale}ms", shape=(len(filenames), 768), dtype="float32")
            output.create_dataset("completed", shape=(len(filenames),), dtype="bool", data=False)
            output.attrs["windows_ms"] = json.dumps(list(windows_ms))
            output.attrs["hops_ms"] = json.dumps(list(hops_ms))
            output.attrs["sample_rate"] = 16000
            output.attrs["encoder_checkpoint"] = checkpoint_id
        for index, filename in enumerate(filenames):
            if output["completed"][index]:
                continue
            embeddings = extractor.extract(filename, device=device)
            for scale, embedding in zip(windows_ms, embeddings):
                if embedding.shape != (768,) or not np.isfinite(embedding).all():
                    raise ValueError(f"Invalid {scale}ms feature for {filename}")
                output[f"audio_embeddings_{scale}ms"][index] = embedding
            output["completed"][index] = True
            output.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-h5", required=True)
    parser.add_argument("--output-h5", required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--windows-ms", nargs=3, type=int, default=[20, 100, 400])
    parser.add_argument("--hops-ms", nargs=3, type=int, default=[10, 50, 200])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    extract(args.input_h5, args.output_h5, args.checkpoint_id, tuple(args.windows_ms), tuple(args.hops_ms), torch.device(args.device))
