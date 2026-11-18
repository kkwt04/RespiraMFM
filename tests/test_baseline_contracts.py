"""Synthetic contracts for the untouched RespiraMFM baseline.

These tests require the venv dependencies but do not require real audio,
downloaded weights, or a GPU.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from RespFeatureExtractor.util import pre_process_audio_mel_t
from evaluate import evaluate
from model_utils import AudioTextDataset


def test_mel_preprocessing_returns_time_major_64_bin_spectrogram():
    audio = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)
    mel = pre_process_audio_mel_t(audio)
    assert mel.ndim == 2
    assert mel.shape[1] == 64


def test_hdf5_dataset_contract_and_embedding_dimension(tmp_path):
    feature_path = tmp_path / "features.h5"
    string_type = h5py.string_dtype(encoding="utf-8")
    with h5py.File(feature_path, "w") as store:
        store.create_dataset("filenames", data=["sample.wav"], dtype=string_type)
        store.create_dataset("labels", data=np.array([1], dtype=np.int64))
        store.create_dataset("audio_embeddings", data=np.zeros((1, 768), dtype=np.float32))
        store.create_dataset("text_embeddings", data=np.zeros((1, 1), dtype=np.float32))
        store.create_dataset("metadata", data=["age: 40"], dtype=string_type)
    dataset = AudioTextDataset(SimpleNamespace(multimodal_config=1), str(feature_path), "ukcovid19")
    _, embedding, _, _, label = dataset[0]
    assert embedding.shape == (768,)
    assert embedding.dtype == torch.float32
    assert label.item() == 1


def test_evaluation_uses_continuous_positive_class_scores():
    class FakeModel(torch.nn.Module):
        def forward(self, filenames, embeddings, prompts, contexts):
            return SimpleNamespace(logits=torch.tensor([[3.0, 1.0], [0.0, 4.0]]))

    batch = (["negative.wav", "positive.wav"], torch.zeros((2, 768)), ["p", "p"], ["c", "c"], torch.tensor([0, 1]))
    metrics, labels, predictions, probabilities = evaluate(FakeModel(), [batch], "ukcovid19", torch.device("cpu"))
    assert labels == [0, 1]
    assert predictions == [0, 1]
    assert probabilities[0] < probabilities[1]
    assert metrics["auroc"] == 1.0


def test_checkpoint_serialization_contract(tmp_path):
    checkpoint = tmp_path / "checkpoint.pth"
    state = {"linear.weight": torch.ones((2, 2)), "linear.bias": torch.zeros(2)}
    torch.save(state, checkpoint)
    loaded = torch.load(checkpoint, map_location="cpu")
    assert set(loaded) == set(state)
    assert loaded["linear.weight"].shape == (2, 2)
