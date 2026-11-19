"""Unexecuted synthetic contracts for the optional multi-scale pathway."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import h5py

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from RespFeatureExtractor.multiscale import (
    mel_spectrogram_from_waveform,
    milliseconds_to_samples,
    multiscale_mel_spectrograms,
)
from multiscale import MultiScaleFusion, MultiScaleProjectionModel, load_multiscale_checkpoint
from model_utils import AudioTextDataset

def test_milliseconds_to_samples_uses_actual_sample_rate():
    assert milliseconds_to_samples(20, 16000) == 320
    assert milliseconds_to_samples(100, 16000) == 1600
    assert milliseconds_to_samples(400, 16000) == 6400

def test_three_scales_preserve_64_mel_bins_and_distinct_time_resolution():
    waveform = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)
    scales = multiscale_mel_spectrograms(waveform, 16000)
    assert len(scales) == 3
    assert all(scale.shape[1] == 64 for scale in scales)
    assert scales[0].shape[0] > scales[1].shape[0] > scales[2].shape[0]
    baseline = mel_spectrogram_from_waveform(waveform, 16000, 64, 32)
    assert baseline.shape[1] == 64

def test_fusion_contract_and_gradients():
    fusion = MultiScaleFusion()
    inputs = [torch.randn(2, 768, requires_grad=True) for _ in range(3)]
    output = fusion(*inputs)
    assert output.shape == (2, 768)
    output.sum().backward()
    assert all(parameter.grad is not None for parameter in fusion.parameters())

def test_projection_model_uses_one_aligner_after_fusion():
    aligner = torch.nn.Linear(768, 2560)
    model = MultiScaleProjectionModel(MultiScaleFusion(), aligner)
    output = model(*(torch.randn(3, 768) for _ in range(3)))
    assert output.shape == (3, 2560)
    assert model.aligner is aligner

def test_frozen_encoder_has_no_trainable_parameters():
    encoder = torch.nn.Linear(64, 768)
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    assert not any(parameter.requires_grad for parameter in encoder.parameters())

def test_multiscale_checkpoint_contains_loadable_fusion_and_aligner(tmp_path):
    fusion = MultiScaleFusion()
    aligner = torch.nn.Linear(768, 2560)
    checkpoint_path = tmp_path / "phase1_multiscale.pth"
    torch.save({"fusion_state_dict": fusion.state_dict(), "aligner_state_dict": aligner.state_dict()}, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    restored_fusion = MultiScaleFusion()
    restored_aligner = torch.nn.Linear(768, 2560)
    restored_fusion.load_state_dict(checkpoint["fusion_state_dict"])
    restored_aligner.load_state_dict(checkpoint["aligner_state_dict"])
    for module in (restored_fusion, restored_aligner):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad = False
    assert not any(parameter.requires_grad for parameter in restored_fusion.parameters())
    assert not any(parameter.requires_grad for parameter in restored_aligner.parameters())


def test_phase1_optimizer_contains_fusion_and_aligner_parameters():
    model = MultiScaleProjectionModel(MultiScaleFusion(), torch.nn.Linear(768, 2560))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert all(id(parameter) in optimized for parameter in model.fusion.parameters())
    assert all(id(parameter) in optimized for parameter in model.aligner.parameters())


def test_phase2_checkpoint_loader_freezes_both_modules(tmp_path):
    source_fusion = MultiScaleFusion()
    source_aligner = torch.nn.Linear(768, 2560)
    checkpoint_path = tmp_path / "phase1.pth"
    torch.save({"fusion_state_dict": source_fusion.state_dict(), "aligner_state_dict": source_aligner.state_dict()}, checkpoint_path)
    fusion = MultiScaleFusion()
    aligner = torch.nn.Linear(768, 2560)
    load_multiscale_checkpoint(fusion, aligner, checkpoint_path, freeze=True)
    assert not fusion.training and not aligner.training
    assert not any(parameter.requires_grad for parameter in fusion.parameters())
    assert not any(parameter.requires_grad for parameter in aligner.parameters())


def test_feature_cache_scale_rows_remain_aligned(tmp_path):
    cache_path = tmp_path / "multiscale.h5"
    with h5py.File(cache_path, "w") as cache:
        cache.create_dataset("filenames", data=[b"a.wav", b"b.wav"])
        for scale, values in ((20, [1, 2]), (100, [3, 4]), (400, [5, 6])):
            cache.create_dataset(f"audio_embeddings_{scale}ms", data=np.array(values, dtype=np.float32)[:, None].repeat(768, axis=1))
    with h5py.File(cache_path, "r") as cache:
        assert cache["filenames"][1] == b"b.wav"
        assert cache["audio_embeddings_20ms"][1, 0] == 2
        assert cache["audio_embeddings_100ms"][1, 0] == 4
        assert cache["audio_embeddings_400ms"][1, 0] == 6


def test_baseline_mode_bypasses_multiscale_cache_fields(tmp_path):
    cache_path = tmp_path / "baseline.h5"
    with h5py.File(cache_path, "w") as cache:
        cache.create_dataset("filenames", data=[b"a.wav"])
        cache.create_dataset("labels", data=[0])
        cache.create_dataset("metadata", data=[b"metadata"])
        cache.create_dataset("text_embeddings", data=np.zeros((1, 1), dtype=np.float32))
        cache.create_dataset("audio_embeddings", data=np.zeros((1, 768), dtype=np.float32))
    dataset = AudioTextDataset(SimpleNamespace(audio_mode="baseline", multimodal_config=1), str(cache_path), "ukcovid19")
    _, embedding, _, _, _ = dataset[0]
    assert isinstance(embedding, torch.Tensor)
    assert embedding.shape == (768,)
