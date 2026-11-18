"""Unexecuted synthetic contracts for the optional multi-scale pathway."""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from RespFeatureExtractor.multiscale import (
    mel_spectrogram_from_waveform,
    milliseconds_to_samples,
    multiscale_mel_spectrograms,
)
from multiscale import MultiScaleFusion, MultiScaleProjectionModel

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
