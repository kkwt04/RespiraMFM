"""Optional multi-scale Mel preprocessing and shared OPERA-CT extraction."""

import math

import librosa
import numpy as np
import torch

from RespFeatureExtractor.feature_extractor import OperaExtractor
from RespFeatureExtractor.util import get_entire_signal_librosa


DEFAULT_SCALE_WINDOWS_MS = (20, 100, 400)
DEFAULT_SCALE_HOPS_MS = (10, 50, 200)


def milliseconds_to_samples(milliseconds, sample_rate):
    """Convert a positive millisecond duration using the actual sample rate."""
    if milliseconds <= 0 or sample_rate <= 0:
        raise ValueError("milliseconds and sample_rate must be positive")
    return max(1, int(round(milliseconds * sample_rate / 1000.0)))


def mel_spectrogram_from_waveform(waveform, sample_rate, window_ms, hop_ms,
                                  n_mels=64, f_min=50, f_max=2000):
    """Return OPERA-compatible time-major, per-sample normalized Mel features."""
    n_fft = milliseconds_to_samples(window_ms, sample_rate)
    hop_length = milliseconds_to_samples(hop_ms, sample_rate)
    mel = librosa.feature.melspectrogram(
        y=np.asarray(waveform), sr=sample_rate, n_mels=n_mels, fmin=f_min,
        fmax=f_max, n_fft=n_fft, win_length=n_fft, hop_length=hop_length,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.max() != mel_db.min():
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
    return mel_db.T.astype(np.float32)


def multiscale_mel_spectrograms(waveform, sample_rate, windows_ms=DEFAULT_SCALE_WINDOWS_MS,
                                hops_ms=DEFAULT_SCALE_HOPS_MS, n_mels=64,
                                f_min=50, f_max=2000):
    if len(windows_ms) != 3 or len(hops_ms) != 3:
        raise ValueError("multi-scale mode requires exactly three windows and hops")
    return tuple(
        mel_spectrogram_from_waveform(waveform, sample_rate, window, hop, n_mels, f_min, f_max)
        for window, hop in zip(windows_ms, hops_ms)
    )


class MultiScaleOperaExtractor:
    """One frozen OPERA-CT instance shared across all temporal scales."""

    def __init__(self, windows_ms=DEFAULT_SCALE_WINDOWS_MS, hops_ms=DEFAULT_SCALE_HOPS_MS,
                 sample_rate=16000, n_mels=64, f_min=50, f_max=2000):
        self.windows_ms = tuple(windows_ms)
        self.hops_ms = tuple(hops_ms)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.base_extractor = OperaExtractor(pretrain="operaCT", from_spec=True)
        self.encoder = self.base_extractor.model
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def spectrograms(self, audio_path, input_sec=8):
        waveform = get_entire_signal_librosa(
            "", audio_path, input_sec=input_sec, sample_rate=self.sample_rate, pad=True,
        )
        return multiscale_mel_spectrograms(
            waveform, self.sample_rate, self.windows_ms, self.hops_ms,
            self.n_mels, self.f_min, self.f_max,
        )

    def extract(self, audio_path, device=None, input_sec=8):
        features = []
        self.encoder.eval()
        with torch.no_grad():
            for spectrogram in self.spectrograms(audio_path, input_sec):
                tensor = torch.from_numpy(spectrogram).unsqueeze(0)
                if device is not None:
                    tensor = tensor.to(device)
                embedding = self.encoder.extract_feature(tensor, dim=768)
                if embedding.shape[-1] != 768 or not torch.isfinite(embedding).all():
                    raise ValueError("OPERA-CT must produce finite 768-dimensional embeddings")
                features.append(embedding.squeeze(0).cpu().numpy().astype(np.float32))
        return tuple(features)
