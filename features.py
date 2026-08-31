"""
MFCC feature extraction.

Deliberately mirrors the framing (window size / stride / mel bins) used by
TensorFlow Lite Micro's "Micro Speech" reference pipeline, so features
computed here line up with what your ESP32-S3 firmware will compute on-device
at inference time. If you change these params, change them in the firmware
too (see esp32_firmware/ in the next step).
"""

import numpy as np
import librosa

import config


def load_audio(path: str, target_len: int = config.CLIP_LENGTH_SAMPLES) -> np.ndarray:
    """Load a wav file, resample to SAMPLE_RATE, pad/truncate to target_len samples."""
    y, _ = librosa.load(path, sr=config.SAMPLE_RATE, mono=True)
    if len(y) < target_len:
        pad = target_len - len(y)
        y = np.pad(y, (pad // 2, pad - pad // 2), mode="constant")
    else:
        y = y[:target_len]
    return y.astype(np.float32)


def compute_mfcc(y: np.ndarray) -> np.ndarray:
    """
    y: 1D float32 waveform, length == CLIP_LENGTH_SAMPLES, values in [-1, 1]
    returns: (N_TIME_FRAMES, N_MFCC) float32 array
    """
    win_length = int(config.SAMPLE_RATE * config.WINDOW_SIZE_MS / 1000)
    hop_length = int(config.SAMPLE_RATE * config.WINDOW_STRIDE_MS / 1000)

    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=config.SAMPLE_RATE,
        n_fft=config.FFT_LENGTH,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=config.N_MELS,
        power=2.0,
        fmin=20,
        fmax=config.SAMPLE_RATE // 2,
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    mfcc = librosa.feature.mfcc(
        S=log_mel,
        n_mfcc=config.N_MFCC + 1,   # +1 because we drop the 0th coefficient (energy)
    )[1:, :]

    mfcc = mfcc.T  # -> (time, n_mfcc)

    # Pad/truncate time axis to the expected fixed frame count
    n_frames = config.N_TIME_FRAMES
    if mfcc.shape[0] < n_frames:
        pad = n_frames - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad), (0, 0)), mode="constant")
    else:
        mfcc = mfcc[:n_frames, :]

    # Per-clip normalization (zero mean, unit variance) — makes int8
    # quantization ranges much more stable across clips.
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)
    return mfcc.astype(np.float32)


def extract_features(path: str) -> np.ndarray:
    """One-shot: load a wav file and return its (time, n_mfcc) MFCC feature map."""
    y = load_audio(path)
    return compute_mfcc(y)


def extract_features_from_waveform(y: np.ndarray) -> np.ndarray:
    """Same as extract_features but takes an already-loaded/augmented waveform."""
    target_len = config.CLIP_LENGTH_SAMPLES
    if len(y) != target_len:
        if len(y) < target_len:
            pad = target_len - len(y)
            y = np.pad(y, (pad // 2, pad - pad // 2), mode="constant")
        else:
            y = y[:target_len]
    return compute_mfcc(y.astype(np.float32))
