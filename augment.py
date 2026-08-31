import glob
import os
import random

import librosa
import numpy as np

import config

_noise_clips_cache = None


def _load_noise_clips():
    global _noise_clips_cache
    if _noise_clips_cache is not None:
        return _noise_clips_cache
    paths = glob.glob(os.path.join(config.NOISE_DIR, "**", "*.wav"), recursive=True)
    clips = []
    for p in paths:
        try:
            y, _ = librosa.load(p, sr=config.SAMPLE_RATE, mono=True)
            if len(y) >= config.CLIP_LENGTH_SAMPLES:
                clips.append(y)
        except Exception:
            continue
    _noise_clips_cache = clips
    return clips


def mix_noise(y: np.ndarray, snr_db_range=None) -> np.ndarray:
    """Mix in a random background noise clip at a random SNR."""
    clips = _load_noise_clips()
    if not clips:
        return y  # no noise clips available — no-op rather than crash

    snr_db_range = snr_db_range or config.AUG_NOISE_SNR_DB_RANGE
    noise = random.choice(clips)

    if len(noise) > len(y):
        start = random.randint(0, len(noise) - len(y))
        noise = noise[start:start + len(y)]
    else:
        reps = int(np.ceil(len(y) / len(noise)))
        noise = np.tile(noise, reps)[:len(y)]

    signal_power = np.mean(y ** 2) + 1e-10
    noise_power = np.mean(noise ** 2) + 1e-10
    snr_db = random.uniform(*snr_db_range)
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(target_noise_power / noise_power)

    mixed = y + noise
    max_val = np.max(np.abs(mixed))
    if max_val > 1.0:
        mixed = mixed / max_val
    return mixed.astype(np.float32)


def time_shift(y: np.ndarray, max_shift_ms=None) -> np.ndarray:
    max_shift_ms = max_shift_ms or config.AUG_TIME_SHIFT_MS
    max_shift = int(config.SAMPLE_RATE * max_shift_ms / 1000)
    shift = random.randint(-max_shift, max_shift)
    return np.roll(y, shift).astype(np.float32)


def pitch_shift(y: np.ndarray, semitone_range=None) -> np.ndarray:
    semitone_range = semitone_range or config.AUG_PITCH_SHIFT_SEMITONES
    n_steps = random.uniform(*semitone_range)
    return librosa.effects.pitch_shift(y, sr=config.SAMPLE_RATE, n_steps=n_steps).astype(np.float32)


def time_stretch(y: np.ndarray, rate_range=None) -> np.ndarray:
    rate_range = rate_range or config.AUG_TIME_STRETCH_RANGE
    rate = random.uniform(*rate_range)
    stretched = librosa.effects.time_stretch(y, rate=rate)
    # Re-pad/truncate back to fixed clip length since time_stretch changes duration
    target_len = config.CLIP_LENGTH_SAMPLES
    if len(stretched) < target_len:
        pad = target_len - len(stretched)
        stretched = np.pad(stretched, (pad // 2, pad - pad // 2), mode="constant")
    else:
        stretched = stretched[:target_len]
    return stretched.astype(np.float32)


def volume_jitter(y: np.ndarray, db_range=None) -> np.ndarray:
    db_range = db_range or config.AUG_VOLUME_RANGE_DB
    gain_db = random.uniform(*db_range)
    gain = 10 ** (gain_db / 20)
    out = y * gain
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def augment_waveform(y: np.ndarray) -> np.ndarray:
    """Apply the full randomized augmentation chain used during training."""
    y = y.copy()

    if random.random() < config.AUG_PITCH_SHIFT_PROB:
        y = pitch_shift(y)
    if random.random() < config.AUG_TIME_STRETCH_PROB:
        y = time_stretch(y)

    y = time_shift(y)

    if random.random() < config.AUG_VOLUME_PROB:
        y = volume_jitter(y)
    if random.random() < config.AUG_NOISE_PROB:
        y = mix_noise(y)

    return y
