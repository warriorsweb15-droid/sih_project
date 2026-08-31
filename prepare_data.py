import argparse
import glob
import os
import random
import tarfile
import urllib.request

import numpy as np
import tensorflow as tf

import augment
import config
import features

SPEECH_COMMANDS_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/data/"
    "speech_commands_v0.02.tar.gz"
)


def download_speech_commands():
    """One-time download + extraction of Google Speech Commands v2."""
    os.makedirs(config.SPEECH_COMMANDS_DIR, exist_ok=True)
    archive_path = os.path.join(config.DATA_DIR, "speech_commands_v0.02.tar.gz")

    if not os.path.exists(archive_path):
        print(f"Downloading Speech Commands v2 to {archive_path} ...")
        urllib.request.urlretrieve(SPEECH_COMMANDS_URL, archive_path)
    else:
        print("Archive already downloaded, skipping fetch.")

    print("Extracting...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=config.SPEECH_COMMANDS_DIR)
    print(f"Done. Extracted to {config.SPEECH_COMMANDS_DIR}")


def _list_wavs(directory, pattern="*.wav"):
    return glob.glob(os.path.join(directory, "**", pattern), recursive=True)


def build_manifest():
    """
    Returns list of (filepath, label_index) tuples covering:
      - wake word positives          -> label 2
      - "unknown" negatives from Speech Commands words -> label 1
      - silence/background clips     -> label 0
    """
    manifest = []

    # --- positives ---
    wake_wavs = _list_wavs(os.path.join(config.WAKEWORD_DIR, config.WAKE_WORD_LABEL))
    if len(wake_wavs) < 50:
        print(
            f"WARNING: only found {len(wake_wavs)} wake-word samples in "
            f"{config.WAKEWORD_DIR}/{config.WAKE_WORD_LABEL}/. "
            "The build plan targets 300-500+. Accuracy will suffer with too few."
        )
    for p in wake_wavs:
        manifest.append((p, config.LABELS.index(config.WAKE_WORD_LABEL)))

    n_wake = max(len(wake_wavs), 1)

    # --- "unknown" negatives: sample from all Speech Commands word folders ---
    sc_root = config.SPEECH_COMMANDS_DIR
    word_dirs = [
        d for d in glob.glob(os.path.join(sc_root, "*"))
        if os.path.isdir(d) and not os.path.basename(d).startswith("_")
    ]
    all_unknown_wavs = []
    for d in word_dirs:
        all_unknown_wavs.extend(_list_wavs(d))

    n_unknown_target = int(n_wake * config.UNKNOWN_WORDS_PER_WAKEWORD_SAMPLE)
    random.seed(config.RANDOM_SEED)
    random.shuffle(all_unknown_wavs)
    unknown_sample = all_unknown_wavs[:n_unknown_target] if all_unknown_wavs else []

    if not unknown_sample:
        print(
            "WARNING: no Speech Commands data found under "
            f"{sc_root}. Run with --download_speech_commands first."
        )
    for p in unknown_sample:
        manifest.append((p, config.LABELS.index("_unknown_")))

    # --- silence/background ---
    bg_dirs = [
        os.path.join(sc_root, "_background_noise_"),
        config.NOISE_DIR,
    ]
    bg_wavs = []
    for d in bg_dirs:
        bg_wavs.extend(_list_wavs(d))

    n_silence_target = int(len(manifest) * config.SILENCE_RATIO / (1 - config.SILENCE_RATIO))
    if bg_wavs:
        for i in range(n_silence_target):
            p = bg_wavs[i % len(bg_wavs)]
            manifest.append((p, config.LABELS.index("_silence_")))
    else:
        print("WARNING: no background noise clips found — silence class will be empty.")

    random.shuffle(manifest)
    print(
        f"Manifest built: {len(manifest)} total clips "
        f"(wakeword={len(wake_wavs)}, unknown={len(unknown_sample)}, "
        f"silence≈{n_silence_target})"
    )
    return manifest


def split_manifest(manifest):
    random.seed(config.RANDOM_SEED)
    random.shuffle(manifest)
    n = len(manifest)
    n_test = int(n * config.TEST_SPLIT)
    n_val = int(n * config.VAL_SPLIT)
    test = manifest[:n_test]
    val = manifest[n_test:n_test + n_val]
    train = manifest[n_test + n_val:]
    return train, val, test


def _load_and_featurize(path: str, label: int, is_silence: bool, training: bool):
    if is_silence and random.random() < 0.5:
        # true silence: zeros, occasionally with a touch of low-level noise
        y = np.zeros(config.CLIP_LENGTH_SAMPLES, dtype=np.float32)
        if random.random() < 0.5:
            y = y + np.random.normal(0, 0.005, size=y.shape).astype(np.float32)
    else:
        y = features.load_audio(path)

    if training:
        y = augment.augment_waveform(y)

    mfcc = features.extract_features_from_waveform(y)
    return mfcc, label


def make_dataset(manifest, training: bool, batch_size: int = None):
    """Build a tf.data.Dataset that yields (mfcc, one_hot_label) batches."""
    batch_size = batch_size or config.BATCH_SIZE

    def gen():
        items = list(manifest)
        if training:
            random.shuffle(items)
        for path, label in items:
            is_silence = (label == config.LABELS.index("_silence_"))
            mfcc, lbl = _load_and_featurize(path, label, is_silence, training)
            one_hot = np.zeros(config.NUM_CLASSES, dtype=np.float32)
            one_hot[lbl] = 1.0
            yield mfcc[..., np.newaxis], one_hot  # add channel dim -> (time, mfcc, 1)

    output_signature = (
        tf.TensorSpec(shape=(config.N_TIME_FRAMES, config.N_MFCC, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(config.NUM_CLASSES,), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    if training:
        ds = ds.shuffle(buffer_size=min(2000, len(manifest) + 1))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_speech_commands", action="store_true")
    args = parser.parse_args()

    if args.download_speech_commands:
        download_speech_commands()
    else:
        m = build_manifest()
        tr, va, te = split_manifest(m)
        print(f"train={len(tr)} val={len(va)} test={len(te)}")
