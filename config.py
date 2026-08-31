"""
Central config for the KWS (Keyword Spotting) training pipeline.
Edit these values for your setup — everything else reads from here.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

SPEECH_COMMANDS_DIR = os.path.join(DATA_DIR, "speech_commands_v2")   # negatives (Google Speech Commands v2)
WAKEWORD_DIR = os.path.join(DATA_DIR, "wakeword")                    # your positive samples (.wav)
NOISE_DIR = os.path.join(DATA_DIR, "background_noise")               # DEMAND / other noise clips (.wav)

FEATURES_CACHE_DIR = os.path.join(DATA_DIR, "features_cache")        # cached MFCC .npy files
MODEL_OUT_DIR = os.path.join(PROJECT_ROOT, "models")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

for d in [DATA_DIR, WAKEWORD_DIR, NOISE_DIR, FEATURES_CACHE_DIR, MODEL_OUT_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Wake word
# ---------------------------------------------------------------------------
WAKE_WORD_LABEL = "wakeword"   # folder name under WAKEWORD_DIR must match this

# ---------------------------------------------------------------------------
# Audio params
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
CLIP_DURATION_MS = 1000                       # 1s windows — standard for KWS
CLIP_LENGTH_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION_MS / 1000)

# MFCC / Mel-spectrogram feature extraction
N_MELS = 40                # mel filterbank channels (ESP-SR / TFLM Micro Speech convention)
N_MFCC = 10                 # number of MFCC coefficients to keep (drop 0th)
WINDOW_SIZE_MS = 30
WINDOW_STRIDE_MS = 20
FFT_LENGTH = 512

# Derived: number of time frames per 1s clip
N_TIME_FRAMES = int((CLIP_DURATION_MS - WINDOW_SIZE_MS) / WINDOW_STRIDE_MS) + 1  # ~49

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------
# 3-way classification is the standard KWS setup:
#   0: wake word           1: "unknown" (other speech)      2: silence/background noise
# This is deliberately NOT a big multi-word classifier — it's a lean gatekeeper.
LABELS = ["_silence_", "_unknown_", WAKE_WORD_LABEL]
NUM_CLASSES = len(LABELS)

# How many Speech Commands words to sample as "unknown" negatives per epoch worth of data
UNKNOWN_WORDS_PER_WAKEWORD_SAMPLE = 3.0   # ratio of unknown:wakeword clips (helps false-accept rate)
SILENCE_RATIO = 0.15                       # fraction of total samples that are pure silence/noise

# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
AUG_NOISE_PROB = 0.7
AUG_NOISE_SNR_DB_RANGE = (0, 15)      # lower = noisier
AUG_TIME_SHIFT_MS = 100               # random shift left/right within clip
AUG_PITCH_SHIFT_PROB = 0.3
AUG_PITCH_SHIFT_SEMITONES = (-2, 2)
AUG_TIME_STRETCH_PROB = 0.3
AUG_TIME_STRETCH_RANGE = (0.9, 1.1)
AUG_VOLUME_PROB = 0.5
AUG_VOLUME_RANGE_DB = (-6, 6)

# ---------------------------------------------------------------------------
# Model / training
# ---------------------------------------------------------------------------
MODEL_ARCH = "ds_cnn"          # "ds_cnn" or "crnn"
BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# DS-CNN width/depth — tuned for <200KB int8 target on ESP32-S3
DS_CNN_FILTERS = [64, 64, 64, 64]
DS_CNN_KERNEL = (10, 4)
DS_CNN_STRIDE = (2, 2)

# ---------------------------------------------------------------------------
# Quantization / export
# ---------------------------------------------------------------------------
QUANT_REP_DATASET_SIZE = 300    # number of samples used to calibrate int8 quantization
TARGET_MODEL_SIZE_KB = 200      # judges' target ceiling — quantize.py warns if exceeded
