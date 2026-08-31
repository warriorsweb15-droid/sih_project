"""
Estimate real-world False Accept Rate (per hour) by sliding the quantized
model over long background/idle-room recordings — this is what the judges
actually mean by "<1 false accept/hour" (per-clip FAR on a balanced test set
is a different, easier number; don't present it as if it were per-hour).

Usage:
    python eval_far_realtime.py --audio path/to/long_idle_recording.wav
    (record 1-3 hours of your actual demo room: fans, chatter, typing, etc.)
"""

import argparse

import librosa
import numpy as np
import tensorflow as tf

import config
import features

WAKE_LABEL_IDX = config.LABELS.index(config.WAKE_WORD_LABEL)

# How often to run inference — mirrors the firmware's inference cadence.
STRIDE_MS = 200


def run(audio_path: str, tflite_path: str, threshold: float = 0.5):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    in_scale, in_zero_point = input_details["quantization"]
    out_scale, out_zero_point = output_details["quantization"]

    print(f"Loading {audio_path} ...")
    y, _ = librosa.load(audio_path, sr=config.SAMPLE_RATE, mono=True)
    total_hours = len(y) / config.SAMPLE_RATE / 3600
    print(f"Duration: {total_hours*3600:.0f}s ({total_hours:.2f} hours)")

    stride_samples = int(config.SAMPLE_RATE * STRIDE_MS / 1000)
    clip_len = config.CLIP_LENGTH_SAMPLES

    false_accepts = 0
    n_windows = 0

    for start in range(0, len(y) - clip_len, stride_samples):
        window = y[start:start + clip_len]
        mfcc = features.extract_features_from_waveform(window)[..., np.newaxis]
        mfcc_batch = np.expand_dims(mfcc, axis=0)
        mfcc_int8 = (mfcc_batch / in_scale + in_zero_point).astype(np.int8)

        interpreter.set_tensor(input_details["index"], mfcc_int8)
        interpreter.invoke()
        out_int8 = interpreter.get_tensor(output_details["index"])
        out_float = (out_int8.astype(np.float32) - out_zero_point) * out_scale

        wake_prob = out_float[0, WAKE_LABEL_IDX]
        n_windows += 1
        if wake_prob >= threshold:
            false_accepts += 1

    far_per_hour = false_accepts / max(total_hours, 1e-6)
    print(f"\nWindows evaluated: {n_windows}")
    print(f"False accepts:     {false_accepts}")
    print(f"Estimated FAR:     {far_per_hour:.2f} per hour  (target: <1/hour)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Long background/idle recording (.wav)")
    parser.add_argument(
        "--model",
        default=f"{config.MODEL_OUT_DIR}/kws_model_int8.tflite",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    run(args.audio, args.model, args.threshold)
