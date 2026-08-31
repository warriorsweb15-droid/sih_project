import os

import numpy as np
import tensorflow as tf

import config
import prepare_data

KERAS_MODEL_PATH = os.path.join(config.MODEL_OUT_DIR, "kws_model.keras")
TFLITE_OUT_PATH = os.path.join(config.MODEL_OUT_DIR, "kws_model_int8.tflite")


def make_representative_dataset(manifest, n_samples=None):
    n_samples = n_samples or config.QUANT_REP_DATASET_SIZE
    sample = manifest[:n_samples]
    ds = prepare_data.make_dataset(sample, training=False, batch_size=1)

    def rep_data_gen():
        for x_batch, _ in ds.take(n_samples):
            yield [x_batch]

    return rep_data_gen


def quantize(keras_model_path=KERAS_MODEL_PATH, out_path=TFLITE_OUT_PATH):
    print(f"Loading {keras_model_path} ...")
    model = tf.keras.models.load_model(keras_model_path)

    print("Building representative dataset for calibration...")
    manifest = prepare_data.build_manifest()
    rep_gen = make_representative_dataset(manifest)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_gen
    # Full integer quantization — required for TFLite Micro on ESP32-S3.
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    print("Converting + quantizing...")
    tflite_model = converter.convert()

    with open(out_path, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"\nSaved quantized model: {out_path}")
    print(f"Model size: {size_kb:.1f} KB")

    if size_kb > config.TARGET_MODEL_SIZE_KB:
        print(
            f"WARNING: model size ({size_kb:.1f} KB) exceeds target "
            f"({config.TARGET_MODEL_SIZE_KB} KB). Consider reducing "
            "DS_CNN_FILTERS in config.py and retraining."
        )
    else:
        print(f"Within target ({config.TARGET_MODEL_SIZE_KB} KB). ✓")

    return out_path


def sanity_check_quantized_model(tflite_path=TFLITE_OUT_PATH, manifest_sample=None):
    """Quick check: run a handful of test clips through the quantized
    interpreter and compare predicted class to expected, so you catch a
    broken quantization before flashing to hardware."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    in_scale, in_zero_point = input_details["quantization"]
    out_scale, out_zero_point = output_details["quantization"]

    if manifest_sample is None:
        manifest = prepare_data.build_manifest()
        _, _, manifest_sample = prepare_data.split_manifest(manifest)
        manifest_sample = manifest_sample[:20]

    correct = 0
    for path, label in manifest_sample:
        import features as feat_lib
        y = feat_lib.load_audio(path)
        mfcc = feat_lib.extract_features_from_waveform(y)[..., np.newaxis]
        mfcc_batch = np.expand_dims(mfcc, axis=0)

        mfcc_int8 = (mfcc_batch / in_scale + in_zero_point).astype(np.int8)
        interpreter.set_tensor(input_details["index"], mfcc_int8)
        interpreter.invoke()
        out_int8 = interpreter.get_tensor(output_details["index"])
        out_float = (out_int8.astype(np.float32) - out_zero_point) * out_scale
        pred = int(np.argmax(out_float))
        correct += int(pred == label)

    print(f"Quantized-model sanity check: {correct}/{len(manifest_sample)} correct")


if __name__ == "__main__":
    out_path = quantize()
    sanity_check_quantized_model(out_path)
