"""
Model architectures for KWS.

DS-CNN (Depthwise Separable CNN) is the default: it's the MLPerf Tiny KWS
baseline architecture and gives the best accuracy-per-KB tradeoff for
MCU-class deployment, per the build plan's tech stack rationale.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

import config


def build_ds_cnn(input_shape=None, num_classes=None) -> tf.keras.Model:
    input_shape = input_shape or (config.N_TIME_FRAMES, config.N_MFCC, 1)
    num_classes = num_classes or config.NUM_CLASSES
    filters = config.DS_CNN_FILTERS

    inputs = layers.Input(shape=input_shape, name="mfcc_input")

    # Initial standard conv
    x = layers.Conv2D(
        filters[0], config.DS_CNN_KERNEL, strides=config.DS_CNN_STRIDE,
        padding="same", use_bias=False, name="conv0",
    )(inputs)
    x = layers.BatchNormalization(name="bn0")(x)
    x = layers.ReLU(name="relu0")(x)

    # Depthwise-separable conv blocks
    for i, f in enumerate(filters[1:], start=1):
        x = layers.DepthwiseConv2D(
            (3, 3), padding="same", use_bias=False, name=f"dw{i}",
        )(x)
        x = layers.BatchNormalization(name=f"dw_bn{i}")(x)
        x = layers.ReLU(name=f"dw_relu{i}")(x)

        x = layers.Conv2D(
            f, (1, 1), padding="same", use_bias=False, name=f"pw{i}",
        )(x)
        x = layers.BatchNormalization(name=f"pw_bn{i}")(x)
        x = layers.ReLU(name=f"pw_relu{i}")(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.2, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    return models.Model(inputs, outputs, name="ds_cnn_kws")


def build_crnn(input_shape=None, num_classes=None) -> tf.keras.Model:
    """Alternative: CRNN (conv front-end + GRU) — a bit more accurate on
    longer temporal patterns, at the cost of more RAM/latency than DS-CNN.
    Useful if DS-CNN's false-reject rate isn't hitting target."""
    input_shape = input_shape or (config.N_TIME_FRAMES, config.N_MFCC, 1)
    num_classes = num_classes or config.NUM_CLASSES

    inputs = layers.Input(shape=input_shape, name="mfcc_input")

    x = layers.Conv2D(32, (5, 5), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # collapse freq axis, keep time axis for the GRU
    shape = x.shape
    x = layers.Reshape((shape[1], shape[2] * shape[3]))(x)

    x = layers.GRU(64, return_sequences=False)(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs, name="crnn_kws")


def build_model(arch: str = None) -> tf.keras.Model:
    arch = arch or config.MODEL_ARCH
    if arch == "ds_cnn":
        return build_ds_cnn()
    elif arch == "crnn":
        return build_crnn()
    else:
        raise ValueError(f"Unknown MODEL_ARCH: {arch}")


if __name__ == "__main__":
    m = build_model()
    m.summary()
    total_params = m.count_params()
    print(f"\nTotal params: {total_params:,} (~{total_params * 4 / 1024:.1f} KB as float32)")
