"""
Train the KWS model.

Usage:
    python prepare_data.py --download_speech_commands   # once
    # record your wake-word samples into data/wakeword/wakeword/*.wav
    python train.py

Outputs:
    models/kws_model.keras       <- best checkpoint (float32)
    logs/                        <- TensorBoard logs
    Prints FAR/FRR on the held-out test set at the end.
"""

import os

import numpy as np
import tensorflow as tf

import config
import model as model_lib
import prepare_data

WAKE_LABEL_IDX = config.LABELS.index(config.WAKE_WORD_LABEL)


def compute_class_weights(manifest):
    """Inverse-frequency class weights — keeps the (usually rarer) wake-word
    class from being drowned out by unknown/silence during training."""
    counts = np.zeros(config.NUM_CLASSES)
    for _, label in manifest:
        counts[label] += 1
    counts = np.maximum(counts, 1)
    total = counts.sum()
    weights = total / (config.NUM_CLASSES * counts)
    return {i: float(w) for i, w in enumerate(weights)}


def evaluate_far_frr(model: tf.keras.Model, test_manifest, threshold: float = 0.5):
    """
    False Accept Rate: non-wakeword clips incorrectly classified as wakeword.
    False Reject Rate: wakeword clips NOT classified as wakeword.
    Matches the metrics table in the build plan (Section 6).
    """
    test_ds = prepare_data.make_dataset(test_manifest, training=False, batch_size=64)

    y_true, y_pred_probs = [], []
    for x_batch, y_batch in test_ds:
        probs = model.predict(x_batch, verbose=0)
        y_true.append(np.argmax(y_batch.numpy(), axis=1))
        y_pred_probs.append(probs)

    y_true = np.concatenate(y_true)
    y_pred_probs = np.concatenate(y_pred_probs)
    wake_probs = y_pred_probs[:, WAKE_LABEL_IDX]
    predicted_wake = wake_probs >= threshold

    is_wakeword = (y_true == WAKE_LABEL_IDX)
    is_non_wakeword = ~is_wakeword

    n_wake = max(is_wakeword.sum(), 1)
    n_non_wake = max(is_non_wakeword.sum(), 1)

    false_rejects = np.sum(is_wakeword & ~predicted_wake)
    false_accepts = np.sum(is_non_wakeword & predicted_wake)

    frr = false_rejects / n_wake
    far = false_accepts / n_non_wake  # per-clip FAR; see note below for per-hour estimate

    print("\n--- Test set evaluation ---")
    print(f"Wake-word clips:      {n_wake}")
    print(f"Non-wake-word clips:  {n_non_wake}")
    print(f"False Reject Rate:    {frr*100:.2f}%  (target: <5%)")
    print(f"False Accept Rate:    {far*100:.2f}% of non-wakeword clips")
    print(
        "  Note: to get the judges' target metric (<1 false accept/hour), run "
        "the exported .tflite model continuously against several hours of real "
        "background/idle-room audio with eval_far_realtime.py (see README)."
    )
    return far, frr


def main():
    print("Building data manifest...")
    manifest = prepare_data.build_manifest()
    train_m, val_m, test_m = prepare_data.split_manifest(manifest)
    print(f"train={len(train_m)}  val={len(val_m)}  test={len(test_m)}")

    class_weights = compute_class_weights(train_m)
    print(f"Class weights: {class_weights}")

    train_ds = prepare_data.make_dataset(train_m, training=True)
    val_ds = prepare_data.make_dataset(val_m, training=False)

    model = model_lib.build_model()
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    ckpt_path = os.path.join(config.MODEL_OUT_DIR, "kws_model.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8, restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=config.LOGS_DIR),
    ]

    # NOTE: class_weight is applied per-sample-scalar by Keras only for
    # integer-label losses. With one-hot categorical_crossentropy we instead
    # fold class weight into sample_weight via a small wrapper dataset.
    def add_sample_weight(x, y):
        label_idx = tf.argmax(y, axis=1)
        weights_tensor = tf.constant(
            [class_weights[i] for i in range(config.NUM_CLASSES)], dtype=tf.float32
        )
        sw = tf.gather(weights_tensor, label_idx)
        return x, y, sw

    train_ds_weighted = train_ds.map(add_sample_weight)

    model.fit(
        train_ds_weighted,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=callbacks,
    )

    print(f"\nBest model saved to {ckpt_path}")
    evaluate_far_frr(model, test_m)

    print(
        "\nNext step: run `python quantize.py` to produce the Int8 .tflite "
        "model for deployment, then `python export_esp32.py` to generate the "
        "C header for the ESP32-S3 firmware."
    )


if __name__ == "__main__":
    main()