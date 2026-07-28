import tensorflow as tf
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report

from tensorflow.keras import mixed_precision

from src.data_loader import load_cifar10_python
from src.model import ResidualBlock


# Mixed Precision
mixed_precision.set_global_policy("mixed_float16")

########################################################
# Configuration
########################################################

DATA_PATH = "data/cifar-10-python"
BATCH_SIZE = 512
EPOCHS = 100
NUM_CLASSES = 10

# BASE_LR = 0.3
# LEARNING_RATE = BASE_LR * (BATCH_SIZE / 256)
LEARNING_RATE = 0.1

########################################################
# Output folders
########################################################

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

LINEAR_DIR = Path("linear_probe")
LINEAR_DIR.mkdir(exist_ok=True)

########################################################
# Linear Evaluation
########################################################

def linear_evaluate():

    ####################################################
    # Dataset
    ####################################################

    x_train, y_train, x_test, y_test = load_cifar10_python(DATA_PATH)

    train_dataset = (
        tf.data.Dataset
        .from_tensor_slices((x_train, y_train))
        .shuffle(len(x_train), reshuffle_each_iteration=True)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_dataset = (
        tf.data.Dataset
        .from_tensor_slices((x_test, y_test))
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    ####################################################
    # Load Encoder
    ####################################################

    encoder = tf.keras.models.load_model("saved_models/encoder.keras", custom_objects={"ResidualBlock": ResidualBlock}, compile=False)

    encoder.trainable = False

    for layer in encoder.layers:
        layer.trainable = False

    ####################################################
    # Linear Probe
    ####################################################

    inputs = tf.keras.Input(shape=(32, 32, 3))
    features = encoder(inputs, training=False)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", kernel_initializer="random_normal", bias_initializer="zeros", dtype="float32")(features)
    linear_model = tf.keras.Model(inputs, outputs, name="linear_probe")

    ####################################################
    # Compile
    ####################################################

    optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.9, nesterov=False)
    linear_model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])


    # Callbacks
    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            filepath=LINEAR_DIR / "best.keras",
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            mode="max",
            verbose=1
        )

    ]


    # Train
    history = linear_model.fit(train_dataset, validation_data=test_dataset, epochs=EPOCHS, callbacks=callbacks, verbose=2)


    ####################################################
    # Save History
    ####################################################

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(RESULTS_DIR / "history.csv", index=False)

    ####################################################
    # Plot Curves
    ####################################################

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()



    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Validation")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_curve.png", dpi=300)
    plt.close()

    ####################################################
    # Final Evaluation
    ####################################################

    loss, acc = linear_model.evaluate(test_dataset, verbose=0)

    print()
    print("=" * 50)
    print(f"Final Test Loss     : {loss:.6f}")
    print(f"Final Test Accuracy : {acc:.6f}")
    print("=" * 50)

    ####################################################
    # Prediction
    ####################################################

    y_pred = linear_model.predict(x_test, batch_size=BATCH_SIZE, verbose=0)
    y_pred = np.argmax(y_pred, axis=1)

    ####################################################
    # Classification Report
    ####################################################

    report = classification_report(y_test, y_pred, digits=4)

    print(report)

    with open(RESULTS_DIR / "classification_report.txt", "w") as f:
        f.write(report)

    ####################################################
    # Save Results
    ####################################################

    with open(RESULTS_DIR / "linear_eval_results.txt", "w") as f:

        f.write("=" * 50 + "\n")
        f.write("SimCLR Linear Evaluation\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test Loss     : {loss:.6f}\n")
        f.write(f"Test Accuracy : {acc:.6f}\n")
        f.write(f"Epochs        : {EPOCHS}\n")
        f.write(f"Batch Size    : {BATCH_SIZE}\n")
        f.write(f"Learning Rate : {LEARNING_RATE:.6f}\n")
        f.write(f"Optimizer     : SGD(momentum=0.9)\n")


    # Save Probe
    linear_model.save(LINEAR_DIR / "linear_probe.keras")

    return linear_model, history


if __name__ == "__main__":

    linear_evaluate()