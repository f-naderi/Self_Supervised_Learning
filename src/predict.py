import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
from src.model import ResidualBlock


# Load trained linear probe
model = tf.keras.models.load_model("linear_probe/linear_probe.keras", custom_objects={"ResidualBlock": ResidualBlock}, compile=False)

#######################################################
# Load CIFAR10 class names
#######################################################

with open("data/cifar-10-python/batches.meta", "rb") as f:
    meta = pickle.load(f, encoding="bytes")

label_names = [

    name.decode("utf-8")

    for name in meta[b"label_names"]

]

#######################################################
# Prediction
#######################################################

def predict_image(image, label):

    image = tf.convert_to_tensor(image, dtype=tf.float32)

    if image.shape != (32, 32, 3):
        image = tf.image.resize(image, (32, 32))

    image = tf.expand_dims(image, axis=0)
    pred = model.predict(image, verbose=0)[0]
    top5 = np.argsort(pred)[::-1][:5]

    plt.imshow(tf.squeeze(image))
    plt.axis("off")
    plt.show()

    print("Top-5 Predictions\n")

    for idx in top5:
        print(f"{label_names[idx]:15s} : {pred[idx]:.4f}")

    print("True label :", label_names[int(label)])