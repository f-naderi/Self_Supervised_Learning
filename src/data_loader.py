import os
import pickle
import numpy as np
import tensorflow as tf
import warnings


warnings.filterwarnings("ignore", category=np.exceptions.VisibleDeprecationWarning)

########################################################
# Load CIFAR-10
########################################################

def load_cifar10_python(data_path):

    train_images = []
    train_labels = []

    # Load training batches
    for i in range(1, 6):

        batch_file = os.path.join(data_path, f"data_batch_{i}")

        with open(batch_file, "rb") as f:
            batch = pickle.load(f, encoding="bytes")

        train_images.append(batch[b"data"])
        train_labels.extend(batch[b"labels"])

    x_train = np.concatenate(train_images, axis=0)
    y_train = np.array(train_labels, dtype=np.int32)

    # Load test batch
    test_file = os.path.join(data_path, "test_batch")

    with open(test_file, "rb") as f:
        test_data = pickle.load(f, encoding="bytes")

    x_test = test_data[b"data"]
    y_test = np.array(test_data[b"labels"], dtype=np.int32)

    # Reshape
    x_train = x_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    x_test = x_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    # Normalize
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    return (x_train, y_train, x_test, y_test)


########################################################
# SimCLR Dataset
########################################################

def create_simclr_dataset(images, transform, batch_size=256, shuffle=True):

    dataset = tf.data.Dataset.from_tensor_slices(images)

    # Cache raw images
    dataset = dataset.cache()

    # Shuffle every epoch
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(images), reshuffle_each_iteration=True)

    # Augmentation
    def augment(image):

        view1, view2 = transform(image)
        return view1, view2

    dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # Prefetch
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset