import tensorflow as tf


class NTXentLoss(tf.keras.losses.Loss):

    def __init__(self, temperature=0.5, name="nt_xent_loss"):

        super().__init__(name=name)
        self.temperature = temperature

    def call(self, zis, zjs):

        batch_size = tf.shape(zis)[0]

        # Normalize
        zis = tf.math.l2_normalize(zis, axis=1)
        zjs = tf.math.l2_normalize(zjs, axis=1)

        # Concatenate
        representations = tf.concat([zis, zjs], axis=0)

        # Cosine similarity
        similarity_matrix = tf.matmul(representations, representations, transpose_b=True)
        similarity_matrix = similarity_matrix / self.temperature

        # Remove self similarity
        similarity_matrix = tf.linalg.set_diag(
            similarity_matrix,
            tf.fill([2 * batch_size], tf.constant(-1e9, dtype=similarity_matrix.dtype))
        )

        # Positive labels
        positives = tf.concat([tf.range(batch_size, 2 * batch_size), tf.range(batch_size)], axis=0)

        # Cross Entropy
        loss = tf.keras.losses.sparse_categorical_crossentropy(positives, similarity_matrix, from_logits=True)

        return tf.reduce_mean(loss)