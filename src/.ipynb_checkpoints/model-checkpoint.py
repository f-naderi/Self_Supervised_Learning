import tensorflow as tf


##########################################################
# Residual Block
##########################################################

@tf.keras.utils.register_keras_serializable()
class ResidualBlock(tf.keras.layers.Layer):

    def __init__(self, filters, stride=1, **kwargs):

        super().__init__(**kwargs)

        self.filters = filters
        self.stride = stride
        self.conv1 = tf.keras.layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()

        if stride != 1:

            self.shortcut = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters, 1, strides=stride, use_bias=False),
                tf.keras.layers.BatchNormalization()
            ])

        else:

            self.shortcut = tf.keras.layers.Identity()

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x, training=False):

        shortcut = self.shortcut(x, training=training)
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = x + shortcut

        return tf.nn.relu(x)


    def get_config(self):

        config = super().get_config()
        config.update({"filters": self.filters, "stride": self.stride})

        return config


##########################################################
# ResNet18
##########################################################

def make_layer(filters, blocks, stride):

    layers = []
    layers.append(ResidualBlock(filters, stride))

    for _ in range(blocks - 1):
        layers.append(ResidualBlock(filters))

    return tf.keras.Sequential(layers)


def build_encoder():

    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Conv2D(64, 3, padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = make_layer(64, 2, 1)(x)
    x = make_layer(128, 2, 2)(x)
    x = make_layer(256, 2, 2)(x)
    x = make_layer(512, 2, 2)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    model = tf.keras.Model(inputs, x, name="ResNet18")

    return model


##########################################################
# Projection Head
##########################################################

def build_projection_head(projection_dim=128):

    inputs = tf.keras.Input(shape=(512,))
    x = tf.keras.layers.Dense(512, activation="relu")(inputs)
    outputs = tf.keras.layers.Dense(projection_dim, dtype="float32")(x)

    return tf.keras.Model(inputs, outputs, name="projection_head")


##########################################################
# SimCLR
##########################################################

class SimCLR(tf.keras.Model):

    def __init__(self, projection_dim=128, **kwargs):

        super().__init__(**kwargs)
        self.projection_dim = projection_dim
        self.encoder = build_encoder()
        self.projection_head = build_projection_head(projection_dim)


    def call(self, x, training=False):

        features = self.encoder(x, training=training)
        projections = self.projection_head(features, training=training)

        return projections


    def get_config(self):

        config = super().get_config()
        config.update({"projection_dim": self.projection_dim})

        return config