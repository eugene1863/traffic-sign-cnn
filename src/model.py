"""CNN architecture for traffic-sign classification.

Preprocessing (rescale) and augmentation are baked into the model graph
itself (as Keras layers), not done as a separate numpy/tf.data step. Two
benefits: predict.py can hand the model raw 0-255 uint8 images with zero
extra code, and augmentation layers automatically switch themselves off
at inference time (Keras only applies them when training=True).

Note: traffic signs are direction-sensitive (e.g. "turn left" vs "turn
right", or asymmetric warning signs), so horizontal/vertical flips are
deliberately NOT used as augmentation -- flipping would silently turn a
sign into a different, wrong sign.
"""

from tensorflow import keras
from tensorflow.keras import layers

from . import config


def build_augmentation() -> keras.Sequential:
    return keras.Sequential(
        [
            layers.RandomRotation(0.06),
            layers.RandomZoom(0.1),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )


def _conv_block(x, filters: int, name: str):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.MaxPooling2D(2, name=f"{name}_pool")(x)
    x = layers.Dropout(0.25, name=f"{name}_drop")(x)
    return x


def build_model(
    num_classes: int,
    img_size: int = config.IMG_SIZE,
    channels: int = config.CHANNELS,
    learning_rate: float = config.LEARNING_RATE,
) -> keras.Model:
    inputs = keras.Input(shape=(img_size, img_size, channels), name="image")

    x = build_augmentation()(inputs)
    x = layers.Rescaling(1.0 / 255)(x)

    x = _conv_block(x, 32, "block1")
    x = _conv_block(x, 64, "block2")
    x = _conv_block(x, 128, "block3")

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="traffic_sign_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    build_model(num_classes=43).summary()
