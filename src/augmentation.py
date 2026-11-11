
import tensorflow as tf
from tensorflow.keras import layers


def build_augmentation():
    """Random transformations for training images only."""

    return tf.keras.Sequential(
        [
            layers.RandomFlip(
                "horizontal",
                seed=42
            ),
            layers.RandomRotation(
                factor=0.08,
                fill_mode="reflect",
                seed=42
            ),
            layers.RandomZoom(
                height_factor=0.1,
                width_factor=0.1,
                fill_mode="reflect",
                seed=42
            ),
        ],
        name="data_augmentation"
    )
