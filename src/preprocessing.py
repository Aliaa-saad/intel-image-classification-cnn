
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers

from src.augmentation import build_augmentation


BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "data" / "clean" / "seg_train"
TEST_DIR = BASE_DIR / "data" / "clean" / "seg_test"

CLASS_NAMES = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street",
]

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.20
SEED = 42

AUTOTUNE = tf.data.AUTOTUNE


def normalize_images(images, labels):
    """Convert RGB pixel values from 0-255 to 0-1."""
    images = tf.cast(images, tf.float32) / 255.0
    return images, labels


def load_datasets(
    batch_size=BATCH_SIZE,
    apply_augmentation=True
):
    """
    Returns:
        train_ds, val_ds, test_ds, class_names

    Images: float32 tensors in [0, 1]
    Labels: integer class IDs (0 to 5)
    """

    if not TRAIN_DIR.is_dir():
        raise FileNotFoundError(f"Missing: {TRAIN_DIR}")

    if not TEST_DIR.is_dir():
        raise FileNotFoundError(f"Missing: {TEST_DIR}")

    # Split ONLY the original training dataset.
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        color_mode="rgb",
        shuffle=True,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        color_mode="rgb",
        shuffle=True,
    )

    # The test dataset stays completely separate.
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        color_mode="rgb",
        shuffle=False,
    )

    # Apply normalization to all three datasets.
    train_ds = train_ds.map(
        normalize_images,
        num_parallel_calls=AUTOTUNE
    )

    val_ds = val_ds.map(
        normalize_images,
        num_parallel_calls=AUTOTUNE
    )

    test_ds = test_ds.map(
        normalize_images,
        num_parallel_calls=AUTOTUNE
    )

    # Random augmentation is applied ONLY to training images.
    if apply_augmentation:
        augmentation = build_augmentation()

        def augment_images(images, labels):
            images = augmentation(images, training=True)
            return images, labels

        train_ds = train_ds.map(
            augment_images,
            num_parallel_calls=AUTOTUNE
        )

    # Prefetch improves input pipeline performance.
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, CLASS_NAMES
