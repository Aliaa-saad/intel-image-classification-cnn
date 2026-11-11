
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import tensorflow as tf

from src.preprocessing import (
    load_datasets,
    CLASS_NAMES,
    IMAGE_SIZE,
    BATCH_SIZE,
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def inspect_batch(dataset, name):
    images, labels = next(iter(dataset))

    print(f"\n{name}")
    print("Image batch shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("Data type:", images.dtype)
    print("Minimum pixel value:", float(tf.reduce_min(images)))
    print("Maximum pixel value:", float(tf.reduce_max(images)))

    assert images.shape[1:] == (*IMAGE_SIZE, 3)
    assert images.dtype == tf.float32
    assert int(tf.reduce_min(images)) >= 0
    assert float(tf.reduce_max(images)) <= 1.0
    assert bool(tf.reduce_all(labels >= 0))
    assert bool(tf.reduce_all(labels < len(CLASS_NAMES)))

    return images, labels


def save_sample_images(images, labels, class_names):
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))

    for index, ax in enumerate(axes.flat):
        ax.imshow(images[index].numpy())
        ax.set_title(class_names[int(labels[index])])
        ax.axis("off")

    plt.tight_layout()
    output_path = RESULTS_DIR / "sample_images.png"
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"\nSample images saved: {output_path}")


def main():
    print("Loading datasets...")

    train_ds, val_ds, test_ds, class_names = load_datasets()

    print("\nClasses:", class_names)

    assert class_names == CLASS_NAMES

    train_images, train_labels = inspect_batch(
        train_ds, "TRAINING"
    )

    inspect_batch(val_ds, "VALIDATION")
    inspect_batch(test_ds, "TESTING")

    # Batch 0 may contain 32 images; the last batch may be smaller.
    assert train_images.shape[0] <= BATCH_SIZE

    save_sample_images(
        train_images, train_labels, class_names
    )

    print("\nALL PIPELINE CHECKS PASSED!")


if __name__ == "__main__":
    main()
