
from pathlib import Path
from collections import defaultdict
from hashlib import sha256

import tensorflow as tf
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.preprocessing import (
    TRAIN_DIR,
    TEST_DIR,
    CLASS_NAMES,
    IMAGE_SIZE,
    VALIDATION_SPLIT,
    BATCH_SIZE,
    SEED,
)
from src.augmentation import build_augmentation


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

EXTENSIONS = {".jpg", ".jpeg", ".png"}


def image_hash(path):
    """Calculate SHA256 hash of an image file."""
    digest = sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def get_split_paths():
    """Recreate the same train/validation split."""

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        class_names=CLASS_NAMES,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        class_names=CLASS_NAMES,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    train_paths = {Path(p).resolve() for p in train_ds.file_paths}
    val_paths = {Path(p).resolve() for p in val_ds.file_paths}

    test_paths = {
        p.resolve()
        for class_name in CLASS_NAMES
        for p in (TEST_DIR / class_name).iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    }

    return train_paths, val_paths, test_paths


def audit_splits():
    print("\n========== DATA LEAKAGE CHECK ==========")

    train_paths, val_paths, test_paths = get_split_paths()

    print("Training:", len(train_paths))
    print("Validation:", len(val_paths))
    print("Testing:", len(test_paths))

    # Check that no file path belongs to multiple splits.
    assert not train_paths & val_paths, "Train/Val path overlap!"
    assert not train_paths & test_paths, "Train/Test path overlap!"
    assert not val_paths & test_paths, "Val/Test path overlap!"

    all_paths = train_paths | val_paths | test_paths

    hashes = {}
    for path in all_paths:
        hashes[path] = image_hash(path)

    train_hashes = {hashes[p] for p in train_paths}
    val_hashes = {hashes[p] for p in val_paths}
    test_hashes = {hashes[p] for p in test_paths}

    overlaps = {
        "Train/Validation": train_hashes & val_hashes,
        "Train/Test": train_hashes & test_hashes,
        "Validation/Test": val_hashes & test_hashes,
    }

    for name, duplicates in overlaps.items():
        print(f"{name} duplicate hashes: {len(duplicates)}")

    if any(overlaps.values()):
        hash_to_paths = defaultdict(list)

        for path, digest in hashes.items():
            hash_to_paths[digest].append(str(path))

        for name, duplicates in overlaps.items():
            for digest in list(duplicates)[:3]:
                print(f"\nDuplicate in {name}:")
                for path in hash_to_paths[digest]:
                    print(" ", path)

        raise AssertionError(
            "Duplicate image content found across dataset splits."
        )

    print("\nPASS: No exact duplicate files across splits!")
    return train_paths


def test_augmentation(train_paths):
    print("\n========== AUGMENTATION CHECK ==========")

    sample_path = sorted(train_paths)[0]
    print("Sample:", sample_path)

    with Image.open(sample_path) as img:
        img = img.convert("RGB")
        img = img.resize(IMAGE_SIZE)
        original = tf.convert_to_tensor(img)

    original = tf.cast(original, tf.float32) / 255.0
    batch = tf.expand_dims(original, axis=0)

    augmentation = build_augmentation()

    augmented_images = []

    for _ in range(4):
        augmented = augmentation(batch, training=True)
        augmented_images.append(augmented[0])

    differences = [
        float(tf.reduce_mean(tf.abs(original - image)))
        for image in augmented_images
    ]

    print("Mean differences:", differences)

    assert all(
        image.shape == original.shape
        for image in augmented_images
    ), "Augmentation changed the image shape!"

    assert all(
        float(tf.reduce_min(image)) >= 0
        and float(tf.reduce_max(image)) <= 1
        for image in augmented_images
    ), "Augmentation produced invalid pixel values!"

    assert any(
        difference > 0.00001
        for difference in differences
    ), "Augmentation did not visibly change pixel values!"

    fig, axes = plt.subplots(1, 5, figsize=(15, 4))

    axes[0].imshow(original.numpy())
    axes[0].set_title("Original")
    axes[0].axis("off")

    for index, image in enumerate(augmented_images):
        axes[index + 1].imshow(
            tf.clip_by_value(image, 0, 1).numpy()
        )
        axes[index + 1].set_title(f"Augmented {index + 1}")
        axes[index + 1].axis("off")

    plt.tight_layout()

    output_path = RESULTS_DIR / "augmentation_comparison.png"
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    print("Saved:", output_path)
    print("PASS: Augmentation changes image pixels!")


def main():
    train_paths = audit_splits()
    test_augmentation(train_paths)

    print("\nALL QUALITY CHECKS PASSED!")


if __name__ == "__main__":
    main()
