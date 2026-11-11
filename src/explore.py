
from pathlib import Path
from collections import Counter
from PIL import Image, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "data" / "clean" / "seg_train"
TEST_DIR = BASE_DIR / "data" / "clean" / "seg_test"

CLASSES = [
    "buildings", "forest", "glacier",
    "mountain", "sea", "street"
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def explore_dataset(directory):
    if not directory.is_dir():
        raise FileNotFoundError(f"Dataset not found: {directory}")

    print(f"\nExploring: {directory}")
    total_images = 0
    invalid_files = []
    image_sizes = Counter()
    image_modes = Counter()

    for class_name in CLASSES:
        class_dir = directory / class_name

        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class: {class_dir}")

        paths = sorted(
            path for path in class_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        print(f"{class_name}: {len(paths)} images")
        total_images += len(paths)

        for path in paths:
            try:
                # Verify file integrity.
                with Image.open(path) as image:
                    image.verify()

                # Reopen to confirm pixels can be decoded.
                with Image.open(path) as image:
                    image.load()
                    image_sizes[image.size] += 1
                    image_modes[image.mode] += 1

            except (UnidentifiedImageError, OSError, ValueError) as error:
                invalid_files.append((str(path), str(error)))

    print(f"\nTotal images: {total_images}")
    print(f"Valid images: {total_images - len(invalid_files)}")
    print(f"Invalid images: {len(invalid_files)}")
    print(f"Most common image sizes: {image_sizes.most_common(5)}")
    print(f"Image color modes: {dict(image_modes)}")

    for path, error in invalid_files[:10]:
        print(f"INVALID: {path} | {error}")

    return invalid_files


if __name__ == "__main__":
    train_invalid = explore_dataset(TRAIN_DIR)
    test_invalid = explore_dataset(TEST_DIR)

    if train_invalid or test_invalid:
        raise RuntimeError(
            "Invalid images found. Fix them before training."
        )

    print("\nDataset exploration completed successfully!")
