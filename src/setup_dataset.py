
from pathlib import Path
import shutil
import subprocess
import sys

import kagglehub


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

DATASET_HANDLE = "puneet6060/intel-image-classification"

CLASSES = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street",
]

TRAIN_DIR = DATA_DIR / "seg_train" / "seg_train"
TEST_DIR = DATA_DIR / "seg_test" / "seg_test"

CLEAN_TRAIN = DATA_DIR / "clean" / "seg_train"
CLEAN_TEST = DATA_DIR / "clean" / "seg_test"

EXTENSIONS = {".jpg", ".jpeg", ".png"}


def valid_split(folder):
    """Check that all six classes contain image files."""
    return all(
        (folder / name).is_dir()
        and any(
            p.is_file() and p.suffix.lower() in EXTENSIONS
            for p in (folder / name).iterdir()
        )
        for name in CLASSES
    )


def find_split(download_path, split_name):
    """Find a Kaggle split even when folders are nested."""
    candidates = [
        download_path,
        *download_path.rglob(split_name),
    ]

    for candidate in candidates:
        if candidate.is_dir() and valid_split(candidate):
            return candidate

        nested = candidate / split_name

        if nested.is_dir() and valid_split(nested):
            return nested

    raise FileNotFoundError(
        f"Cannot locate {split_name} in {download_path}"
    )


def copy_split(source, destination):
    """Copy images into the project's expected structure."""
    if valid_split(destination):
        print(f"Already available: {destination}")
        return

    # Do not overwrite partially existing folders.
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(
            f"Destination is not empty but incomplete: {destination}"
        )

    print(f"Copying {source} -> {destination}")
    shutil.copytree(source, destination, dirs_exist_ok=True)


def run_cleaning():
    """Execute the existing dataset cleaning script."""
    print("\nCleaning and deduplicating dataset...")

    subprocess.run(
        [sys.executable, "-m", "src.clean_dataset"],
        cwd=BASE_DIR,
        check=True,
    )


def main():
    print("========== INTEL DATASET SETUP ==========")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Reuse existing original data when already downloaded.
    if valid_split(TRAIN_DIR) and valid_split(TEST_DIR):
        print("\nOriginal dataset already exists.")
        print("Skipping download.")

    else:
        print("\nDownloading Intel Dataset from Kaggle...")

        downloaded_path = Path(
            kagglehub.dataset_download(DATASET_HANDLE)
        )

        print(f"Downloaded dataset location: {downloaded_path}")

        source_train = find_split(
            downloaded_path, "seg_train"
        )

        source_test = find_split(
            downloaded_path, "seg_test"
        )

        copy_split(source_train, TRAIN_DIR)
        copy_split(source_test, TEST_DIR)

    # Check cleaned data before running cleaning again.
    if valid_split(CLEAN_TRAIN) and valid_split(CLEAN_TEST):
        print("\nClean dataset already exists.")
        print("Skipping cleaning.")

    else:
        run_cleaning()

    if not (
        valid_split(CLEAN_TRAIN)
        and valid_split(CLEAN_TEST)
    ):
        raise RuntimeError(
            "Dataset setup failed: clean data was not created."
        )

    print("\n========== SETUP COMPLETE ==========")
    print("Original training:", TRAIN_DIR)
    print("Original testing:", TEST_DIR)
    print("Clean training:", CLEAN_TRAIN)
    print("Clean testing:", CLEAN_TEST)
    print("\nDataset is ready for CNN preprocessing!")


if __name__ == "__main__":
    main()
