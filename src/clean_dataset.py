
from pathlib import Path
from collections import defaultdict
from hashlib import sha256
import csv
import shutil

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_TRAIN = BASE_DIR / "data" / "seg_train" / "seg_train"
RAW_TEST = BASE_DIR / "data" / "seg_test" / "seg_test"

CLEAN_DIR = BASE_DIR / "data" / "clean"
CLEAN_TRAIN = CLEAN_DIR / "seg_train"
CLEAN_TEST = CLEAN_DIR / "seg_test"

REPORT_DIR = BASE_DIR / "results"
REPORT_DIR.mkdir(exist_ok=True)

CLASSES = [
    "buildings", "forest", "glacier",
    "mountain", "sea", "street"
]

EXTENSIONS = {".jpg", ".jpeg", ".png"}


def calculate_hash(path):
    digest = sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def collect_images():
    groups = defaultdict(list)

    for split, root in [
        ("train", RAW_TRAIN),
        ("test", RAW_TEST),
    ]:
        if not root.is_dir():
            raise FileNotFoundError(root)

        for class_name in CLASSES:
            folder = root / class_name

            if not folder.is_dir():
                raise FileNotFoundError(folder)

            for path in sorted(folder.iterdir()):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in EXTENSIONS:
                    continue

                digest = calculate_hash(path)
                groups[digest].append({
                    "path": path,
                    "split": split,
                    "class": class_name,
                })

    return groups


def clean_dataset():
    print("Scanning original dataset...")

    groups = collect_images()

    # Rebuild only our generated clean directory.
    # The original seg_train and seg_test remain untouched.
    if CLEAN_DIR.exists():
        shutil.rmtree(CLEAN_DIR)

    for root in [CLEAN_TRAIN, CLEAN_TEST]:
        for class_name in CLASSES:
            (root / class_name).mkdir(parents=True, exist_ok=True)

    kept = {"train": 0, "test": 0}
    removed = []
    conflict_groups = 0

    for digest, records in sorted(groups.items()):
        labels = {record["class"] for record in records}

        # Never guess the correct class when labels conflict.
        if len(labels) > 1:
            conflict_groups += 1

            for record in records:
                removed.append({
                    "path": str(record["path"]),
                    "reason": "conflicting_labels",
                    "hash": digest,
                })

            continue


        # Preserve test images when the same image exists
        # in both training and testing sets.
        # Keep the selection deterministic.
        records = sorted(
            records,
            key=lambda item: (
                0 if item["split"] == "test" else 1,
                str(item["path"]),
            ),
        )

        selected = records[0]

        destination_root = (
            CLEAN_TRAIN
            if selected["split"] == "train"
            else CLEAN_TEST
        )


        destination = (
            destination_root
            / selected["class"]
            / selected["path"].name
        )

        shutil.copy2(selected["path"], destination)
        kept[selected["split"]] += 1

        for duplicate in records[1:]:
            removed.append({
                "path": str(duplicate["path"]),
                "reason": "exact_duplicate",
                "hash": digest,
            })

    report_path = REPORT_DIR / "cleaning_report.csv"

    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["path", "reason", "hash"]
        )
        writer.writeheader()
        writer.writerows(removed)

    print("\n========== CLEANING RESULTS ==========")
    print("Conflicting-label groups:", conflict_groups)
    print("Removed files:", len(removed))
    print("Clean training images:", kept["train"])
    print("Clean testing images:", kept["test"])
    print("Report:", report_path)
    print("\nDataset cleaning completed!")


if __name__ == "__main__":
    clean_dataset()
