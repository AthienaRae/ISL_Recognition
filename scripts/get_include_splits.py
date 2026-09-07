from pathlib import Path
import csv

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOCAL_DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "include"
    / "extracted"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "splits"
)

OUTPUT_FILE = OUTPUT_DIR / "include_greetings_splits.csv"


def normalize_path(path_string: str) -> str:
    """
    Convert paths to a consistent forward-slash format
    for matching Hugging Face metadata with local files.
    """
    return path_string.replace("\\", "/").strip()


def main():

    print("=" * 75)
    print("INCLUDE Official Split Metadata")
    print("=" * 75)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nLoading metadata from AI4Bharat / Hugging Face...")

    dataset = load_dataset(
        "ai4bharat/INCLUDE"
    )

    print("\nAvailable splits:")

    for split_name, split_data in dataset.items():
        print(
            f"  {split_name:<10}: "
            f"{len(split_data)} samples"
        )

    rows = []

    print("\nFiltering only Greetings videos...")

    for split_name, split_data in dataset.items():

        for sample in split_data:

            parent_label = str(
                sample["parent_label"]
            ).strip()

            if parent_label.lower() != "greetings":
                continue

            video_path = normalize_path(
                sample["video_path"]
            )

            local_path = (
                LOCAL_DATASET_DIR
                / Path(video_path)
            )

            rows.append(
                {
                    "split": split_name,
                    "parent_label": parent_label,
                    "label": sample["label"],
                    "video_path": video_path,
                    "include_50": sample["include_50"],
                    "local_exists": local_path.exists(),
                }
            )

    if not rows:
        print("\nERROR: No Greetings samples found.")
        return

    fieldnames = [
        "split",
        "parent_label",
        "label",
        "video_path",
        "include_50",
        "local_exists",
    ]

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 75)
    print("Greetings split summary")
    print("=" * 75)

    for split_name in dataset.keys():

        split_rows = [
            row
            for row in rows
            if row["split"] == split_name
        ]

        existing = sum(
            row["local_exists"]
            for row in split_rows
        )

        print(
            f"{split_name:<10}: "
            f"{len(split_rows):>3} metadata entries | "
            f"{existing:>3} local matches"
        )

    missing = [
        row
        for row in rows
        if not row["local_exists"]
    ]

    print()
    print(f"Total Greetings metadata rows : {len(rows)}")
    print(f"Local videos matched          : {len(rows) - len(missing)}")
    print(f"Missing local files           : {len(missing)}")

    print(f"\nCSV saved to:\n{OUTPUT_FILE}")

    if missing:

        print("\nFirst missing paths:")

        for row in missing[:10]:
            print(
                f"  {row['split']:<6} "
                f"{row['video_path']}"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()