from pathlib import Path
import csv
from collections import Counter, defaultdict


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "include_greetings_splits.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "include_greetings_full_clean.csv"
)


def normalize_path(path_string: str) -> str:
    return path_string.replace("\\", "/").strip()


def main():

    print("=" * 80)
    print("Clean INCLUDE Greetings Split")
    print("Protocol: FULL INCLUDE")
    print("=" * 80)

    # ---------------------------------------------------------
    # Load original metadata
    # ---------------------------------------------------------

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        rows = list(csv.DictReader(f))

    print(f"\nOriginal metadata rows : {len(rows)}")

    # ---------------------------------------------------------
    # Keep only FULL INCLUDE
    # include_50=False
    # ---------------------------------------------------------

    full_include_rows = [
        row
        for row in rows
        if row["include_50"].strip().lower() == "false"
    ]

    print(
        f"Full INCLUDE rows      : "
        f"{len(full_include_rows)}"
    )

    # ---------------------------------------------------------
    # Deduplicate by physical video path
    # ---------------------------------------------------------

    unique_rows = {}

    for row in full_include_rows:

        video_path = normalize_path(
            row["video_path"]
        )

        if video_path in unique_rows:

            previous = unique_rows[video_path]

            # Same video must not belong to two different splits
            if previous["split"] != row["split"]:

                raise ValueError(
                    "\nCROSS-SPLIT CONFLICT FOUND:\n"
                    f"Video: {video_path}\n"
                    f"Split 1: {previous['split']}\n"
                    f"Split 2: {row['split']}"
                )

        else:

            clean_row = dict(row)

            clean_row["video_path"] = video_path

            unique_rows[video_path] = clean_row

    cleaned_rows = list(unique_rows.values())

    print(
        f"Unique physical videos : "
        f"{len(cleaned_rows)}"
    )

    # ---------------------------------------------------------
    # Verify local files
    # ---------------------------------------------------------

    missing_local = [
        row
        for row in cleaned_rows
        if row["local_exists"].strip().lower() != "true"
    ]

    print(
        f"Missing local videos   : "
        f"{len(missing_local)}"
    )

    # ---------------------------------------------------------
    # Check split leakage
    # ---------------------------------------------------------

    video_to_splits = defaultdict(set)

    for row in cleaned_rows:

        video_to_splits[
            row["video_path"]
        ].add(
            row["split"]
        )

    leakage = {
        path: splits
        for path, splits in video_to_splits.items()
        if len(splits) > 1
    }

    print(
        f"Cross-split leakage    : "
        f"{len(leakage)}"
    )

    # ---------------------------------------------------------
    # Split distribution
    # ---------------------------------------------------------

    split_counts = Counter(
        row["split"]
        for row in cleaned_rows
    )

    print("\nSplit distribution:")

    for split_name in ["train", "val", "test"]:

        print(
            f"  {split_name:<6}: "
            f"{split_counts.get(split_name, 0)} videos"
        )

    # ---------------------------------------------------------
    # Class distribution
    # ---------------------------------------------------------

    print("\nClass distribution:")

    labels = sorted(
        set(row["label"] for row in cleaned_rows)
    )

    for label in labels:

        class_rows = [
            row
            for row in cleaned_rows
            if row["label"] == label
        ]

        class_split_counts = Counter(
            row["split"]
            for row in class_rows
        )

        print(
            f"  {label:<25} "
            f"total={len(class_rows):>2} | "
            f"train={class_split_counts.get('train', 0):>2} | "
            f"val={class_split_counts.get('val', 0):>2} | "
            f"test={class_split_counts.get('test', 0):>2}"
        )

    # ---------------------------------------------------------
    # Final safety checks
    # ---------------------------------------------------------

    if missing_local:

        print("\nERROR: Some local videos are missing.")

        for row in missing_local[:10]:
            print(row["video_path"])

        return

    if leakage:

        print("\nERROR: Cross-split leakage detected.")

        for path, splits in leakage.items():
            print(path, splits)

        return

    # ---------------------------------------------------------
    # Save clean file
    # ---------------------------------------------------------

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
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            sorted(
                cleaned_rows,
                key=lambda row: (
                    row["split"],
                    row["label"],
                    row["video_path"]
                )
            )
        )

    print("\n" + "=" * 80)

    print(
        f"Clean split CSV saved to:\n"
        f"{OUTPUT_FILE}"
    )

    print("\nVALIDATION PASSED ✓")
    print("One physical video → one benchmark split")

    print("=" * 80)


if __name__ == "__main__":
    main()