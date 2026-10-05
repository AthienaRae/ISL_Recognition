from pathlib import Path
import csv
from collections import Counter, defaultdict


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLIT_FILE = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "include_greetings_splits.csv"
)

LOCAL_GREETINGS_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "include"
    / "extracted"
    / "Greetings"
)

VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv"}


def normalize_path(path_string):
    return path_string.replace("\\", "/").strip()


def main():

    print("=" * 80)
    print("INCLUDE Greetings Split Audit")
    print("=" * 80)

    # -------------------------------------------------------
    # Read Hugging Face split metadata CSV
    # -------------------------------------------------------

    with open(
        SPLIT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        rows = list(csv.DictReader(f))

    print(f"\nMetadata rows: {len(rows)}")

    # -------------------------------------------------------
    # Unique paths
    # -------------------------------------------------------

    paths = [
        normalize_path(row["video_path"])
        for row in rows
    ]

    path_counts = Counter(paths)

    unique_paths = set(paths)

    print(f"Unique metadata video paths: {len(unique_paths)}")

    duplicates = {
        path: count
        for path, count in path_counts.items()
        if count > 1
    }

    print(f"Duplicated video paths: {len(duplicates)}")

    # -------------------------------------------------------
    # Count actual local videos
    # -------------------------------------------------------

    local_files = [
        file_path
        for file_path in LOCAL_GREETINGS_DIR.rglob("*")
        if (
            file_path.is_file()
            and file_path.suffix.lower() in VIDEO_EXTENSIONS
        )
    ]

    print(f"Actual local video files: {len(local_files)}")

    # -------------------------------------------------------
    # Split membership by path
    # -------------------------------------------------------

    path_to_splits = defaultdict(set)

    for row in rows:

        video_path = normalize_path(
            row["video_path"]
        )

        path_to_splits[video_path].add(
            row["split"]
        )

    cross_split = {
        path: splits
        for path, splits in path_to_splits.items()
        if len(splits) > 1
    }

    print(
        f"Videos appearing in multiple splits: "
        f"{len(cross_split)}"
    )

    # -------------------------------------------------------
    # Unique paths per split
    # -------------------------------------------------------

    print("\nUnique videos per split:")

    for split_name in ["train", "val", "test"]:

        split_paths = {
            normalize_path(row["video_path"])
            for row in rows
            if row["split"] == split_name
        }

        print(
            f"  {split_name:<6}: "
            f"{len(split_paths)} unique videos"
        )

    # -------------------------------------------------------
    # Duplicate examples
    # -------------------------------------------------------

    if duplicates:

        print("\n" + "=" * 80)
        print("Example duplicated paths")
        print("=" * 80)

        for path, count in list(duplicates.items())[:20]:
            print(
                f"{count}x  {path}"
            )

    # -------------------------------------------------------
    # Cross-split leakage examples
    # -------------------------------------------------------

    if cross_split:

        print("\n" + "=" * 80)
        print("WARNING: Paths occurring in multiple splits")
        print("=" * 80)

        for path, splits in list(cross_split.items())[:20]:

            print(
                f"{path}"
            )

            print(
                f"    splits: "
                f"{sorted(splits)}"
            )

    # -------------------------------------------------------
    # Compare local filenames against metadata
    # -------------------------------------------------------

    local_relative_paths = set()

    for file_path in local_files:

        relative = file_path.relative_to(
            PROJECT_ROOT
            / "data"
            / "raw"
            / "include"
            / "extracted"
        )

        local_relative_paths.add(
            normalize_path(str(relative))
        )

    metadata_set = set(unique_paths)

    missing_from_metadata = (
        local_relative_paths - metadata_set
    )

    metadata_without_local = (
        metadata_set - local_relative_paths
    )

    print("\n" + "=" * 80)
    print("LOCAL ↔ METADATA COMPARISON")
    print("=" * 80)

    print(
        f"Local videos absent from metadata : "
        f"{len(missing_from_metadata)}"
    )

    print(
        f"Metadata videos absent locally    : "
        f"{len(metadata_without_local)}"
    )

    if missing_from_metadata:

        print("\nFirst local videos missing from metadata:")

        for path in sorted(
            missing_from_metadata
        )[:20]:
            print(f"  {path}")

    if metadata_without_local:

        print("\nFirst metadata paths missing locally:")

        for path in sorted(
            metadata_without_local
        )[:20]:
            print(f"  {path}")

    print("\nAudit complete.")


if __name__ == "__main__":
    main()