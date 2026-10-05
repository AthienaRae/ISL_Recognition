from pathlib import Path
import csv
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "include"
    / "extracted"
    / "Greetings"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "include_greetings_metadata.csv"
)

VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv"}


def clean_class_name(folder_name: str) -> tuple[str, str]:
    """
    Example:
        '48. Hello'
        ->
        class_id='48'
        label='Hello'
    """

    match = re.match(r"^\s*(\d+)\.\s*(.+?)\s*$", folder_name)

    if match:
        return match.group(1), match.group(2)

    return "", folder_name.strip()


def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found:\n{DATASET_DIR}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = []

    class_directories = sorted(
        [p for p in DATASET_DIR.iterdir() if p.is_dir()],
        key=lambda p: p.name
    )

    print("=" * 75)
    print("INCLUDE Greetings Metadata Builder")
    print("=" * 75)

    for class_dir in class_directories:

        class_id, label = clean_class_name(class_dir.name)

        videos = sorted(
            [
                file_path
                for file_path in class_dir.rglob("*")
                if (
                    file_path.is_file()
                    and file_path.suffix.lower() in VIDEO_EXTENSIONS
                )
            ]
        )

        print(
            f"{class_id:>3} | "
            f"{label:<25} | "
            f"{len(videos):>3} videos"
        )

        for video_path in videos:

            relative_path = video_path.relative_to(PROJECT_ROOT)

            rows.append(
                {
                    "class_id": class_id,
                    "label": label,
                    "filename": video_path.name,
                    "relative_path": str(relative_path),
                    "extension": video_path.suffix.lower(),
                }
            )

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "class_id",
                "label",
                "filename",
                "relative_path",
                "extension",
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 75)
    print(f"Classes found : {len(class_directories)}")
    print(f"Videos found  : {len(rows)}")
    print(f"CSV saved to  : {OUTPUT_FILE}")
    print("=" * 75)


if __name__ == "__main__":
    main()