import argparse
import hashlib
import sys
from pathlib import Path

import requests


ZENODO_RECORD_ID = "4010759"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_DIR = PROJECT_ROOT / "data" / "raw" / "include"

CHUNK_SIZE = 1024 * 1024  # 1 MB


def human_size(num_bytes):
    size = float(num_bytes)

    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024

    return f"{size:.2f} PB"


def calculate_md5(file_path):
    md5 = hashlib.md5()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            md5.update(chunk)

    return md5.hexdigest()


def get_zenodo_files():
    print("Fetching INCLUDE metadata from Zenodo...")

    try:
        response = requests.get(ZENODO_API_URL, timeout=60)
        response.raise_for_status()

    except requests.RequestException as exc:
        print(f"\nFailed to contact Zenodo:\n{exc}")
        sys.exit(1)

    record = response.json()

    files = record.get("files", [])

    if not files:
        print("No files found in Zenodo record.")
        sys.exit(1)

    return files


def select_files(files, category=None, download_all=False):

    # Ignore README and original shell script
    zip_files = [
        file_info
        for file_info in files
        if file_info["key"].lower().endswith(".zip")
    ]

    if download_all:
        return zip_files

    if category:
        category_lower = category.lower()

        selected = [
            file_info
            for file_info in zip_files
            if file_info["key"].lower().startswith(category_lower)
        ]

        if not selected:
            print(f'\nNo ZIP files found for category "{category}".')
            sys.exit(1)

        return selected

    print("\nERROR: Specify either:")
    print("  --category CATEGORY")
    print("or")
    print("  --all")

    sys.exit(1)


def verify_existing_file(path, expected_size, expected_md5):

    if not path.exists():
        return False

    if path.stat().st_size != expected_size:
        return False

    print(f"\nChecking existing file: {path.name}")

    actual_md5 = calculate_md5(path)

    if actual_md5 == expected_md5:
        print("Already downloaded and checksum verified.")
        return True

    print("Existing file has incorrect checksum.")
    print("It will be downloaded again.")

    path.unlink()

    return False


def download_file(file_info):

    filename = file_info["key"]
    expected_size = file_info["size"]

    checksum_value = file_info.get("checksum", "")

    if checksum_value.startswith("md5:"):
        expected_md5 = checksum_value.split(":", 1)[1]
    else:
        expected_md5 = checksum_value

    links = file_info.get("links", {})

    download_url = links.get("content") or links.get("self")

    destination = DOWNLOAD_DIR / filename

    print("\n" + "=" * 80)
    print(f"File : {filename}")
    print(f"Size : {human_size(expected_size)}")
    print("=" * 80)

    # -------------------------------------------------
    # Already completed?
    # -------------------------------------------------

    if verify_existing_file(
        destination,
        expected_size,
        expected_md5
    ):
        return

    # -------------------------------------------------
    # Resume partially downloaded file
    # -------------------------------------------------

    existing_size = 0

    if destination.exists():
        existing_size = destination.stat().st_size

    headers = {}

    if 0 < existing_size < expected_size:

        headers["Range"] = f"bytes={existing_size}-"

        print(
            f"Partial download detected: "
            f"{human_size(existing_size)}"
        )

        print("Attempting to resume...")

    try:

        response = requests.get(
            download_url,
            headers=headers,
            stream=True,
            timeout=(30, 300)
        )

        response.raise_for_status()

    except requests.RequestException as exc:

        print(f"\nDownload failed:\n{exc}")
        return

    # -------------------------------------------------
    # Check whether server accepted Range request
    # -------------------------------------------------

    if existing_size > 0 and response.status_code == 206:

        file_mode = "ab"
        downloaded = existing_size

        print("Server accepted resume request.")

    else:

        if existing_size > 0:
            print(
                "Server did not support resume for this request."
            )
            print("Restarting this file from beginning.")

        file_mode = "wb"
        downloaded = 0

    # -------------------------------------------------
    # Download
    # -------------------------------------------------

    try:

        with open(destination, file_mode) as f:

            for chunk in response.iter_content(
                chunk_size=CHUNK_SIZE
            ):

                if not chunk:
                    continue

                f.write(chunk)

                downloaded += len(chunk)

                percent = (
                    downloaded / expected_size * 100
                    if expected_size
                    else 0
                )

                print(
                    f"\r"
                    f"{human_size(downloaded)} / "
                    f"{human_size(expected_size)} "
                    f"({percent:6.2f}%)",
                    end="",
                    flush=True
                )

    except KeyboardInterrupt:

        print("\n\nDownload interrupted.")
        print("Run the same command again to resume.")

        sys.exit(130)

    except OSError as exc:

        print(f"\nFile writing error: {exc}")
        return

    print("\nDownload complete.")

    # -------------------------------------------------
    # Size verification
    # -------------------------------------------------

    actual_size = destination.stat().st_size

    if actual_size != expected_size:

        print(
            f"WARNING: File size mismatch.\n"
            f"Expected : {human_size(expected_size)}\n"
            f"Actual   : {human_size(actual_size)}"
        )

        return

    # -------------------------------------------------
    # MD5 verification
    # -------------------------------------------------

    print("Verifying MD5 checksum...")

    actual_md5 = calculate_md5(destination)

    if actual_md5 == expected_md5:

        print("MD5 VERIFIED ✓")

    else:

        print("MD5 CHECK FAILED ✗")

        print(f"Expected: {expected_md5}")
        print(f"Actual  : {actual_md5}")

        print(
            "\nThe file may be corrupted. "
            "Delete it and download again."
        )


def show_categories(files):

    categories = set()

    for file_info in files:

        filename = file_info["key"]

        if not filename.lower().endswith(".zip"):
            continue

        category = filename.split("_")[0]

        # Handle multi-word categories
        if filename.startswith("Days_and_Time"):
            category = "Days_and_Time"

        elif filename.startswith("Means_of_Transportation"):
            category = "Means_of_Transportation"

        categories.add(category)

    print("\nAvailable INCLUDE categories:\n")

    for category in sorted(categories):
        print(f"  {category}")


def main():

    parser = argparse.ArgumentParser(
        description="Download INCLUDE dataset from Zenodo."
    )

    parser.add_argument(
        "--category",
        type=str,
        help="Download one INCLUDE category."
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all INCLUDE ZIP files."
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available categories."
    )

    args = parser.parse_args()

    DOWNLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    files = get_zenodo_files()

    if args.list:

        show_categories(files)
        return

    selected_files = select_files(
        files,
        category=args.category,
        download_all=args.all
    )

    total_size = sum(
        file_info["size"]
        for file_info in selected_files
    )

    print("\n" + "=" * 80)
    print("INCLUDE DATASET DOWNLOAD")
    print("=" * 80)

    print(f"Files selected : {len(selected_files)}")
    print(f"Total size     : {human_size(total_size)}")
    print(f"Destination    : {DOWNLOAD_DIR}")

    print("\nSelected files:")

    for file_info in selected_files:
        print(
            f"  {file_info['key']} "
            f"({human_size(file_info['size'])})"
        )

    print()

    for file_info in selected_files:
        download_file(file_info)

    print("\n" + "=" * 80)
    print("DOWNLOAD PROCESS FINISHED")
    print("=" * 80)


if __name__ == "__main__":
    main()