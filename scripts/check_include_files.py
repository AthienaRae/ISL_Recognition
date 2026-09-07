import sys
import requests

ZENODO_RECORD_ID = "4010759"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"


def human_readable_size(size_bytes):
    size = float(size_bytes)

    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024

    return f"{size:.2f} PB"


def main():
    print("=" * 80)
    print("INCLUDE Dataset - Zenodo File Inspector")
    print("=" * 80)

    print(f"\nRecord ID : {ZENODO_RECORD_ID}")
    print(f"API URL   : {ZENODO_API_URL}")
    print("\nContacting Zenodo...")

    try:
        response = requests.get(ZENODO_API_URL, timeout=60)
        response.raise_for_status()

    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not connect to Zenodo.")
        sys.exit(1)

    except requests.exceptions.Timeout:
        print("\nERROR: Zenodo request timed out.")
        sys.exit(1)

    except requests.exceptions.HTTPError as error:
        print(f"\nHTTP ERROR: {error}")
        sys.exit(1)

    except requests.exceptions.RequestException as error:
        print(f"\nREQUEST ERROR: {error}")
        sys.exit(1)

    record = response.json()

    metadata = record.get("metadata", {})
    files = record.get("files", [])

    print("\nConnection successful.")

    print(f"\nDataset          : {metadata.get('title', 'Unknown')}")
    print(f"Publication date : {metadata.get('publication_date', 'Unknown')}")
    print(f"Number of files  : {len(files)}")

    if not files:
        print("\nNo files are exposed by this Zenodo record.")
        return

    print("\n" + "-" * 110)
    print(
        f"{'No.':<5}"
        f"{'Filename':<55}"
        f"{'Size':<15}"
        f"{'Checksum'}"
    )
    print("-" * 110)

    total_size = 0

    for index, file_info in enumerate(files, start=1):
        filename = file_info.get("key", "Unknown")
        size_bytes = file_info.get("size", 0)
        checksum = file_info.get("checksum", "N/A")

        total_size += size_bytes

        print(
            f"{index:<5}"
            f"{filename:<55}"
            f"{human_readable_size(size_bytes):<15}"
            f"{checksum}"
        )

    print("-" * 110)
    print(
        f"\nTotal exposed size: "
        f"{human_readable_size(total_size)}"
    )

    print("\n" + "=" * 80)
    print("DOWNLOAD LINKS")
    print("=" * 80)

    for index, file_info in enumerate(files, start=1):
        filename = file_info.get("key", "Unknown")
        links = file_info.get("links", {})

        download_url = links.get("content") or links.get("self")

        print(f"\n[{index}] {filename}")
        print(f"    {download_url}")

    print("\nInspection complete.")


if __name__ == "__main__":
    main()