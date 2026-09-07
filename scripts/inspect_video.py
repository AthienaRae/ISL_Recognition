from pathlib import Path
import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "include"
    / "extracted"
    / "Greetings"
    / "48. Hello"
    / "MVI_0029.MOV"
)


def main():

    print("=" * 70)
    print("INCLUDE Video Inspection")
    print("=" * 70)

    print(f"\nVideo:\n{VIDEO_PATH}")

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"\nVideo not found:\n{VIDEO_PATH}"
        )

    capture = cv2.VideoCapture(str(VIDEO_PATH))

    if not capture.isOpened():
        raise RuntimeError(
            "\nOpenCV could not open the video."
        )

    width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    frame_count = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    duration = (
        frame_count / fps
        if fps > 0
        else 0
    )

    print("\nVideo properties")
    print("-" * 70)

    print(f"Resolution   : {width} x {height}")
    print(f"FPS          : {fps:.2f}")
    print(f"Frame count  : {frame_count}")
    print(f"Duration     : {duration:.2f} seconds")

    # Test actual frame decoding

    decoded_frames = 0
    first_frame = None

    while True:

        success, frame = capture.read()

        if not success:
            break

        if first_frame is None:
            first_frame = frame.copy()

        decoded_frames += 1

    capture.release()

    print(f"Decoded      : {decoded_frames} frames")

    if decoded_frames == frame_count:
        print("\nFrame decoding: OK ✓")
    else:
        print(
            "\nWARNING: Reported and decoded "
            "frame counts differ."
        )

    if first_frame is not None:

        output_dir = (
            PROJECT_ROOT
            / "results"
            / "video_inspection"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path = (
            output_dir
            / "hello_first_frame.jpg"
        )

        cv2.imwrite(
            str(output_path),
            first_frame
        )

        print(
            f"\nFirst frame saved to:\n"
            f"{output_path}"
        )

    print("\n" + "=" * 70)
    print("Inspection complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()