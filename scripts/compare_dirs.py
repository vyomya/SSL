import argparse
import os
import sys
from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import VIDEO_DIR, AUDIO_DIR, PROCESSED_DIR, SHORTLIST_DIR, CHECKPOINT_DIR, VGGSOUND_CSV

def gather_files(path):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Directory not found: {path}")

    root_path = os.path.abspath(path)
    files = set()
    for root, _, filenames in os.walk(root_path):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            rel_path = os.path.relpath(file_path, root_path)
            files.add(rel_path)
    return files


def compare_paths(audio_path, video_path):
    audio_files = gather_files(audio_path)
    video_files = gather_files(video_path)

    common_files = audio_files.intersection(video_files)
    audio_only = audio_files.difference(video_files)
    video_only = video_files.difference(audio_files)

    print(f"Audio path: {audio_path}")
    print(f"Video path: {video_path}")
    print(f"Common files: {len(common_files)}")
    print(f"Audio-only files: {len(audio_only)}")
    print(f"Video-only files: {len(video_only)}")

    return common_files, audio_only, video_only


def main():
    parser = argparse.ArgumentParser(
        description="Compare files in two folders and report common and different file counts."
    )
    parser.add_argument("audio_path", help="Path to the audio folder")
    parser.add_argument("video_path", help="Path to the video folder")
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the list of common and different files",
    )
    args = parser.parse_args()

    try:
        common_files, audio_only, video_only = compare_paths(
            args.audio_path, args.video_path
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    if args.list:
        if common_files:
            print("\nCommon files:")
            for path in sorted(common_files):
                print(path)
        if audio_only:
            print("\nFiles only in audio folder:")
            for path in sorted(audio_only):
                print(path)
        if video_only:
            print("\nFiles only in video folder:")
            for path in sorted(video_only):
                print(path)


if __name__ == "__main__":
    main()
