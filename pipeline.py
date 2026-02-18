import json
from pathlib import Path
from utils import (
    extract_frames,
    generate_frame_pairs,
    run_superglue,
    analyze_motion,
    compute_audio_stats,
    split_video_audio
)
import shutil
# ================= CONFIG =================

BASE_DIR = Path("/Users/vyomagarwal/Documents/projects/sound_localization/VGGSound/audioset_data")
VIDEO_DIR = BASE_DIR / "video"
AUDIO_DIR = BASE_DIR / "audio"
OUTPUT_DIR = BASE_DIR / "processed"

SUPERGLUE_SCRIPT = "/Users/vyomagarwal/Documents/projects/sound_localization/SuperGluePretrainedNetwork/match_pairs.py"
FRAME_INTERVAL_SEC = 1
FORCE_CPU = False

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
AUDIO_EXTS = {".wav"}

# ================= PIPELINE =================

def main():
    shortlisted_video_dir = BASE_DIR / "shortlisted" / "video"
    shortlisted_audio_dir = BASE_DIR / "shortlisted" / "audio"
    shortlisted_video_dir.mkdir(parents=True, exist_ok=True)
    shortlisted_audio_dir.mkdir(parents=True, exist_ok=True)
    videos = [v for v in VIDEO_DIR.iterdir() if v.suffix in VIDEO_EXTS]
    for video_path in videos:
        stem = video_path.stem
        audio_path = AUDIO_DIR / f"{stem}.wav"
        if not audio_path.exists():
            continue

        print(f"\n🚀 Processing {stem}")

        sample_root = OUTPUT_DIR / stem
        clip_dirs = split_video_audio(
            video_path, audio_path, sample_root, clip_sec=5,stride=2
        )

        for clip_dir in clip_dirs:
            try:
                frames_dir = clip_dir / "frames"
                matches_dir = clip_dir / "matches"
                pairs_file = clip_dir / "pairs.txt"

                extract_frames(clip_dir / "video.mp4", frames_dir, FRAME_INTERVAL_SEC)
                generate_frame_pairs(frames_dir, pairs_file)

                run_superglue(
                    frames_dir, pairs_file, matches_dir,
                    SUPERGLUE_SCRIPT, FORCE_CPU
                )

                motion = analyze_motion(matches_dir)
                audio_stats = compute_audio_stats(clip_dir / "audio.wav")

                result = {
                    "sample": clip_dir.name,
                    "visual_motion": motion,
                    "audio_spatial": audio_stats
                }

                result_path = clip_dir / "result.json"
                with open(result_path, "w") as f:
                    json.dump(result, f, indent=2)

                # -------- SHORTLIST RULE --------
                if (
                    motion["left"] > motion["right"] and
                    audio_stats["dominant_side"] != "left"
                ):
                    out_name = f"{stem}_{clip_dir.name}"
                    shutil.copy(
                        clip_dir / "video.mp4",
                        shortlisted_video_dir / f"{out_name}.mp4"
                    )
                    shutil.copy(
                        clip_dir / "audio.wav",
                        shortlisted_audio_dir / f"{out_name}.wav"
                    )

                    print(f"⭐ Shortlisted: {out_name}")

            except Exception as e:
                print(f"❌ Failed {clip_dir.name}: {e}")



if __name__ == "__main__":
    main()
