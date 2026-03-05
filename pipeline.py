"""
pipeline_combined.py

Handles three cases for each clip of every video:
  1. OLD processed  — clip_dir exists, result.json has OLD format (no video_path/start_sec),
                      and video.mp4 / audio.wav are present inside clip_dir.
                      → Migrates result.json to new format, re-evaluates shortlist rule,
                        adds to index.json. Does NOT re-run SuperGlue.
  2. NEW processed  — clip_dir exists, result.json already has new format fields
                      (video_path, audio_path, start_sec, end_sec).
                      → Skips reprocessing entirely, re-evaluates shortlist rule,
                        adds to index.json.
  3. UNPROCESSED    — clip_dir does not exist or result.json is missing.
                      → Runs the full new pipeline (frames → SuperGlue → motion/audio stats
                         from original files, no clip copies saved).
"""

import json
import subprocess
from pathlib import Path

from utils import (
    extract_frames_from_segment,
    generate_frame_pairs,
    run_superglue,
    run_loftr,
    analyze_motion_loftr,
    analyze_motion,
    compute_audio_stats_segment,
)

# ================= CONFIG =================

BASE_DIR = Path("/fs/nexus-scratch/vyomwal5/vgg/")
VIDEO_DIR = BASE_DIR / "video"
AUDIO_DIR = BASE_DIR / "audio"
OUTPUT_DIR = BASE_DIR / "processed"

SUPERGLUE_SCRIPT = "/nfshomes/vyomwal5/SSL/SuperGluePretrainedNetwork/match_pairs.py"
FRAME_INTERVAL_SEC = 1
FORCE_CPU = False

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

CLIP_SEC = 5
STRIDE = 2

# ================= HELPERS =================

def get_video_duration(video_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    return float(subprocess.check_output(cmd).decode().strip())


def is_new_format(result: dict) -> bool:
    """New-format result.json has video_path, audio_path, start_sec, end_sec."""
    return all(k in result for k in ("video_path", "audio_path", "start_sec", "end_sec"))


def is_old_format(result: dict, clip_dir: Path) -> bool:
    """Old-format result.json lacks timing fields but has video.mp4 / audio.wav in the clip dir."""
    return (
        not is_new_format(result) and
        (clip_dir / "video.mp4").exists() and
        (clip_dir / "audio.wav").exists()
    )


def passes_shortlist_rule(motion: dict, audio_stats: dict) -> bool:
    return (
        motion["left"] > motion["right"] and
        audio_stats["dominant_side"] != "left"
    )


def migrate_old_result(result: dict, clip_dir: Path, video_path: Path, audio_path: Path,
                       start_sec: int, end_sec: int) -> dict:
    """
    Upgrade an old result.json in-place to the new format by injecting
    the path and timing fields. Recomputes audio stats from the original
    file so the result no longer depends on the saved clip audio.wav.
    """
    # Recompute audio stats from original file (old clip audio may differ slightly
    # due to ffmpeg re-encode; using the original is more correct)
    duration_sec = end_sec - start_sec
    audio_stats = compute_audio_stats_segment(audio_path, start_sec=start_sec, duration_sec=duration_sec)

    result.update({
        "video_path": str(video_path),
        "audio_path": str(audio_path),
        "start_sec": start_sec,
        "end_sec": end_sec,
        "audio_spatial": audio_stats,   # overwrite with recomputed stats from original
    })
    return result


# ================= PIPELINE =================

def process_video(video_path: Path, audio_path: Path):
    stem = video_path.stem
    sample_root = OUTPUT_DIR / stem

    duration = get_video_duration(video_path)
    shortlisted_loftr = []
    shortlisted =[]
    start = 0
    while start + CLIP_SEC <= duration:
        end = start + CLIP_SEC
        clip_name = f"clip_{start:04d}_{end:04d}"
        clip_dir = sample_root / clip_name
        result_path = clip_dir / "result.json"  # final result file for both cases (migrated or new)

        try:
            # ── Determine which case we're in ──────────────────────────────
            if result_path.exists():
                with open(result_path) as f:
                    result = json.load(f)

                if is_new_format(result):
                    # ── CASE 2: already in new format, nothing to reprocess ──
                    print(f"  ✅ Already processed (new): {clip_name}")
                    motion = result["visual_motion"]
                    audio_stats = result["audio_spatial"]
                    motion_loftr = result.get("visual_motion_loftr")


                elif is_old_format(result, clip_dir):
                    # ── CASE 1: old format — migrate result.json ─────────────
                    print(f"  🔄 Migrating old format: {clip_name}")
                    result = migrate_old_result(
                        result, clip_dir, video_path, audio_path, start, end
                    )
                    with open(result_path, "w") as f:
                        json.dump(result, f, indent=2)
                    motion = result["visual_motion"]
                    audio_stats = result["audio_spatial"]

                else:
                    # result.json exists but is neither clearly old nor new —
                    # treat as unprocessed and rerun
                    print(f"  ⚠️  Ambiguous result.json, reprocessing: {clip_name}")
                    result = None

            else:
                result = None

            # ── CASE 3: unprocessed (or ambiguous above) ──────────────────
            if result is None:
                print(f"  🚀 Processing new: {clip_name}")
                clip_dir.mkdir(parents=True, exist_ok=True)

                frames_dir = clip_dir / "frames"
                matches_dir = clip_dir / "matches"
                pairs_file = clip_dir / "pairs.txt"

                extract_frames_from_segment(
                    video_path, frames_dir, FRAME_INTERVAL_SEC,
                    start_sec=start, duration_sec=CLIP_SEC
                )
                generate_frame_pairs(frames_dir, pairs_file)

                # loftr
                run_loftr(
                    frames_dir, pairs_file, matches_dir
                )

                motion_loftr = analyze_motion_loftr(matches_dir)

                # superglue
                run_superglue(
                    frames_dir, pairs_file, matches_dir,
                    SUPERGLUE_SCRIPT, FORCE_CPU
                )

                motion = analyze_motion(matches_dir)

                audio_stats = compute_audio_stats_segment(
                    audio_path, start_sec=start, duration_sec=CLIP_SEC
                )

                result = {
                    "sample": clip_name,
                    "video_path": str(video_path),
                    "audio_path": str(audio_path),
                    "start_sec": start,
                    "end_sec": end,
                    "visual_motion_loftr": motion_loftr,
                    "visual_motion": motion,
                    "audio_spatial": audio_stats,
                }
                with open(result_path, "w") as f:
                    json.dump(result, f, indent=2)

            # ── Shortlist rule (same for all cases) ───────────────────────
            if passes_shortlist_rule(motion, audio_stats):
                entry = {
                    "video_id": stem,
                    "clip_id": clip_name,
                    "video_path": str(video_path),
                    "audio_path": str(audio_path),
                    "start_sec": start,
                    "end_sec": end,
                }
                shortlisted.append(entry)
                print(f"  ⭐ Shortlisted: {stem}_{clip_name}")
            if passes_shortlist_rule(motion_loftr, audio_stats):
                entry = {
                    "video_id": stem,
                    "clip_id": clip_name,
                    "video_path": str(video_path),
                    "audio_path": str(audio_path),
                    "start_sec": start,
                    "end_sec": end,
                }
                shortlisted_loftr.append(entry)
                print(f"  ⭐ Shortlisted LoFTR: {stem}_{clip_name}")

        except Exception as e:
            print(f"  ❌ Failed {clip_name}: {e}")

        start += STRIDE
    return shortlisted_loftr, shortlisted


def main():
    shortlist_loftr_path = BASE_DIR / "shortlisted" / "index_loftr.json"
    shortlist_index_path = BASE_DIR / "shortlisted" / "index.json"
    shortlist_index_path.parent.mkdir(parents=True, exist_ok=True)
    shortlist_loftr_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing index so we don't lose entries from previous runs
    if shortlist_index_path.exists():
        with open(shortlist_index_path) as f:
            existing_index = json.load(f)
        # Key by (video_id, clip_id) for deduplication
        existing_keys = {(e["video_id"], e["clip_id"]) for e in existing_index}
        print(f"📋 Loaded existing index with {len(existing_index)} entries")
    else:
        existing_index = []
        existing_keys = set()

    if shortlist_loftr_path.exists():
        with open(shortlist_loftr_path) as f:
            existing_loftr_index = json.load(f)
        # Key by (video_id, clip_id) for deduplication
        existing_loftr_keys = {(e["video_id"], e["clip_id"]) for e in existing_loftr_index}
        print(f"📋 Loaded existing LoFTR index with {len(existing_loftr_index)} entries")
    else:
        existing_loftr_index = []
        existing_loftr_keys = set()

    shortlisted_loftr = []   # fresh list rebuilt for this run; merged with existing below
    shortlisted = []       # final shortlist for this run (after SuperGlue re-eval)

    videos = sorted([v for v in VIDEO_DIR.iterdir() if v.suffix in VIDEO_EXTS],reverse=True)

    for i,video_path in enumerate(videos):
        print(f"\nProcessing video {i+1}/{len(videos)}: {video_path.name}")
        stem = video_path.stem
        audio_path = AUDIO_DIR / f"{stem}.wav"
        if not audio_path.exists():
            print(f"\n⚠️  No matching audio for {stem}, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"📹 {stem}")
        a, b = process_video(video_path, audio_path)
        shortlisted_loftr.extend(a)
        shortlisted.extend(b)

    # Merge: keep existing entries not touched this run, add/update this run's results
    new_keys = {(e["video_id"], e["clip_id"]) for e in shortlisted}
    merged = [e for e in existing_index if (e["video_id"], e["clip_id"]) not in new_keys]
    merged.extend(shortlisted)

    with open(shortlist_index_path, "w") as f:
        json.dump(merged, f, indent=2)

    # Merge: keep existing entries not touched this run, add/update this run's results
    new_keys_loftr = {(e["video_id"], e["clip_id"]) for e in shortlisted_loftr}
    merged_loftr = [e for e in existing_loftr_index if (e["video_id"], e["clip_id"]) not in new_keys_loftr]
    merged_loftr.extend(shortlisted_loftr)

    with open(shortlist_loftr_path, "w") as f:
        json.dump(merged_loftr, f, indent=2)

    print(f"\n✅ index_loftr.json saved → {shortlist_loftr_path}")
    print(f"   Total entries: {len(merged_loftr)}  "
          f"(this run: {len(shortlisted_loftr)} | carried over: {len(merged_loftr) - len(shortlisted_loftr)})")


if __name__ == "__main__":
    main()