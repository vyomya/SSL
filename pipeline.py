"""
pipeline.py

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
import os
from utils import (
    extract_frames_from_segment,
    generate_frame_pairs,
    run_superglue,
    run_loftr,
    analyze_motion_loftr,
    analyze_motion_superglue,
    compute_audio_stats_segment,
)

# ================= CONFIG =================

from config import VIDEO_DIR, AUDIO_DIR, PROCESSED_DIR as OUTPUT_DIR, SHORTLIST_DIR, SUPERGLUE_SCRIPT
RECOMPUTE_MOTION = False

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

def write_json_atomic(path, obj):
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)

def is_new_format(result: dict) -> bool:
    vm = result.get("visual_motion_loftr") if result else None
    return isinstance(vm, dict) and all(k in vm for k in (
        "camera_world_side", "camera_net_flow", "object_world_side", "object_net_flow"))

def is_old_format(result: dict, clip_dir: Path) -> bool:
    """Old-format result.json lacks timing fields but has video.mp4 / audio.wav in the clip dir."""
    return (
        not is_new_format(result) and
        (clip_dir / "video.mp4").exists() and
        (clip_dir / "audio.wav").exists()
    )

def passes_shortlist_rule(motion: dict, audio_stats: dict) -> bool:
    """
    Filter samples for training using corrected world-space motion labels.
    
    motion dict expected keys (from _compute_motion_with_ransac):
        camera_world_side : "left" | "right" | None
        camera_net_flow   : float | None   (raw point flow, + = points flowed right)
        object_world_side : "left" | "right" | None
        object_net_flow   : float | None
        inlier_ratio      : float
        num_inliers       : int
        num_outliers      : int
        mean_confidence   : float
        total_matches     : int

    audio_stats dict expected keys (from compute_audio_stats_segment):
        dominant_side     : "left" | "right" | "ambiguous"
        abs_mean_iid_db   : float
        iid_consistency   : float
    """

    if motion is None or audio_stats is None:
        return False

    # ── 1. Audio quality gates — check first, cheapest rejection ──────────

    if audio_stats.get("dominant_side") == "ambiguous":
        return False

    # IID must be a real signal, not floating point noise
    # if audio_stats.get("abs_mean_iid_db", 0.0) < :  # e.g. 1.0 dB
    #     return False

    # IID must be consistent across time windows
    if audio_stats.get("iid_consistency", 0.0) < 0.6:
        return False

    audio_side = audio_stats["dominant_side"]

    # ── 2. Match quality gates ─────────────────────────────────────────────

    if motion.get("mean_confidence", 0.0) < 0.5:
        return False

    if motion.get("total_matches", 0) < 50:
        return False

    # ── 3. Determine what is moving ────────────────────────────────────────

    camera_net_flow = motion.get("camera_net_flow")
    object_net_flow = motion.get("object_net_flow")
    camera_side     = motion.get("camera_world_side")
    object_side     = motion.get("object_world_side")
    inlier_ratio    = motion.get("inlier_ratio", 0.0)
    num_inliers     = motion.get("num_inliers", 0)
    num_outliers    = motion.get("num_outliers", 0)

    has_camera = (
        camera_side is not None and
        camera_net_flow is not None and
        abs(camera_net_flow) > 1.0        # e.g. 2.0 pixels
    )

    has_object = (
        object_side is not None and
        object_net_flow is not None and 
        abs(object_net_flow) > 1.0
    )

    # Nothing moved significantly
    if not has_camera and not has_object:
        return False

    # ── 4. Case-based direction agreement ─────────────────────────────────

    # Case 1: Camera motion only
    if has_camera and not has_object:
        if inlier_ratio < 0.5 or num_inliers < 50:
            return False
        return camera_side == audio_side

    # Case 2: Object motion only (static camera)
    if has_object and not has_camera:
        if num_outliers < 30:
            return False
        return object_side == audio_side

    # Case 3: Both moving
    if has_camera and has_object:

        # 3a: All three agree — strongest signal
        if camera_side == object_side == audio_side:
            return True

        # 3b: Camera agrees, object disagrees — trust camera if background dominates
        if camera_side == audio_side and object_side != audio_side:
            return inlier_ratio > 0.75

        # 3c: Object agrees, camera disagrees — trust object if it's strong enough
        if object_side == audio_side and camera_side != audio_side:
            foreground_ratio = num_outliers / max(num_inliers + num_outliers, 1)
            object_stronger  = abs(object_net_flow) > abs(camera_net_flow)
            return foreground_ratio > 0.3 and object_stronger

        # 3d: Neither agrees with audio
        return False

    return False

    


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
            motion = motion_loftr = audio_stats = None
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
                    if RECOMPUTE_MOTION and (clip_dir / "matches").exists():
                        motion = analyze_motion_superglue(clip_dir / "matches")
                        motion_loftr = analyze_motion_loftr(clip_dir / "matches")
                        result["visual_motion"], result["visual_motion_loftr"] = motion, motion_loftr
                        write_json_atomic(result_path, result)

                elif is_old_format(result, clip_dir):
                    # ── CASE 1: old format — migrate result.json ─────────────
                    print(f"  🔄 Migrating old format: {clip_name}")
                    result = migrate_old_result(
                        result, clip_dir, video_path, audio_path, start, end
                    )
                    write_json_atomic(result_path, result)
                    motion = result["visual_motion"]
                    audio_stats = result["audio_spatial"]
                    motion_loftr = result.get("visual_motion_loftr")
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

                motion = analyze_motion_superglue(matches_dir)

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
                write_json_atomic(result_path, result)

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
    shortlist_loftr_path = SHORTLIST_DIR / "index_loftr.json"
    shortlist_index_path = SHORTLIST_DIR / "index.json"
    shortlist_index_path.parent.mkdir(parents=True, exist_ok=True)
    shortlist_loftr_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing index so we don't lose entries from previous runs

    shortlisted_loftr = []   # fresh list rebuilt for this run; merged with existing below
    shortlisted = []       # final shortlist for this run (after SuperGlue re-eval)

    videos = sorted([v for v in VIDEO_DIR.iterdir() if v.suffix in VIDEO_EXTS])
    # videos = videos[3500:5500]  # TEMP LIMIT for testing
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

    write_json_atomic(shortlist_index_path, shortlisted)
    write_json_atomic(shortlist_loftr_path, shortlisted_loftr)

    print(f"✅ index.json       {len(shortlisted):>6} clips → {shortlist_index_path}")
    print(f"✅ index_loftr.json {len(shortlisted_loftr):>6} clips → {shortlist_loftr_path}")

if __name__ == "__main__":
    main()