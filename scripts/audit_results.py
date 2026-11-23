from pathlib import Path
import json 
import os
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import VIDEO_DIR, AUDIO_DIR, PROCESSED_DIR, SHORTLIST_DIR, CHECKPOINT_DIR, VGGSOUND_CSV

def safe_load_json(path):
    """
    Load JSON file, handling corruption from double-writes.
    If file has extra data, try to load just the first valid object.
    """
    with open(path, "r") as f:
        content = f.read()
    
    # First try normal load
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # File is corrupted — try to decode just the first valid JSON object
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(content.strip())
        print(f"  ⚠️  Corrupted JSON (extra data) — loaded first object only: {path}")
        return obj
    except json.JSONDecodeError:
        print(f"  ❌  Unrecoverable JSON: {path}")
        return None

def is_new_format(result: dict) -> bool:
    if not result or result.get("visual_motion_loftr") is None:
        return False
    return all(k in result["visual_motion_loftr"] for k in [
        "camera_world_side", "camera_net_flow",
        "object_world_side", "object_net_flow"
    ])

count = 0
n     = 0
corrupted = 0

BASE_DIR   = Path("/fs/nexus-scratch/vyomwal5/vgg/")
VIDEO_DIR  = BASE_DIR / "video"
videos = sorted(
    [v for v in VIDEO_DIR.iterdir() if v.suffix in {".mp4", ".avi", ".mov", ".mkv"}],
    reverse=True
)

for i, video_path in enumerate(videos):
    result_path = PROCESSED_DIR / video_path.stem

    if not result_path.exists():
        continue

    for clip_name in sorted(os.listdir(result_path)):
        n += 1
        clip_path = result_path / clip_name

        if not clip_path.is_dir():
            continue

        clip_result_path = clip_path / "result.json"

        if not clip_result_path.exists():
            count += 1
            continue

        result = safe_load_json(clip_result_path)

        if result is None:
            corrupted += 1
            count += 1
            continue

        if not is_new_format(result):
            count += 1
            continue

print(f"\n{'='*50}")
print(f"Total clips scanned:        {n}")
print(f"Missing/old format:         {count}")
print(f"Unrecoverable corruptions:  {corrupted}")
print(f"New format clips:           {n - count}")