from collections import Counter
import json
from pathlib import Path
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import VIDEO_DIR, AUDIO_DIR, PROCESSED_DIR, SHORTLIST_DIR, CHECKPOINT_DIR, VGGSOUND_CSV

def check_label_distribution(manifest_path, processed_dir):
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    sides      = Counter()
    has_camera = 0
    has_object = 0
    neither    = 0
    old_format = 0
    missing    = 0
    none_vm    = 0

    MOTION_THRESHOLD = 1.0

    for entry in manifest:
        result_path = (Path(processed_dir) / entry['video_id'] /
                       entry['clip_id'] / "result.json")

        if not result_path.exists():
            missing += 1
            continue

        try:
            with open(result_path) as f:
                data = json.load(f)
        except json.JSONDecodeError:
            missing += 1
            continue

        vm = data.get('visual_motion_loftr')

        # ── Guard: None or wrong type ─────────────────────────────────────
        if vm is None:
            none_vm += 1
            continue

        if not isinstance(vm, dict):
            old_format += 1
            continue

        # ── Guard: old format (has 'left'/'right' vote counts, not world sides) ──
        if 'camera_world_side' not in vm:
            old_format += 1
            continue

        camera_side     = vm.get('camera_world_side')
        object_side     = vm.get('object_world_side')
        camera_net_flow = vm.get('camera_net_flow') or 0.0
        object_net_flow = vm.get('object_net_flow') or 0.0

        hc = camera_side is not None and abs(camera_net_flow) > MOTION_THRESHOLD
        ho = object_side is not None and abs(object_net_flow) > MOTION_THRESHOLD

        if hc:            has_camera += 1
        if ho:            has_object += 1
        if not hc and not ho: neither += 1

        side = camera_side if hc else (object_side if ho else "ambiguous")
        if side == "ambiguous":
            print(f"⚠️  Ambiguous entry: {result_path}")
        sides[side] += 1

    total = len(manifest)
    print(f"\nTotal manifest entries: {total}")
    print(f"\nData quality:")
    print(f"  Missing/corrupted:  {missing}    ({100*missing/total:.1f}%)")
    print(f"  None visual_motion: {none_vm}    ({100*none_vm/total:.1f}%)")
    print(f"  Old format:         {old_format} ({100*old_format/total:.1f}%)")
    print(f"  New format (valid): {total - missing - none_vm - old_format}")

    valid = total - missing - none_vm - old_format
    if valid == 0:
        print("\n⚠️  No new-format entries found — all results need reprocessing.")
        return

    print(f"\nMotion source breakdown (of valid):")
    print(f"  Has camera motion: {has_camera} ({100*has_camera/valid:.1f}%)")
    print(f"  Has object motion: {has_object} ({100*has_object/valid:.1f}%)")
    print(f"  Neither:           {neither}    ({100*neither/valid:.1f}%)")

    print(f"\nSide distribution (of valid):")
    for side, count in sides.most_common():
        print(f"  {side:>10}: {count:>6} ({100*count/valid:.1f}%)")

check_label_distribution(
    SHORTLIST_DIR / "index_loftr.json",
    PROCESSED_DIR
)