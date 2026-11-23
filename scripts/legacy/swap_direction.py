import json
import os
import argparse


def swap_directions(origin_path: str):
    """
    Traverse origin_path -> folders -> subfolders -> subsubfolders,
    find result.json files, and swap left/right and up/down in visual_motion_loftr.
    """
    modified_count = 0
    skipped_count = 0

    # Walk 3 levels deep: origin -> level1 -> level2 -> level3
    for level1 in sorted(os.listdir(origin_path)):
        level1_path = os.path.join(origin_path, level1)
        if not os.path.isdir(level1_path):
            continue

        for level2 in sorted(os.listdir(level1_path)):
            level2_path = os.path.join(level1_path, level2)
            if not os.path.isdir(level2_path):
                continue


            result_json_path = os.path.join(level2_path, "result.json")
            if not os.path.isfile(result_json_path):
                print(f"  [SKIP] No result.json in: {level2_path}")
                skipped_count += 1
                continue

            try:
                with open(result_json_path, "r") as f:
                    data = json.load(f)

                if "visual_motion_loftr" not in data:
                    print(f"  [SKIP] 'visual_motion_loftr' key missing in: {result_json_path}")
                    skipped_count += 1
                    continue

                vml = data["visual_motion_loftr"]

                # Swap left <-> right and up <-> down
                original = {
                    "left": vml.get("left"),
                    "right": vml.get("right"),
                    "up": vml.get("up"),
                    "down": vml.get("down"),
                }

                vml["left"]  = original["right"]
                vml["right"] = original["left"]
                vml["up"]    = original["down"]
                vml["down"]  = original["up"]

                data["visual_motion_loftr"] = vml

                with open(result_json_path, "w") as f:
                    json.dump(data, f, indent=2)

                print(f"  [OK] Swapped directions in: {result_json_path}")
                print(f"       left {original['left']} -> {vml['left']}  |  "
                        f"right {original['right']} -> {vml['right']}  |  "
                        f"up {original['up']} -> {vml['up']}  |  "
                        f"down {original['down']} -> {vml['down']}")
                modified_count += 1

            except (json.JSONDecodeError, IOError) as e:
                print(f"  [ERROR] Could not process {result_json_path}: {e}")
                skipped_count += 1

    print(f"\nDone. Modified: {modified_count} file(s), Skipped/Not found: {skipped_count} folder(s).")


def main():
    parser = argparse.ArgumentParser(
        description="Swap left/right and up/down in visual_motion_loftr inside result.json files."
    )
    parser.add_argument(
        "origin_path",
        help="Root directory to search (3 levels deep for result.json files)."
    )
    args = parser.parse_args()

    origin_path = os.path.abspath(args.origin_path)

    if not os.path.isdir(origin_path):
        print(f"[ERROR] origin_path does not exist or is not a directory: {origin_path}")
        return

    print(f"Scanning: {origin_path}\n")
    swap_directions(origin_path)


if __name__ == "__main__":
    main()