import soundfile as sf
from pathlib import Path
from collections import Counter
import sys

def check_stereo(audio_dir):
    audio_dir = Path(audio_dir)
    files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3")) + \
            list(audio_dir.glob("*.flac"))

    print(f"Total audio files found: {len(files)}")
    if not files:
        print("No audio files found. Check your path.")
        return

    channel_counts = Counter()
    sample_rates   = Counter()
    errors         = []

    for i, f in enumerate(files):
        if i % 500 == 0:
            print(f"  Processing {i}/{len(files)}...", flush=True)
        try:
            info = sf.info(str(f))
            channel_counts[info.channels] += 1
            sample_rates[info.samplerate] += 1
        except Exception as e:
            errors.append((f.name, str(e)))

    print(f"\n{'='*40}")
    print(f"Channel distribution:")
    for ch, count in sorted(channel_counts.items()):
        label = {1: "Mono", 2: "Stereo"}.get(ch, f"{ch}-channel")
        print(f"  {label} ({ch}ch): {count:>6} files  ({100*count/len(files):.1f}%)")

    print(f"\nSample rate distribution:")
    for sr, count in sorted(sample_rates.items()):
        print(f"  {sr} Hz: {count:>6} files  ({100*count/len(files):.1f}%)")

    if errors:
        print(f"\nFailed to read {len(errors)} files:")
        for name, err in errors[:10]:
            print(f"  {name}: {err}")

    stereo_count = channel_counts.get(2, 0)
    mono_count   = channel_counts.get(1, 0)
    print(f"\n{'='*40}")
    print(f"SUMMARY:")
    print(f"  Stereo files: {stereo_count} ({100*stereo_count/len(files):.1f}%)")
    print(f"  Mono files:   {mono_count}   ({100*mono_count/len(files):.1f}%)")

    if stereo_count == 0:
        print("\n⚠️  All files are MONO — IID supervision is not possible.")
        print("   Consider Option B (egomotion only) or Option C (synthesized stereo).")
    elif stereo_count < len(files) * 0.5:
        print(f"\n⚠️  Less than 50% stereo — IID supervision will only apply to {stereo_count} files.")
    else:
        print(f"\n✅  Majority stereo — IID supervision is viable.")

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "/fs/nexus-scratch/vyomwal5/vgg/audio"
    check_stereo(folder)